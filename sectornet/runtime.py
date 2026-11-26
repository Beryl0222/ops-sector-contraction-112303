"""计划执行：前向收缩、反向算子与进程临时张量预算追踪。

临时张量指执行过程中物化的副本与中间结果；基础块本身的持久
存储不计入。所有物化都经过 MemTracker 记账，超出预算即抛出
BudgetExceeded，由上层把对应请求标记为受限。
"""
from __future__ import annotations

from contextlib import contextmanager

import numpy as np

from .model import DTYPE
from .plan import Contract, Leaf, Slice


class BudgetExceeded(RuntimeError):
    """进程临时张量预算不足。"""


class MemTracker:
    """记录当前存活临时字节与峰值，超限即拒绝。"""

    def __init__(self, limit_bytes):
        self.limit = int(limit_bytes)
        self.live = 0
        self.peak = 0

    def alloc(self, nbytes):
        nbytes = int(nbytes)
        self.live += nbytes
        self.peak = max(self.peak, self.live)
        if self.live > self.limit:
            self.live -= nbytes
            raise BudgetExceeded(
                "临时张量预算不足：需要 %d 字节，预算 %d 字节" % (self.live + nbytes, self.limit)
            )

    def free(self, nbytes):
        self.live -= int(nbytes)

    @contextmanager
    def hold(self, nbytes):
        self.alloc(nbytes)
        try:
            yield
        finally:
            self.free(nbytes)


def contract_arrays(left_axes, left, right_axes, right):
    """按轴名收缩两个稠密数组，返回 (输出轴顺序, 结果)。"""
    common = [axis for axis in left_axes if axis in right_axes]
    left_pos = [left_axes.index(axis) for axis in common]
    right_pos = [right_axes.index(axis) for axis in common]
    out = np.tensordot(left, right, axes=(left_pos, right_pos))
    out_axes = [a for a in left_axes if a not in common] + [
        a for a in right_axes if a not in common
    ]
    return tuple(out_axes), out


def align_axes(axes, array, target_axes):
    """把数组转置到目标轴顺序（视图，不复制）。"""
    axes = tuple(axes)
    target_axes = tuple(target_axes)
    if axes == target_axes:
        return array
    order = [axes.index(axis) for axis in target_axes]
    return np.transpose(array, order)


def keep_slice_lists(network, node, key, keep, slices):
    """各位置在完整视图块中的有效坐标：先保留坐标、再取分段区间。"""
    keep = dict(keep)
    slices = dict(slices)
    extents = network.node_extents(node, key)
    lists = []
    for pos, axis in enumerate(node.axes):
        coords = list(range(extents[pos]))
        if axis in keep:
            coords = [coords[c] for c in keep[axis]]
        if axis in slices:
            start, stop = slices[axis]
            coords = coords[start:stop]
        lists.append(tuple(coords))
    return tuple(lists)


def _is_contiguous(coords):
    return list(coords) == list(range(coords[0], coords[0] + len(coords))) if coords else True


class LeafContext:
    """叶子物化上下文：按节点扇区键从当前块存储取数并施加视图。"""

    def __init__(self, network, store):
        self.network = network
        self.store = store

    def materialize(self, plan: Leaf):
        """返回 (数组, 临时字节)。纯视图临时字节为 0。"""
        node = self.network.node_map[plan.node]
        base_key = node.base_key(plan.key)
        state = self.store.state(node.own, base_key)
        if state.kind != "data":
            raise ValueError("叶子 %s/%s 无数据（%s）" % (node.own, base_key, state.kind))
        base = state.data
        perm = node.permutation()
        arr = np.transpose(base, perm) if perm != tuple(range(len(perm))) else base
        # 视图坐标（重排后各位置）与保留坐标/分段区间复合。
        view_lists = self.network.view_lists(node, plan.key)
        ks_lists = keep_slice_lists(self.network, node, plan.key, plan.keep, plan.slices)
        composed = tuple(
            tuple(view_lists[p][c] for c in ks_lists[p]) for p in range(len(node.axes))
        )
        if all(_is_contiguous(coords) for coords in composed):
            index = tuple(
                slice(coords[0], coords[0] + len(coords)) if coords else slice(0, 0)
                for coords in composed
            )
            return arr[index], 0
        out = arr[np.ix_(*[np.asarray(coords, dtype=np.intp) for coords in composed])]
        return out, out.nbytes

    def leaf_full_shape(self, node_id, key):
        node = self.network.node_map[node_id]
        return self.network.node_extents(node, key)


def execute_plan(plan, ctx, tracker):
    """执行计划，返回 (输出轴顺序, 数组, 结果占用的临时字节)。

    调用方负责在结果用完后 tracker.free(返回的临时字节)。
    """
    if isinstance(plan, Leaf):
        array, temp = ctx.materialize(plan)
        if temp:
            tracker.alloc(temp)
        return plan.axes, array, temp
    if isinstance(plan, Contract):
        # 物化顺序按计划记录，收缩始终按 (left, right) 保证轴顺序确定。
        if plan.left_first:
            left_axes, left_arr, left_held = execute_plan(plan.left, ctx, tracker)
            right_axes, right_arr, right_held = execute_plan(plan.right, ctx, tracker)
        else:
            right_axes, right_arr, right_held = execute_plan(plan.right, ctx, tracker)
            left_axes, left_arr, left_held = execute_plan(plan.left, ctx, tracker)
        out_axes, out = contract_arrays(left_axes, left_arr, right_axes, right_arr)
        tracker.alloc(out.nbytes)
        if left_held:
            tracker.free(left_held)
        if right_held:
            tracker.free(right_held)
        return out_axes, out, out.nbytes
    if isinstance(plan, Slice):
        out = np.zeros(plan.shape, dtype=DTYPE)
        tracker.alloc(out.nbytes)
        axis_pos = plan.out_axes.index(plan.axis) if plan.combine == "concat" else None
        try:
            for chunk, bounds in zip(plan.chunks, plan.ranges):
                chunk_axes, chunk_arr, chunk_held = execute_plan(chunk, ctx, tracker)
                aligned = align_axes(chunk_axes, chunk_arr, plan.out_axes)
                if plan.combine == "concat":
                    index = [slice(None)] * len(plan.out_axes)
                    index[axis_pos] = slice(bounds[0], bounds[1])
                    out[tuple(index)] = aligned
                else:
                    # sum 模式：分段轴在段内被收缩，各段输出形状一致，直接累加。
                    out += aligned
                if chunk_held:
                    tracker.free(chunk_held)
        except BaseException:
            tracker.free(out.nbytes)
            raise
        return plan.out_axes, out, out.nbytes
    raise TypeError("未知的计划算子：%r" % (plan,))


def einsum_to(tracker, left_axes, left, right_axes, right, out_axes):
    """预算内的单步二元收缩：结果超出余量时自动分段。"""
    out_axes = tuple(out_axes)
    left_map = dict(zip(left_axes, left.shape))
    right_map = dict(zip(right_axes, right.shape))
    extents = {}
    extents.update(left_map)
    extents.update(right_map)
    out_shape = tuple(extents[a] for a in out_axes)
    need = int(np.prod(out_shape, dtype=np.int64)) * out_dtype_size()
    if tracker.live + need <= tracker.limit:
        axes, out = contract_arrays(tuple(left_axes), left, tuple(right_axes), right)
        out = align_axes(axes, out, out_axes)
        tracker.alloc(out.nbytes)
        return out
    # 分段：优先切输出轴（拼接），其次切公共收缩轴（求和）。
    candidates = sorted(
        (a for a in set(left_axes) | set(right_axes) if extents[a] > 1),
        key=lambda a: -extents[a],
    )
    for axis in candidates:
        in_out = axis in out_axes
        in_left = axis in left_axes
        in_right = axis in right_axes
        if not in_out and not (in_left and in_right):
            continue  # 单边自由轴且不在输出：切它无意义
        count = 2
        extent = extents[axis]
        while count <= extent:
            count = min(count, extent)
            ranges = _ranges(extent, count)
            try:
                out = np.zeros(out_shape, dtype=DTYPE)
                tracker.alloc(out.nbytes)
            except BudgetExceeded:
                if count == extent:
                    break
                count = min(count * 2, extent)
                continue
            try:
                for start, stop in ranges:
                    l = _slice_array(left, left_axes, axis, start, stop)
                    r = _slice_array(right, right_axes, axis, start, stop)
                    piece = einsum_to(tracker, left_axes, l, right_axes, r, out_axes)
                    if in_out:
                        pos = out_axes.index(axis)
                        index = [slice(None)] * len(out_axes)
                        index[pos] = slice(start, stop)
                        out[tuple(index)] = piece
                    else:
                        out += piece
                    tracker.free(piece.nbytes)
            except BudgetExceeded:
                tracker.free(out.nbytes)
                if count == extent:
                    break
                count = min(count * 2, extent)
                continue
            return out
        # 该轴行不通，尝试下一条轴。
    raise BudgetExceeded(
        "临时张量预算不足：单步收缩需要 %d 字节结果，当前余量 %d 字节"
        % (need, tracker.limit - tracker.live)
    )


def out_dtype_size():
    return np.dtype(DTYPE).itemsize


def _ranges(extent, count):
    base, extra = divmod(extent, count)
    ranges = []
    start = 0
    for k in range(count):
        stop = start + base + (1 if k < extra else 0)
        ranges.append((start, stop))
        start = stop
    return ranges


def _slice_array(array, axes, axis, start, stop):
    if axis not in axes:
        return array
    pos = axes.index(axis)
    index = [slice(None)] * len(axes)
    index[pos] = slice(start, stop)
    return array[tuple(index)]


def _plan_out_nbytes(plan):
    return int(np.prod(plan.shape, dtype=np.int64)) * out_dtype_size() if plan.shape else out_dtype_size()


def slice_plan(plan, axis, bounds):
    """生成“只计算输出沿 axis 的 [start, stop) 段”的子计划。

    反向梯度结果过大时用它把反传分段；叶子的 slices 记录了
    相对完整视图块的区间，sink 据此把梯度散布回原坐标。
    """
    start, stop = bounds
    if isinstance(plan, Leaf):
        pos = plan.axes.index(axis)
        slices = dict(plan.slices)
        prev = slices.get(axis, (0, plan.shape[pos]))
        slices[axis] = (prev[0] + start, prev[0] + stop)
        shape = list(plan.shape)
        shape[pos] = stop - start
        temp = plan.temp_bytes
        if temp:
            temp = temp * (stop - start) // max(1, plan.shape[pos])
        return Leaf(
            node=plan.node,
            key=plan.key,
            axes=plan.axes,
            shape=tuple(shape),
            keep=plan.keep,
            slices=tuple(sorted(slices.items())),
            temp_bytes=temp,
        )
    if isinstance(plan, Contract):
        left = slice_plan(plan.left, axis, bounds) if axis in plan.left.axes else plan.left
        right = slice_plan(plan.right, axis, bounds) if axis in plan.right.axes else plan.right
        shape = list(plan.shape)
        shape[plan.out_axes.index(axis)] = stop - start
        return Contract(left, right, plan.out_axes, tuple(shape), plan.left_first)
    if isinstance(plan, Slice):
        shape = list(plan.shape)
        shape[plan.out_axes.index(axis)] = stop - start
        if axis != plan.axis:
            chunks = tuple(slice_plan(c, axis, bounds) for c in plan.chunks)
            return Slice(plan.axis, plan.combine, plan.ranges, chunks, plan.out_axes, tuple(shape))
        if plan.combine != "concat":
            raise BudgetExceeded("无法沿求和分段轴继续切分反向计划")
        chunks = []
        ranges = []
        for (cs, ce), chunk in zip(plan.ranges, plan.chunks):
            lo, hi = max(cs, start), min(ce, stop)
            if lo < hi:
                chunks.append(slice_plan(chunk, axis, (lo - cs, hi - cs)))
                ranges.append((lo - start, hi - start))
        return Slice(plan.axis, plan.combine, tuple(ranges), tuple(chunks), plan.out_axes, tuple(shape))
    raise TypeError("未知的计划算子：%r" % (plan,))


def _backward_side(child_plan, z_axes, grad, other_axes, other, ctx, tracker, sink):
    """计算 child 的梯度并反传；梯度结果过大时沿其输出轴分段。"""
    child_axes = child_plan.axes
    need = _plan_out_nbytes(child_plan)
    if tracker.live + need <= tracker.limit:
        grad_child = einsum_to(tracker, z_axes, grad, other_axes, other, child_axes)
        try:
            backward_plan(child_plan, grad_child, ctx, tracker, sink)
        finally:
            tracker.free(grad_child.nbytes)
        return
    # 梯度结果本身超预算：沿 child 输出轴中最大的一条分段。
    extents = dict(zip(child_axes, child_plan.shape))
    axis = max(child_axes, key=lambda a: extents[a])
    extent = extents[axis]
    avail = max(1, tracker.limit - tracker.live)
    count = min(extent, max(2, -(-need // avail)))
    if count == 1 and need > avail:
        raise BudgetExceeded(
            "临时张量预算不足：反向梯度需要 %d 字节，余量 %d 字节" % (need, avail)
        )
    for start, stop in _ranges(extent, count):
        chunk_plan = slice_plan(child_plan, axis, (start, stop))
        if axis in other_axes:
            z_piece, other_piece = grad, _slice_array(other, other_axes, axis, start, stop)
        else:
            z_piece, other_piece = _slice_array(grad, z_axes, axis, start, stop), other
        grad_piece = einsum_to(
            tracker, z_axes, z_piece, other_axes, other_piece, chunk_plan.axes
        )
        try:
            backward_plan(chunk_plan, grad_piece, ctx, tracker, sink)
        finally:
            tracker.free(grad_piece.nbytes)


def backward_plan(plan, grad, ctx, tracker, sink):
    """对计划反向传播：grad 为计划输出的上游梯度（按 plan 输出轴顺序）。

    前向中间值按需重新物化（计划确定，重算结果一致），叶子的梯度
    通过 sink 回调收集。所有临时占用经 tracker 记账。
    """
    if isinstance(plan, Leaf):
        sink(plan, grad)
        return
    if isinstance(plan, Contract):
        left_axes, left, left_held = execute_plan(plan.left, ctx, tracker)
        right_axes, right, right_held = execute_plan(plan.right, ctx, tracker)
        try:
            _backward_side(plan.left, plan.out_axes, grad, right_axes, right, ctx, tracker, sink)
            _backward_side(plan.right, plan.out_axes, grad, left_axes, left, ctx, tracker, sink)
        finally:
            if left_held:
                tracker.free(left_held)
            if right_held:
                tracker.free(right_held)
        return
    if isinstance(plan, Slice):
        axis_pos = plan.out_axes.index(plan.axis) if plan.combine == "concat" else None
        for chunk, bounds in zip(plan.chunks, plan.ranges):
            if plan.combine == "concat":
                index = [slice(None)] * len(plan.out_axes)
                index[axis_pos] = slice(bounds[0], bounds[1])
                piece = grad[tuple(index)]
            else:
                piece = grad
            piece = align_axes(plan.out_axes, piece, chunk.axes)
            backward_plan(chunk, piece, ctx, tracker, sink)
        return
    raise TypeError("未知的计划算子：%r" % (plan,))
