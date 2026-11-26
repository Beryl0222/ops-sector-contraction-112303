"""预算内收缩计划：叶子、二元收缩与分段算子及其临时内存估计。

计划只描述形状与切法，不持有数据；同一合法输出分量的值不随
路径或分段切法变化。计划可序列化，作为输出证明的一部分保存，
之后可用旧输入复得。

- ``Leaf``：物化一个节点视图块（可带输出轴保留坐标与分段切片）。
- ``Contract``：二元收缩，记录输出轴顺序与执行先后。
- ``Slice``：沿一条轴分块计算；轴在输出中则拼接（concat），
  否则该轴为被收缩轴，分段求和（sum）。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DTYPE = np.float64
ITEMSIZE = np.dtype(DTYPE).itemsize


def nbytes(shape):
    size = ITEMSIZE
    for dim in shape:
        size *= int(dim)
    return size


@dataclass(frozen=True)
class Leaf:
    """物化节点视图块：keep 为输出轴保留坐标，slices 为分段区间。"""

    node: str
    key: str
    axes: tuple
    shape: tuple
    keep: tuple  # ((轴名, (坐标...)), ...)
    slices: tuple  # ((轴名, (起点, 终点)), ...)
    temp_bytes: int  # 物化副本的临时占用；纯视图为 0

    @property
    def out_axes(self):
        return self.axes

    def to_dict(self):
        return {
            "op": "leaf",
            "node": self.node,
            "key": self.key,
            "axes": list(self.axes),
            "shape": list(self.shape),
            "keep": [[a, list(c)] for a, c in self.keep],
            "slices": [[a, list(r)] for a, r in self.slices],
            "temp_bytes": self.temp_bytes,
        }

    @staticmethod
    def from_dict(data):
        return Leaf(
            node=data["node"],
            key=data["key"],
            axes=tuple(data["axes"]),
            shape=tuple(data["shape"]),
            keep=tuple((a, tuple(c)) for a, c in data["keep"]),
            slices=tuple((a, tuple(r)) for a, r in data["slices"]),
            temp_bytes=int(data["temp_bytes"]),
        )


@dataclass(frozen=True)
class Contract:
    left: object
    right: object
    out_axes: tuple
    shape: tuple
    left_first: bool = True

    @property
    def axes(self):
        return self.out_axes

    def to_dict(self):
        return {
            "op": "contract",
            "left": plan_to_dict(self.left),
            "right": plan_to_dict(self.right),
            "out_axes": list(self.out_axes),
            "shape": list(self.shape),
            "left_first": self.left_first,
        }

    @staticmethod
    def from_dict(data):
        return Contract(
            left=plan_from_dict(data["left"]),
            right=plan_from_dict(data["right"]),
            out_axes=tuple(data["out_axes"]),
            shape=tuple(data["shape"]),
            left_first=bool(data["left_first"]),
        )


@dataclass(frozen=True)
class Slice:
    axis: str
    combine: str  # "concat" 或 "sum"
    ranges: tuple  # 每段 (起点, 终点)
    chunks: tuple  # 每段一个子计划
    out_axes: tuple
    shape: tuple

    @property
    def axes(self):
        return self.out_axes

    def to_dict(self):
        return {
            "op": "slice",
            "axis": self.axis,
            "combine": self.combine,
            "ranges": [list(r) for r in self.ranges],
            "chunks": [plan_to_dict(c) for c in self.chunks],
            "out_axes": list(self.out_axes),
            "shape": list(self.shape),
        }

    @staticmethod
    def from_dict(data):
        return Slice(
            axis=data["axis"],
            combine=data["combine"],
            ranges=tuple(tuple(r) for r in data["ranges"]),
            chunks=tuple(plan_from_dict(c) for c in data["chunks"]),
            out_axes=tuple(data["out_axes"]),
            shape=tuple(data["shape"]),
        )


def plan_to_dict(plan):
    return plan.to_dict()


def plan_from_dict(data):
    op = data["op"]
    if op == "leaf":
        return Leaf.from_dict(data)
    if op == "contract":
        return Contract.from_dict(data)
    if op == "slice":
        return Slice.from_dict(data)
    raise ValueError("未知的计划算子：%s" % op)


def plan_leaves(plan):
    """按计划中的出现顺序列出全部叶子。"""
    if isinstance(plan, Leaf):
        return [plan]
    if isinstance(plan, Contract):
        return plan_leaves(plan.left) + plan_leaves(plan.right)
    leaves = []
    for chunk in plan.chunks:
        leaves.extend(plan_leaves(chunk))
    return leaves


def estimate(plan):
    """返回 (峰值临时字节, 结果临时字节)。叶子纯视图不计临时占用。"""
    if isinstance(plan, Leaf):
        return plan.temp_bytes, plan.temp_bytes
    if isinstance(plan, Contract):
        left_peak, left_temp = estimate(plan.left)
        right_peak, right_temp = estimate(plan.right)
        out = nbytes(plan.shape)
        if plan.left_first:
            peak = max(left_peak, left_temp + right_peak, left_temp + right_temp + out)
        else:
            peak = max(right_peak, right_temp + left_peak, left_temp + right_temp + out)
        return peak, out
    # Slice：输出缓冲与各段峰值（段按顺序执行，取最大）。
    buffer = nbytes(plan.shape)
    chunk_peak = 0
    for chunk in plan.chunks:
        peak, _ = estimate(chunk)
        chunk_peak = max(chunk_peak, peak)
    return buffer + chunk_peak, buffer


class Factor:
    """规划期的因子描述：轴、形状、物化后的临时字节与已建计划。"""

    __slots__ = ("axes", "shape", "temp", "plan")

    def __init__(self, axes, shape, temp, plan):
        self.axes = tuple(axes)
        self.shape = tuple(shape)
        self.temp = int(temp)
        self.plan = plan

    @property
    def nbytes(self):
        return nbytes(self.shape)


def contract_shape(left_axes, left_shape, right_axes, right_shape):
    """按轴名收缩两个因子，返回 (输出轴顺序, 输出形状)。

    输出轴顺序与 np.tensordot 的自然顺序一致：左因子自由轴在前。
    """
    common = [axis for axis in left_axes if axis in right_axes]
    out_axes = [a for a in left_axes if a not in common] + [
        a for a in right_axes if a not in common
    ]
    left_map = dict(zip(left_axes, left_shape))
    right_map = dict(zip(right_axes, right_shape))
    out_shape = tuple(left_map.get(a, right_map.get(a)) for a in out_axes)
    return tuple(out_axes), out_shape


def _greedy(factors):
    """贪心二元收缩：每步选结果最小（并列时收缩轴最多）的一对。"""
    work = list(factors)
    while len(work) > 1:
        best = None
        for i in range(len(work)):
            for j in range(i + 1, len(work)):
                out_axes, out_shape = contract_shape(
                    work[i].axes, work[i].shape, work[j].axes, work[j].shape
                )
                shared = len(work[i].axes) + len(work[j].axes) - len(out_axes)
                score = (nbytes(out_shape), -shared, i, j)
                if best is None or score < best[0]:
                    best = (score, i, j, out_axes, out_shape)
        _, i, j, out_axes, out_shape = best
        left, right = work[i], work[j]
        plan = _order_contract(left, right, out_axes, out_shape)
        work[i] = Factor(out_axes, out_shape, nbytes(out_shape), plan)
        del work[j]
    return work[0]


def _factor_peak(factor):
    peak, _ = estimate(factor.plan)
    return peak


def _order_contract(left, right, out_axes, out_shape):
    """选择执行先后使峰值更小，并生成 Contract 计划。"""
    out = nbytes(out_shape)
    peak_lr = max(_factor_peak(left), left.temp + _factor_peak(right),
                  left.temp + right.temp + out)
    peak_rl = max(_factor_peak(right), right.temp + _factor_peak(left),
                  left.temp + right.temp + out)
    left_first = peak_lr <= peak_rl
    return Contract(left.plan, right.plan, out_axes, out_shape, left_first)


def _chunk_ranges(extent, count):
    """把 [0, extent) 均分为 count 段连续区间。"""
    base, extra = divmod(extent, count)
    ranges = []
    start = 0
    for k in range(count):
        stop = start + base + (1 if k < extra else 0)
        ranges.append((start, stop))
        start = stop
    return tuple(ranges)


def slice_factor(factor, axis, bounds):
    """生成因子沿 axis 取区间 bounds 的子描述。"""
    if axis not in factor.axes:
        return factor
    pos = factor.axes.index(axis)
    start, stop = bounds
    shape = list(factor.shape)
    shape[pos] = stop - start
    plan = factor.plan
    temp = factor.temp
    if isinstance(plan, Leaf):
        slices = dict(plan.slices)
        prev = slices.get(axis, (0, plan.shape[pos]))
        slices[axis] = (prev[0] + start, prev[0] + stop)
        slices = tuple(sorted(slices.items()))
        # 叶子切片是基础块上的连续视图，不产生新的物化副本；
        # 若叶子本身需要物化（keep/indices），副本尺寸随之缩小。
        temp = plan.temp_bytes * nbytes(shape) // max(1, nbytes(plan.shape))
        plan = Leaf(
            node=plan.node,
            key=plan.key,
            axes=plan.axes,
            shape=tuple(shape),
            keep=plan.keep,
            slices=slices,
            temp_bytes=temp,
        )
    else:
        # 非叶子因子被切片：包一层子计划是不必要的；贪心规划只在
        # 叶子上切片，中间因子由子问题内部重新规划。这里不应到达。
        raise AssertionError("只能对叶子因子切片")
    return Factor(factor.axes, tuple(shape), temp, plan)


def plan_contraction(factors, out_axes, budget_bytes, _forbidden=frozenset()):
    """在临时预算内规划一组因子的收缩。

    返回 (计划或 None, 参考字节)：计划为 None 表示受限，参考字节
    给出规划器见到的最小峰值估计，用于说明受限的请求。
    """
    out_axes = tuple(out_axes)
    top = _greedy(factors)
    peak, _ = estimate(top.plan)
    if peak <= budget_bytes:
        return top.plan, peak

    best_required = peak
    # 尝试分段：按“含该轴的因子总字节”从大到小选轴。
    axis_load = {}
    for factor in factors:
        for axis, dim in zip(factor.axes, factor.shape):
            axis_load[axis] = axis_load.get(axis, 0) + nbytes(factor.shape)
    candidates = [
        axis
        for axis, _ in sorted(axis_load.items(), key=lambda kv: -kv[1])
        if axis not in _forbidden
        and any(dim > 1 for f in factors for a, dim in zip(f.axes, f.shape) if a == axis)
    ]
    # 输出缓冲：输出轴的完整形状（分段轴取完整长度）。
    full_extent = {}
    for factor in factors:
        for axis, dim in zip(factor.axes, factor.shape):
            full_extent[axis] = max(full_extent.get(axis, 0), dim)
    out_shape = tuple(full_extent[a] for a in out_axes)
    out_buffer = nbytes(out_shape)

    for axis in candidates:
        extent = full_extent[axis]
        combine = "concat" if axis in out_axes else "sum"
        count = 2
        while count <= extent:
            count = min(count, extent)
            ranges = _chunk_ranges(extent, count)
            sub_budget = budget_bytes - out_buffer
            if sub_budget < 0:
                break
            chunks = []
            ok = True
            worst = 0
            for bounds in ranges:
                sub_factors = [slice_factor(f, axis, bounds) for f in factors]
                sub_plan, sub_required = plan_contraction(
                    sub_factors, out_axes, sub_budget, _forbidden | {axis}
                )
                if sub_plan is None:
                    ok = False
                    worst = max(worst, sub_required)
                    break
                chunks.append(sub_plan)
                sub_peak, _ = estimate(sub_plan)
                worst = max(worst, sub_peak)
            if ok:
                plan = Slice(axis, combine, ranges, tuple(chunks), out_axes, out_shape)
                peak, _ = estimate(plan)
                if peak <= budget_bytes:
                    return plan, peak
            else:
                best_required = min(best_required, out_buffer + worst)
            if count == extent:
                break
            count = min(count * 2, extent)
    return None, best_required
