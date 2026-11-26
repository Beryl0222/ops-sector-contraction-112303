"""查询引擎：受限收缩、标量损失、反向算子、参数源替换与输出证明。

同一套轴命名、块等价（三态块）及归属契约（own/视图）同时驱动
训练（正向 + 反向）与局部探针（按扇区与局部坐标的分量查询）。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .gf2 import enumerate_assignments
from .model import (
    DTYPE,
    KIND_DATA,
    KIND_MISSING,
    KIND_ZERO,
    BlockStore,
    RecordError,
    network_from_record,
)
from .plan import (
    Factor,
    Leaf,
    nbytes,
    plan_contraction,
    plan_from_dict,
    plan_leaves,
    plan_to_dict,
)
from .runtime import (
    BudgetExceeded,
    LeafContext,
    MemTracker,
    _is_contiguous,
    align_axes,
    backward_plan,
    execute_plan,
    keep_slice_lists,
)

DEFAULT_SCRATCH_BYTES = 64 * 1024 * 1024


# ---------------------------------------------------------------------------
# 查询


@dataclass(frozen=True)
class Query:
    """一次分量查询：输出轴、逐分量的扇区键与局部坐标，可选损失权重。"""

    open_axes: tuple
    sectors: tuple
    coordinates: tuple
    weights: "tuple | None" = None

    @classmethod
    def from_record(cls, data):
        weights = data.get("weights")
        return cls(
            open_axes=tuple(str(a) for a in data["open_axes"]),
            sectors=tuple(str(s) for s in data["sectors"]),
            coordinates=tuple(tuple(int(c) for c in coords) for coords in data["coordinates"]),
            weights=tuple(float(w) for w in weights) if weights is not None else None,
        )

    def to_dict(self):
        data = {
            "open_axes": list(self.open_axes),
            "sectors": list(self.sectors),
            "coordinates": [list(c) for c in self.coordinates],
        }
        if self.weights is not None:
            data["weights"] = list(self.weights)
        return data

    @classmethod
    def from_dict(cls, data):
        return cls.from_record(data)


# ---------------------------------------------------------------------------
# 结果类型


@dataclass(frozen=True)
class ComponentResult:
    """单个被请求分量的结论：可用值 / 无法确定 / 受限。"""

    index: int
    sector_key: str
    coordinates: tuple
    status: str  # "available" | "undetermined" | "restricted"
    value: "float | None"
    missing: tuple  # 参与但缺失的 (块, 扇区)
    reason: str

    @property
    def available(self):
        return self.status == "available"


@dataclass
class _Ref:
    node: str
    key: str
    block: str
    base_key: str
    kind: str


@dataclass
class _Assignment:
    sectors: dict
    status: str  # planned | pruned | missing | zero-killed | restricted
    refs: list
    plan: object = None
    required_bytes: int = 0
    missing: tuple = ()
    pruned_by: "tuple | None" = None
    zero_factor: "tuple | None" = None  # 恰好全零的在场叶子 (块, 扇区)


@dataclass
class _Group:
    sector_key: str
    component_indices: list
    kept: dict
    positions: list
    kept_shape: tuple
    status: str = "available"
    reason: str = ""
    missing: tuple = ()
    assignments: list = field(default_factory=list)
    output: object = None


class QueryResult:
    """一次查询的全部分量结论；只保留可用扇区即可交给下游。"""

    def __init__(self, engine, query, components, groups, loss, peak_bytes):
        self._engine = engine
        self.query = query
        self.components = components
        self.loss = loss
        self.peak_bytes = peak_bytes
        self._groups = groups

    def available(self):
        """可用分量：(下标, 值) 列表，可直接喂给下游。"""
        return [(c.index, c.value) for c in self.components if c.available]

    def unavailable(self):
        return [c for c in self.components if not c.available]

    def proof(self):
        """输出证明：记录计划、扇区赋值与块身份，可用旧输入复得。"""
        engine = self._engine
        groups = []
        for group in self._groups:
            assignments = []
            for asn in group.assignments:
                entry = {
                    "sectors": {a: int(b) for a, b in sorted(asn.sectors.items())},
                    "status": asn.status,
                }
                if asn.plan is not None:
                    entry["plan"] = plan_to_dict(asn.plan)
                    entry["leaves"] = [
                        {
                            "node": ref.node,
                            "key": ref.key,
                            "block": ref.block,
                            "base_key": ref.base_key,
                            "identity": engine.store.identity(ref.block, ref.base_key),
                        }
                        for ref in asn.refs
                        if ref.kind == KIND_DATA
                    ]
                if asn.missing:
                    entry["missing"] = [list(m) for m in asn.missing]
                if asn.pruned_by:
                    entry["pruned_by"] = list(asn.pruned_by)
                if asn.required_bytes:
                    entry["required_bytes"] = asn.required_bytes
                assignments.append(entry)
            groups.append(
                {
                    "sector_key": group.sector_key,
                    "kept": {a: list(c) for a, c in group.kept.items()},
                    "component_indices": list(group.component_indices),
                    "positions": [list(p) for p in group.positions],
                    "status": group.status,
                    "assignments": assignments,
                }
            )
        return {
            "format": "sectornet-proof/1",
            "structure_hash": engine.network.structure_hash(),
            "scratch_bytes": engine.scratch_bytes,
            "query": self.query.to_dict(),
            "components": [
                {
                    "index": c.index,
                    "sector_key": c.sector_key,
                    "coordinates": list(c.coordinates),
                    "status": c.status,
                    "value": c.value,
                    "missing": [list(m) for m in c.missing],
                    "reason": c.reason,
                }
                for c in self.components
            ],
            "loss": self.loss,
            "peak_bytes": self.peak_bytes,
            "groups": groups,
            "identities": engine.store.identities(),
        }


class GradResult:
    """反向结果：基础块坐标下的梯度、状态与复查（eligibility）报告。"""

    def __init__(self, engine, result, grads, statuses, peak_bytes, restricted_reason=None):
        self._engine = engine
        self.result = result
        self.grads = grads  # (块, 扇区) -> ndarray（基础块原坐标）
        self.statuses = statuses  # (块, 扇区) -> (状态, 原因)
        self.peak_bytes = peak_bytes  # 反向过程的临时张量峰值
        self.delivered_bytes = sum(int(g.nbytes) for g in grads.values())  # 交付的梯度占用
        self.restricted_reason = restricted_reason

    @property
    def restricted(self):
        return self.restricted_reason is not None

    def for_block(self, block, sector):
        """基础参数某扇区的梯度（原坐标）；未收到梯度返回 None。"""
        return self.grads.get((block, sector))

    def status(self, block, sector):
        return self.statuses.get((block, sector), ("none", "未参与所请求的输出"))

    def eligibility(self):
        """复查：哪些输出可用于训练、哪些失去资格、参数为何没有梯度。"""
        outputs = [
            {
                "index": c.index,
                "sector_key": c.sector_key,
                "coordinates": list(c.coordinates),
                "trainable": c.available,
                "reason": c.reason if c.reason else "可用于训练",
            }
            for c in self.result.components
        ]
        parameters = []
        store = self._engine.store
        for block in store.block_ids():
            for sector in store.sectors(block):
                state = store.state(block, sector)
                status, reason = self.status(block, sector)
                parameters.append(
                    {
                        "block": block,
                        "sector": sector,
                        "kind": state.kind,
                        "received_gradient": (block, sector) in self.grads,
                        "status": status,
                        "reason": reason,
                    }
                )
        return {"outputs": outputs, "parameters": parameters}


class VerifyReport:
    """输出证明的复得报告。"""

    def __init__(self):
        self.structure_ok = False
        self.components = []  # {index, recorded, reproduced, match, note}
        self.blocks = []  # {block, sector, recorded_identity, current_identity, match}
        self.notes = []
        self.ok = False

    def summary(self):
        lines = ["结构一致：%s" % self.structure_ok, "证明可复得：%s" % self.ok]
        for comp in self.components:
            lines.append(
                "分量 %d：记录=%r 复得=%r 一致=%s %s"
                % (comp["index"], comp["recorded"], comp["reproduced"], comp["match"], comp["note"])
            )
        changed = [b for b in self.blocks if not b["match"]]
        for block in changed:
            lines.append("块身份变化：%(block)s/%(sector)s" % block)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# 引擎


class Engine:
    """同一节点图下的受限收缩与反向算子。

    Engine 不可变：``with_replacement`` 返回新引擎，旧引擎及其
    输出证明仍可用旧输入复得。
    """

    def __init__(self, network, store, scratch_bytes=DEFAULT_SCRATCH_BYTES, scale=None):
        self.network = network
        self.store = store
        self.scratch_bytes = int(scratch_bytes)
        self.scale = scale or {}

    @classmethod
    def from_record(cls, record, scratch_bytes=None):
        network = network_from_record(record)
        store = BlockStore.from_record(network, record["blocks"])
        scale = record.get("scale", {})
        budget = scratch_bytes if scratch_bytes is not None else scale.get(
            "scratch_bytes", DEFAULT_SCRATCH_BYTES
        )
        return cls(network, store, budget, scale=scale)

    def with_replacement(self, source, sector, values):
        """替换单个参数源的一个扇区，返回使用新块身份与真实占用的新引擎。"""
        return Engine(
            self.network,
            self.store.replace(source, sector, values),
            self.scratch_bytes,
            scale=self.scale,
        )

    # -- 查询 ------------------------------------------------------------

    def _validate_query(self, query):
        dangling = set(self.network.dangling_axes())
        if set(query.open_axes) != dangling:
            raise RecordError(
                "查询的输出轴 %s 应与网络的开放轴 %s 一致"
                % (sorted(query.open_axes), sorted(dangling))
            )
        if len(query.sectors) != len(query.coordinates):
            raise RecordError("sectors 与 coordinates 的数量应一致")
        if query.weights is not None and len(query.weights) != len(query.sectors):
            raise RecordError("weights 的数量应与所查分量一致")
        for key, coords in zip(query.sectors, query.coordinates):
            if len(key) != len(query.open_axes) or any(c not in "01" for c in key):
                raise RecordError("扇区键 %s 与输出轴不符" % key)
            if len(coords) != len(query.open_axes):
                raise RecordError("坐标 %s 应与输出轴一一对应" % (coords,))
            for axis, bit, coord in zip(query.open_axes, key, coords):
                extent = self.network.axis_extent(axis, bit)
                if not 0 <= coord < extent:
                    raise RecordError(
                        "坐标长度必须符合所查扇区：轴 %s 扇区 %s 长度为 %d，坐标为 %d"
                        % (axis, bit, extent, coord)
                    )

    def _leaf_factor(self, node, key, kept):
        """构造叶子的规划因子：输出轴保留坐标已施加，临时占用按真实切法计。"""
        keep = tuple(sorted((a, tuple(c)) for a, c in kept.items() if a in node.axes))
        ks_lists = keep_slice_lists(self.network, node, key, keep, ())
        shape = tuple(len(c) for c in ks_lists)
        view_lists = self.network.view_lists(node, key)
        composed = tuple(
            tuple(view_lists[p][c] for c in ks_lists[p]) for p in range(len(node.axes))
        )
        contiguous = all(_is_contiguous(coords) for coords in composed)
        temp = 0 if contiguous else nbytes(shape)
        plan = Leaf(node.id, key, node.axes, shape, keep, (), temp)
        return Factor(node.axes, shape, temp, plan)

    def run(self, query):
        """执行分量查询，返回 QueryResult（含标量损失与输出证明素材）。"""
        query = query if isinstance(query, Query) else Query.from_record(query)
        self._validate_query(query)
        ctx = LeafContext(self.network, self.store)
        tracker = MemTracker(self.scratch_bytes)

        # 按扇区键分组，同组共享扇区赋值枚举与收缩计划。
        groups_of = {}
        for index, key in enumerate(query.sectors):
            groups_of.setdefault(key, []).append(index)

        components = [None] * len(query.sectors)
        groups = []
        for sector_key, indices in groups_of.items():
            group = self._run_group(query, sector_key, indices, ctx, tracker)
            groups.append(group)
            for ci, pos in zip(group.component_indices, group.positions):
                if group.status == "available":
                    value = float(group.output[pos])
                    components[ci] = ComponentResult(
                        ci, sector_key, query.coordinates[ci], "available", value,
                        (), group.reason,
                    )
                elif group.status == "undetermined":
                    components[ci] = ComponentResult(
                        ci, sector_key, query.coordinates[ci], "undetermined", None,
                        group.missing, group.reason,
                    )
                else:
                    components[ci] = ComponentResult(
                        ci, sector_key, query.coordinates[ci], "restricted", None,
                        (), group.reason,
                    )

        weights = query.weights or (1.0,) * len(query.sectors)
        loss = None
        for comp, weight in zip(components, weights):
            if comp.available:
                loss = (loss or 0.0) + weight * comp.value
        return QueryResult(self, query, components, groups, loss, tracker.peak)

    def _run_group(self, query, sector_key, indices, ctx, tracker):
        kept = {}
        for ci in indices:
            for axis, coord in zip(query.open_axes, query.coordinates[ci]):
                kept.setdefault(axis, set()).add(coord)
        kept = {axis: sorted(coords) for axis, coords in kept.items()}
        positions = [
            tuple(kept[axis].index(coord) for axis, coord in zip(query.open_axes, query.coordinates[ci]))
            for ci in indices
        ]
        kept_shape = tuple(len(kept[axis]) for axis in query.open_axes)
        group = _Group(sector_key, indices, kept, positions, kept_shape)

        fixed = {axis: int(bit) for axis, bit in zip(query.open_axes, sector_key)}
        equations = [(set(node.axes), node.parity) for node in self.network.nodes]
        solutions = enumerate_assignments(list(self.network.axes), equations, fixed)
        if not solutions:
            group.output = np.zeros(kept_shape, dtype=DTYPE)
            group.reason = "奇偶约束下无合法扇区赋值，输出为确定零"
            return group

        out_budget = self.scratch_bytes
        for solution in solutions:
            group.assignments.append(
                self._analyze_assignment(solution, kept, query.open_axes, out_budget, ctx)
            )

        missing_all = []
        restricted_required = 0
        for asn in group.assignments:
            if asn.status == "missing":
                missing_all.extend(asn.missing)
            elif asn.status == "restricted":
                restricted_required = max(restricted_required, asn.required_bytes)
        if missing_all:
            group.status = "undetermined"
            seen = []
            for item in missing_all:
                if item not in seen:
                    seen.append(item)
            group.missing = tuple(seen)
            group.reason = "缺失块实际参与该输出：" + "、".join(
                "%s/%s" % item for item in seen
            )
            return group
        if restricted_required:
            group.status = "restricted"
            group.reason = (
                "临时张量预算不足：该扇区至少需要约 %d 字节，预算 %d 字节"
                % (restricted_required, self.scratch_bytes)
            )
            return group

        out = np.zeros(kept_shape, dtype=DTYPE)
        # 输出是交付给调用者的结果，不计入临时预算；只有中间结果记账。
        for asn in group.assignments:
            if asn.status != "planned":
                continue  # pruned / zero-killed：乘积确定为零
            axes, arr, held = execute_plan(asn.plan, ctx, tracker)
            out += align_axes(axes, arr, query.open_axes)
            if held:
                tracker.free(held)
        group.output = out
        return group

    def _analyze_assignment(self, solution, kept, open_axes, out_budget, ctx):
        refs = []
        for node in self.network.nodes:
            key = "".join(str(solution[axis]) for axis in node.axes)
            base_key = node.base_key(key)
            state = self.store.state(node.own, base_key)
            refs.append(_Ref(node.id, key, node.own, base_key, state.kind))

        kinds = [ref.kind for ref in refs]
        if KIND_ZERO in kinds:
            zero_ref = refs[kinds.index(KIND_ZERO)]
            return _Assignment(
                solution, "pruned", refs, pruned_by=(zero_ref.block, zero_ref.base_key)
            )

        if KIND_MISSING in kinds:
            missing = tuple(
                (ref.block, ref.base_key) for ref in refs if ref.kind == KIND_MISSING
            )
            # 缺失块是否实际参与：任一在场叶子恰好全零，则乘积确定为零。
            for ref in refs:
                if ref.kind != KIND_DATA:
                    continue
                node = self.network.node_map[ref.node]
                factor = self._leaf_factor(node, ref.key, kept)
                arr, temp = ctx.materialize(factor.plan)
                if arr.size and not np.any(arr):
                    return _Assignment(
                        solution, "zero-killed", refs, missing=missing,
                        zero_factor=(ref.block, ref.base_key),
                    )
            return _Assignment(solution, "missing", refs, missing=missing)

        factors = []
        for ref in refs:
            node = self.network.node_map[ref.node]
            factors.append(self._leaf_factor(node, ref.key, kept))
        plan, required = plan_contraction(factors, open_axes, out_budget)
        if plan is None:
            return _Assignment(solution, "restricted", refs, required_bytes=required)
        return _Assignment(solution, "planned", refs, plan=plan)

    # -- 反向 ------------------------------------------------------------

    def backward(self, result, upstream=None):
        """把真正采用的输出的上游梯度归还到每个原始参数块及数据块。"""
        query = result.query
        if upstream is None:
            upstream = query.weights or (1.0,) * len(query.sectors)
        if isinstance(upstream, dict):
            upstream = [upstream.get(i, 0.0) for i in range(len(query.sectors))]
        upstream = [float(u) for u in upstream]

        ctx = LeafContext(self.network, self.store)
        tracker = MemTracker(self.scratch_bytes)
        leaf_grads = {}

        def sink(plan, grad):
            key = (plan.node, plan.key)
            gfull = leaf_grads.get(key)
            if gfull is None:
                node = self.network.node_map[plan.node]
                # 梯度数组是交付给调用者的结果，不计入临时预算。
                gfull = np.zeros(self.network.node_extents(node, plan.key), dtype=DTYPE)
                leaf_grads[key] = gfull
            node = self.network.node_map[plan.node]
            ks_lists = keep_slice_lists(self.network, node, plan.key, plan.keep, plan.slices)
            index = np.ix_(*[np.asarray(coords, dtype=np.intp) for coords in ks_lists])
            np.add.at(gfull, index, grad)

        try:
            for group in result._groups:
                if group.status != "available":
                    continue  # 只有真正采用的输出回传梯度
                g_out = np.zeros(group.kept_shape, dtype=DTYPE)
                tracker.alloc(g_out.nbytes)
                try:
                    for ci, pos in zip(group.component_indices, group.positions):
                        g_out[pos] += upstream[ci]
                    for asn in group.assignments:
                        if asn.status != "planned":
                            continue
                        grad = align_axes(query.open_axes, g_out, asn.plan.axes)
                        backward_plan(asn.plan, grad, ctx, tracker, sink)
                finally:
                    tracker.free(g_out.nbytes)
        except BudgetExceeded as exc:
            return GradResult(self, result, {}, {}, tracker.peak, restricted_reason=str(exc))

        # 视图反传：重排/切片后的全部贡献加回基础参数的原坐标。
        base_grads = {}
        for (node_id, key), gfull in leaf_grads.items():
            node = self.network.node_map[node_id]
            base_key = node.base_key(key)
            base_shape = self.network.base_extents(node, base_key)
            view_lists = self.network.view_lists(node, key)
            perm = node.permutation()
            index = [None] * len(perm)
            for p in range(len(perm)):
                index[perm[p]] = np.asarray(view_lists[p], dtype=np.intp)
            grids = np.meshgrid(*index, indexing="ij")
            # 梯度按视图位置排列，散布前转置到基础块轴顺序。
            order = [perm.index(q) for q in range(len(perm))]
            gfull_base = np.transpose(gfull, order) if order != list(range(len(perm))) else gfull
            gbase = np.zeros(base_shape, dtype=DTYPE)
            np.add.at(gbase, tuple(grids), gfull_base)
            slot = (node.own, base_key)
            if slot in base_grads:
                base_grads[slot] = base_grads[slot] + gbase
            else:
                base_grads[slot] = gbase

        statuses = self._gradient_statuses(result)
        # 梯度确定为零的扇区（被结构零或零因子消除）交付零梯度数组。
        for slot, (status, _reason) in statuses.items():
            if status == "zero" and slot not in base_grads:
                block, sector = slot
                shape = tuple(
                    self.network.axis_extent(axis, bit)
                    for axis, bit in zip(self.network.owner_axes[block], sector)
                )
                base_grads[slot] = np.zeros(shape, dtype=DTYPE)
        return GradResult(self, result, base_grads, statuses, tracker.peak)

    def _gradient_statuses(self, result):
        """汇总每个基础块扇区的梯度状态与原因。"""
        fates = {}

        def note(block, sector, fate):
            fates.setdefault((block, sector), set()).add(fate)

        for group in result._groups:
            for asn in group.assignments:
                for ref in asn.refs:
                    slot = (ref.block, ref.base_key)
                    if asn.status == "planned":
                        note(*slot, "computed")
                    elif asn.status == "pruned":
                        note(*slot, "pruned")
                    elif asn.status == "missing":
                        note(*slot, "missing-cofactor" if ref.kind == KIND_DATA else "missing-self")
                    elif asn.status == "zero-killed":
                        if (ref.block, ref.base_key) == asn.zero_factor:
                            note(*slot, "zero-factor")
                        elif ref.kind == KIND_DATA:
                            note(*slot, "zero-killed-cofactor")
                        else:
                            note(*slot, "missing-self")
                    elif asn.status == "restricted":
                        note(*slot, "restricted")

        statuses = {}
        for block in self.store.block_ids():
            for sector in self.store.sectors(block):
                slot = (block, sector)
                state = self.store.state(block, sector)
                if state.kind == KIND_ZERO:
                    statuses[slot] = ("structural-zero", "声明为结构零的位置是固定约束，没有可训练的参数")
                    continue
                if state.kind == KIND_MISSING:
                    statuses[slot] = ("missing", "数据未到达，不是可训练参数")
                    continue
                slot_fates = fates.get(slot, set())
                notes = []
                if slot_fates & {"missing-cofactor", "missing-self"}:
                    notes.append("另服务于无法确定的输出")
                if "restricted" in slot_fates:
                    notes.append("另有扇区因预算不足未完成收缩")
                extra = ("；" + "；".join(notes)) if notes else ""
                if "computed" in slot_fates:
                    if "zero-factor" in slot_fates:
                        statuses[slot] = (
                            "partial",
                            "部分梯度取决于缺块：所在乘积含缺失块，该叶恰好全零" + extra,
                        )
                    else:
                        statuses[slot] = ("full", "梯度完整" + extra)
                elif "zero-factor" in slot_fates:
                    statuses[slot] = (
                        "undetermined",
                        "梯度无法确定：所在乘积含缺失块，该叶恰好全零" + extra,
                    )
                elif "pruned" in slot_fates or "zero-killed-cofactor" in slot_fates:
                    if "pruned" in slot_fates:
                        reason = "梯度恒为零：所在乘积被结构零消除"
                    else:
                        reason = "梯度确定为零：所在乘积含恰好全零的因子"
                    statuses[slot] = ("zero", reason + extra)
                elif slot_fates & {"missing-cofactor", "missing-self"}:
                    statuses[slot] = ("none", "仅服务于无法确定的输出，未参与训练")
                elif "restricted" in slot_fates:
                    statuses[slot] = ("restricted", "所在扇区因预算不足未完成收缩")
                else:
                    statuses[slot] = ("none", "未参与所请求的输出")
        return statuses

    # -- 证明复得 ----------------------------------------------------------

    def verify(self, proof):
        """用当前输入复得已保存的输出证明。"""
        report = VerifyReport()
        if proof.get("format") != "sectornet-proof/1":
            report.notes.append("证明格式未知")
            return report
        report.structure_ok = proof.get("structure_hash") == self.network.structure_hash()
        if not report.structure_ok:
            report.notes.append("节点图结构已变化，证明不再适用")
            return report

        query = Query.from_dict(proof["query"])
        ctx = LeafContext(self.network, self.store)
        tracker = MemTracker(self.scratch_bytes)
        recorded_components = {c["index"]: c for c in proof["components"]}

        for group in proof["groups"]:
            kept = {axis: list(coords) for axis, coords in group["kept"].items()}
            kept_shape = tuple(len(kept[axis]) for axis in query.open_axes)
            out = np.zeros(kept_shape, dtype=DTYPE)
            reproducible = True
            note = ""
            for asn in group["assignments"]:
                if "plan" not in asn:
                    continue
                plan = plan_from_dict(asn["plan"])
                for leaf in plan_leaves(plan):
                    node = self.network.node_map[leaf.node]
                    state = self.store.state(node.own, node.base_key(leaf.key))
                    if state.kind != KIND_DATA:
                        reproducible = False
                        note = "块 %s/%s 当前无数据" % (node.own, node.base_key(leaf.key))
                        break
                if not reproducible:
                    break
                axes, arr, held = execute_plan(plan, ctx, tracker)
                out += align_axes(axes, arr, query.open_axes)
                if held:
                    tracker.free(held)
            for ci, pos in zip(group["component_indices"], group["positions"]):
                recorded = recorded_components[ci]
                entry = {
                    "index": ci,
                    "recorded": recorded["value"],
                    "reproduced": None,
                    "match": False,
                    "note": "",
                }
                if recorded["status"] != "available":
                    entry["note"] = "记录中该分量不可用于训练（%s）" % recorded["status"]
                    entry["match"] = True
                elif not reproducible:
                    entry["note"] = note or "当前输入下无法复得"
                else:
                    value = float(out[tuple(pos)])
                    entry["reproduced"] = value
                    entry["match"] = value == recorded["value"]
                    if not entry["match"]:
                        entry["note"] = "复得值与记录不一致（输入可能已更换）"
                report.components.append(entry)

        for slot, identity in proof["identities"].items():
            block, sector = slot.split("/", 1)
            try:
                current = self.store.identity(block, sector)
            except KeyError:
                current = None
            report.blocks.append(
                {
                    "block": block,
                    "sector": sector,
                    "recorded_identity": identity,
                    "current_identity": current,
                    "match": current == identity,
                }
            )
        report.ok = report.structure_ok and all(c["match"] for c in report.components)
        return report
