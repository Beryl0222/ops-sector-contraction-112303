"""对称分块张量网络的数据模型、记录校验与视图坐标映射。

坐标与枚举约定（与 data/record.json 配套）：

- ``axes`` 给出轴在扇区 0、1 的长度；连接同名两条轴，每轴至多出现两次。
- 每个块以其轴顺序编码扇区键（如 ``"010"``）；节点各轴扇区位异或须等于
  节点 ``parity``，其余扇区为结构零，不存储。
- ``blocks`` 以 ``owner_axes`` 的顺序存储基础块：嵌套数组为数据，
  ``null`` 声明未到的数据（缺失块），字符串 ``"zero"`` 声明固定的结构零。
- 节点的 ``own`` 指向基础块；``axis_permutation`` 按输出轴位置给出原轴
  下标；``indices`` 以节点轴顺序写扇区键，值为重排之后各轴要读取的坐标
  列表，可重复取位。未提供视图变换时使用原布局。
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np

DTYPE = np.float64
KIND_DATA = "data"
KIND_MISSING = "missing"
KIND_ZERO = "zero"


class RecordError(ValueError):
    """交换记录违反输入契约。"""


def _sector_key(bits):
    return "".join(str(int(b) & 1) for b in bits)


@dataclass(frozen=True)
class BlockState:
    """基础块某个扇区的状态：数据 / 缺失 / 固定结构零。"""

    kind: str
    data: "np.ndarray | None" = None

    @property
    def nbytes(self):
        return 0 if self.data is None else int(self.data.nbytes)


@dataclass(frozen=True)
class Node:
    """网络节点：带名轴、奇偶约束与对基础块的视图引用。"""

    id: str
    axes: tuple
    parity: int
    own: str
    axis_permutation: "tuple | None" = None
    # 节点扇区键 -> 重排之后各位置要读取的坐标列表（可重复取位）。
    indices: "dict | None" = None

    def legal_keys(self):
        """满足奇偶约束的全部节点扇区键（按节点轴顺序）。"""
        count = len(self.axes)
        return [
            _sector_key((k >> (count - 1 - p)) & 1 for p in range(count))
            for k in range(1 << count)
            if bin(k).count("1") % 2 == self.parity
        ]

    def permutation(self):
        """重排表：位置 p 读取基础块的第 permutation()[p] 条轴。"""
        if self.axis_permutation is None:
            return tuple(range(len(self.axes)))
        return tuple(self.axis_permutation)

    def base_key(self, node_key):
        """节点扇区键换算为基础块扇区键（异或不变量随重排保持）。"""
        perm = self.permutation()
        return "".join(node_key[perm.index(q)] for q in range(len(perm)))

    def view_lists(self, node_key, position_extents):
        """重排之后各位置要读取的坐标列表；缺省为整段范围。"""
        if self.indices is None:
            return tuple(tuple(range(position_extents[p])) for p in range(len(self.axes)))
        return tuple(tuple(int(c) for c in coords) for coords in self.indices[node_key])


@dataclass(frozen=True)
class Network:
    """网络结构：轴、基础块布局与节点图（不含块数据）。"""

    axes: dict  # 轴名 -> (扇区0长度, 扇区1长度)
    owner_axes: dict  # 基础块 id -> 轴名元组
    nodes: tuple  # Node 元组

    @property
    def node_map(self):
        return {node.id: node for node in self.nodes}

    def axis_extent(self, axis, sector):
        return self.axes[axis][int(sector)]

    def axis_users(self):
        """轴名 -> 引用它的节点列表（每轴至多两个）。"""
        users = {axis: [] for axis in self.axes}
        for node in self.nodes:
            for axis in node.axes:
                users[axis].append(node)
        return users

    def dangling_axes(self):
        """只出现一次的轴（网络的输出轴）。"""
        return tuple(axis for axis, users in self.axis_users().items() if len(users) == 1)

    def node_extents(self, node, node_key):
        """节点视图块各位置的尺寸（即节点轴在该扇区的长度）。"""
        return tuple(self.axis_extent(axis, bit) for axis, bit in zip(node.axes, node_key))

    def base_extents(self, node, base_key):
        """基础块扇区的各轴尺寸（按 owner_axes 顺序）。"""
        owner = self.owner_axes[node.own]
        return tuple(self.axis_extent(axis, bit) for axis, bit in zip(owner, base_key))

    def view_lists(self, node, node_key):
        """重排之后各位置要读取的坐标列表（缺省为整段范围）。"""
        base_key = node.base_key(node_key)
        extents = self.base_extents(node, base_key)
        perm = node.permutation()
        return node.view_lists(node_key, tuple(extents[q] for q in perm))

    def structure_hash(self):
        """节点图结构指纹：与块数据无关，替换参数源后保持不变。"""
        payload = {
            "axes": {k: list(v) for k, v in sorted(self.axes.items())},
            "owner_axes": {k: list(v) for k, v in sorted(self.owner_axes.items())},
            "nodes": [
                {
                    "id": n.id,
                    "axes": list(n.axes),
                    "parity": n.parity,
                    "own": n.own,
                    "axis_permutation": list(n.axis_permutation) if n.axis_permutation else None,
                    "indices": (
                        {k: [list(c) for c in v] for k, v in sorted(n.indices.items())}
                        if n.indices
                        else None
                    ),
                }
                for n in sorted(self.nodes, key=lambda n: n.id)
            ],
        }
        blob = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()


def _validate_axes(raw):
    axes = {}
    for name, extents in raw.items():
        if (
            not isinstance(extents, list)
            or len(extents) != 2
            or any(not isinstance(e, int) or isinstance(e, bool) or e < 0 for e in extents)
        ):
            raise RecordError("轴 %s 的扇区长度应为两个非负整数" % name)
        axes[str(name)] = (int(extents[0]), int(extents[1]))
    return axes


def _validate_nodes(raw_nodes, axes, owner_axes):
    nodes = []
    seen_ids = set()
    axis_use = {axis: 0 for axis in axes}
    for pos, raw in enumerate(raw_nodes):
        where = "nodes[%d]" % pos
        for field_name in ("id", "axes", "parity", "own"):
            if field_name not in raw:
                raise RecordError("%s 缺少字段 %s" % (where, field_name))
        node_id = str(raw["id"])
        if node_id in seen_ids:
            raise RecordError("节点 id 重复：%s" % node_id)
        seen_ids.add(node_id)
        node_axes = tuple(str(a) for a in raw["axes"])
        for axis in node_axes:
            if axis not in axes:
                raise RecordError("%s 引用了未知轴 %s" % (where, axis))
            axis_use[axis] += 1
        parity = raw["parity"]
        if parity not in (0, 1):
            raise RecordError("%s 的 parity 应为 0 或 1" % where)
        own = str(raw["own"])
        if own not in owner_axes:
            raise RecordError("%s 引用了未知基础块 %s" % (where, own))
        perm = raw.get("axis_permutation")
        if perm is not None:
            perm = tuple(int(p) for p in perm)
            if sorted(perm) != list(range(len(owner_axes[own]))):
                raise RecordError("%s 的 axis_permutation 应为原轴下标的一个排列" % where)
            if len(perm) != len(node_axes):
                raise RecordError("%s 的 axis_permutation 长度应与节点轴数一致" % where)
        elif node_axes != tuple(owner_axes[own]):
            raise RecordError("%s 未提供视图变换时节点轴应与基础块布局一致" % where)
        indices = raw.get("indices")
        if indices is not None:
            if perm is None:
                raise RecordError("%s 提供 indices 时应同时提供 axis_permutation" % where)
            parsed = {}
            for key, lists in indices.items():
                if len(key) != len(node_axes) or any(c not in "01" for c in key):
                    raise RecordError("%s 的 indices 扇区键 %s 非法" % (where, key))
                if sum(int(c) for c in key) % 2 != parity:
                    raise RecordError("%s 的 indices 扇区键 %s 不满足奇偶约束" % (where, key))
                if len(lists) != len(node_axes):
                    raise RecordError("%s 的 indices 键 %s 应按节点轴给出坐标列表" % (where, key))
                parsed[key] = tuple(tuple(int(c) for c in coords) for coords in lists)
            indices = parsed
        node = Node(
            id=node_id,
            axes=node_axes,
            parity=int(parity),
            own=own,
            axis_permutation=perm,
            indices=indices,
        )
        if indices is not None:
            absent = [key for key in node.legal_keys() if key not in indices]
            if absent:
                raise RecordError(
                    "%s 的 indices 缺少合法扇区键：%s" % (where, "、".join(absent))
                )
        nodes.append(node)
    for axis, count in axis_use.items():
        if count > 2:
            raise RecordError("轴 %s 出现了 %d 次（至多两次）" % (axis, count))
    return tuple(nodes)


def _validate_views(network):
    """校验视图坐标：长度符合节点扇区，坐标落在基础块扇区内。"""
    for node in network.nodes:
        for key in node.legal_keys():
            extents = network.node_extents(node, key)
            lists = network.view_lists(node, key)
            base_key = node.base_key(key)
            base_extents = network.base_extents(node, base_key)
            perm = node.permutation()
            for p, coords in enumerate(lists):
                if len(coords) != extents[p]:
                    raise RecordError(
                        "节点 %s 扇区 %s 第 %d 轴读取 %d 个坐标，应为 %d 个"
                        % (node.id, key, p, len(coords), extents[p])
                    )
                limit = base_extents[perm[p]]
                if any(c < 0 or c >= limit for c in coords):
                    raise RecordError(
                        "节点 %s 扇区 %s 第 %d 轴坐标越界（基础块 %s 扇区 %s）"
                        % (node.id, key, p, node.own, base_key)
                    )


class BlockStore:
    """基础块数据状态：数据 / 缺失（null）/ 固定结构零（"zero"）。

    替换单个参数源得到新的 BlockStore，原存储保持不变。
    """

    def __init__(self, network, blocks):
        self._network = network
        self._blocks = blocks  # block id -> {扇区键: BlockState}

    @classmethod
    def from_record(cls, network, raw_blocks):
        blocks = {}
        for block_id, owner in network.owner_axes.items():
            raw = raw_blocks.get(block_id)
            if raw is None:
                raise RecordError("缺少基础块 %s 的数据声明" % block_id)
            entries = {}
            for key, value in raw.items():
                if len(key) != len(owner) or any(c not in "01" for c in key):
                    raise RecordError("基础块 %s 的扇区键 %s 非法" % (block_id, key))
                shape = tuple(
                    network.axis_extent(axis, bit) for axis, bit in zip(owner, key)
                )
                if value is None:
                    entries[key] = BlockState(KIND_MISSING)
                elif value == "zero":
                    entries[key] = BlockState(KIND_ZERO)
                else:
                    data = np.asarray(value, dtype=DTYPE)
                    if data.shape != shape:
                        raise RecordError(
                            "基础块 %s 扇区 %s 的形状 %s 与轴长 %s 不符"
                            % (block_id, key, data.shape, shape)
                        )
                    entries[key] = BlockState(KIND_DATA, np.ascontiguousarray(data))
            blocks[block_id] = entries
        store = cls(network, blocks)
        store.validate_coverage()
        return store

    def validate_coverage(self):
        """每个节点的每个合法扇区键都必须在基础块中有声明。"""
        missing = []
        for node in self._network.nodes:
            for key in node.legal_keys():
                base_key = node.base_key(key)
                if base_key not in self._blocks[node.own]:
                    missing.append("%s/%s（节点 %s）" % (node.own, base_key, node.id))
        if missing:
            raise RecordError("以下合法扇区未在 blocks 中声明：" + "、".join(missing))

    def state(self, block_id, key):
        return self._blocks[block_id][key]

    def sectors(self, block_id):
        return tuple(sorted(self._blocks[block_id]))

    def block_ids(self):
        return tuple(sorted(self._blocks))

    def identity(self, block_id, key):
        """块身份：由状态种类、形状与数据内容决定；替换参数源后改变。"""
        state = self._blocks[block_id][key]
        digest = hashlib.sha256()
        digest.update(state.kind.encode("utf-8"))
        if state.data is not None:
            digest.update(np.asarray(state.data.shape, dtype=np.int64).tobytes())
            digest.update(np.ascontiguousarray(state.data, dtype=DTYPE).tobytes())
        return digest.hexdigest()

    def identities(self):
        return {
            "%s/%s" % (block_id, key): self.identity(block_id, key)
            for block_id in self.block_ids()
            for key in self.sectors(block_id)
        }

    def replace(self, block_id, key, values):
        """替换单个参数源的一个扇区，返回新的 BlockStore。"""
        if block_id not in self._blocks:
            raise RecordError("未知基础块 %s" % block_id)
        owner = self._network.owner_axes[block_id]
        if len(key) != len(owner) or any(c not in "01" for c in key):
            raise RecordError("扇区键 %s 与基础块 %s 的轴数不符" % (key, block_id))
        shape = tuple(self._network.axis_extent(axis, bit) for axis, bit in zip(owner, key))
        data = np.asarray(values, dtype=DTYPE)
        if data.shape != shape:
            raise RecordError(
                "替换块 %s/%s 的形状 %s 与轴长 %s 不符" % (block_id, key, data.shape, shape)
            )
        blocks = {bid: dict(entries) for bid, entries in self._blocks.items()}
        blocks[block_id][key] = BlockState(KIND_DATA, np.ascontiguousarray(data))
        return BlockStore(self._network, blocks)


def network_from_record(record):
    """从交换记录构建并校验网络结构。"""
    for field_name in ("axes", "owner_axes", "nodes", "blocks"):
        if field_name not in record:
            raise RecordError("缺少交换字段：%s" % field_name)
    axes = _validate_axes(record["axes"])
    owner_axes = {}
    for block_id, axes_list in record["owner_axes"].items():
        owner = tuple(str(a) for a in axes_list)
        for axis in owner:
            if axis not in axes:
                raise RecordError("基础块 %s 引用了未知轴 %s" % (block_id, axis))
        owner_axes[str(block_id)] = owner
    nodes = _validate_nodes(record["nodes"], axes, owner_axes)
    network = Network(axes=axes, owner_axes=owner_axes, nodes=nodes)
    _validate_views(network)
    return network
