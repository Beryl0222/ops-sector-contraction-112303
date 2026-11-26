"""测试用参照：不分扇区、不做预算管理的稠密收缩。

把每个节点展开成完整稠密张量（结构零与缺失块位置填零），
用一条 einsum 收缩，作为引擎结果的独立对照。
"""
import copy

import numpy as np

from sectornet import Engine, Query


def _node_dense(engine, node):
    network = engine.network
    full_shape = [sum(network.axes[axis]) for axis in node.axes]
    out = np.zeros(full_shape)
    perm = node.permutation()
    for key in node.legal_keys():
        base_key = node.base_key(key)
        state = engine.store.state(node.own, base_key)
        if state.kind == "data":
            arr = state.data
            if perm != tuple(range(len(perm))):
                arr = np.transpose(arr, perm)
            lists = network.view_lists(node, key)
            block = arr[np.ix_(*[np.asarray(c, dtype=np.intp) for c in lists])]
        else:
            block = np.zeros(network.node_extents(node, key))
        index = []
        for pos, (axis, bit) in enumerate(zip(node.axes, key)):
            offset = network.axes[axis][0] if bit == "1" else 0
            index.append(offset + np.arange(block.shape[pos]))
        out[np.ix_(*index)] = block
    return out


def dense_full_output(engine, open_axes):
    """返回完整输出张量（各开放轴两个扇区拼接后的全长）。"""
    network = engine.network
    axes = list(network.axes)
    subs = {axis: i for i, axis in enumerate(axes)}
    operands = []
    for node in network.nodes:
        operands.append(_node_dense(engine, node))
        operands.append([subs[axis] for axis in node.axes])
    return np.einsum(*operands, [subs[axis] for axis in open_axes], optimize=True)


def dense_component_values(engine, query):
    """按查询分量取值：{(扇区键, 坐标): 值}。缺失块按零处理。"""
    full = dense_full_output(engine, query.open_axes)
    values = {}
    for key, coords in zip(query.sectors, query.coordinates):
        index = tuple(
            coord + (engine.network.axes[axis][0] if bit == "1" else 0)
            for axis, bit, coord in zip(query.open_axes, key, coords)
        )
        values[(key, tuple(coords))] = float(full[index])
    return values


def perturb_record(record, block, sector, index, delta):
    """复制记录并扰动一个基础块元素，用于有限差分。"""
    clone = copy.deepcopy(record)
    data = clone["blocks"][block][sector]
    target = data
    for i in index[:-1]:
        target = target[i]
    target[index[-1]] += delta
    return clone


def finite_difference(engine, record, query, block, sector, eps=1e-6):
    """对损失做有限差分，返回 {(块, 扇区, 下标): 数值梯度}。

    只适用于梯度状态为 full 的扇区（缺块会改变可用性，差分无意义）。
    """
    grads = {}

    def loss_of(rec):
        eng = Engine.from_record(rec, scratch_bytes=engine.scratch_bytes)
        return eng.run(query).loss

    state = engine.store.state(block, sector)
    for index in np.ndindex(*state.data.shape):
        plus = loss_of(perturb_record(record, block, sector, index, eps))
        minus = loss_of(perturb_record(record, block, sector, index, -eps))
        grads[index] = (plus - minus) / (2 * eps)
    return grads
