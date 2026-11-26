"""按部署规格生成合成记录，用于在真实临时预算下验证分段收缩。"""
from __future__ import annotations

import numpy as np


def chain_record(nodes=18, sectors=2, sector_extent=32, scratch_bytes=25165824, seed=0):
    """生成链式网络记录：nodes 个节点、每轴两个扇区、扇区长度 sector_extent。

    两端各留一条开放轴（扇区长度 1），查询取其上的局部分量；
    规模参数与 record.json 的 scale 部署规格同构。
    """
    rng = np.random.default_rng(seed)
    axes = {}
    for t in range(nodes + 1):
        extent = 1 if t in (0, nodes) else sector_extent
        axes["a%d" % t] = [extent, extent]
    # 链上全部奇偶约束的异或和必须为 0， shipped 查询（"00"/"11"）才可满足。
    parities = [t % 2 for t in range(nodes)]
    if sum(parities) % 2:
        parities[-1] ^= 1
    owner_axes = {}
    node_list = []
    blocks = {}
    for t in range(nodes):
        left, right = "a%d" % t, "a%d" % (t + 1)
        block_id = "W%d" % t
        parity = parities[t]
        owner_axes[block_id] = [left, right]
        node_list.append(
            {"id": "N%d" % t, "axes": [left, right], "parity": parity, "own": block_id}
        )
        entries = {}
        for s0 in range(sectors):
            for s1 in range(sectors):
                if (s0 ^ s1) != parity:
                    continue  # 结构零扇区不存储
                key = "%d%d" % (s0, s1)
                shape = (axes[left][s0], axes[right][s1])
                entries[key] = rng.integers(-3, 4, size=shape).tolist()
        blocks[block_id] = entries
    query = {
        "open_axes": ["a0", "a%d" % nodes],
        "sectors": ["00", "11"],
        "coordinates": [[0, 0], [0, 0]],
    }
    return {
        "axes": axes,
        "owner_axes": owner_axes,
        "nodes": node_list,
        "blocks": blocks,
        "replacements": [],
        "query": query,
        "scale": {
            "nodes": nodes,
            "sectors": sectors,
            "sector_extent": sector_extent,
            "scratch_bytes": scratch_bytes,
        },
    }
