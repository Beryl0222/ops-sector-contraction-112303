"""GF(2) 异或约束求解：枚举满足全部节点奇偶约束的扇区赋值。

网络中每条轴取一个扇区位（0/1）。每个节点贡献一条约束：
其各轴扇区位的异或等于节点 parity；被请求的输出轴扇区
作为固定位加入。返回全部合法赋值，供逐赋值收缩使用。
"""
from __future__ import annotations

# 自由位上限：赋值数量是 2^自由位，超过即视为不可枚举的输入。
MAX_FREE_BITS = 20


class AssignmentError(ValueError):
    """扇区赋值无法枚举（自由度过大）。"""


def enumerate_assignments(axes, equations, fixed):
    """枚举全部合法扇区赋值。

    参数:
        axes: 全部轴名（确定变量顺序）。
        equations: [(变量集合, 右端位)] 形式的异或方程。
        fixed: {轴名: 0/1} 固定位（例如被请求的输出轴扇区）。

    返回:
        list[dict[str, int]]：每个解给出各轴的扇区位；若无解返回空列表。
    """
    axes = list(axes)
    index = {axis: pos for pos, axis in enumerate(axes)}
    rows = []
    for variables, rhs in equations:
        mask = 0
        for var in variables:
            mask |= 1 << index[var]
        rows.append((mask, int(rhs) & 1))
    for var, bit in fixed.items():
        rows.append((1 << index[var], int(bit) & 1))

    # 高斯消元：pivots[列] = (行掩码, 右端)，行掩码的最低置位即主元列。
    pivots = {}
    for mask, rhs in rows:
        cur_mask, cur_rhs = mask, rhs
        while cur_mask:
            col = (cur_mask & -cur_mask).bit_length() - 1
            if col in pivots:
                piv_mask, piv_rhs = pivots[col]
                cur_mask ^= piv_mask
                cur_rhs ^= piv_rhs
            else:
                pivots[col] = (cur_mask, cur_rhs)
                break
        else:
            if cur_rhs:
                return []  # 矛盾：无合法赋值

    free_cols = [col for col in range(len(axes)) if col not in pivots]
    if len(free_cols) > MAX_FREE_BITS:
        raise AssignmentError(
            "扇区赋值自由位 %d 个（上限 %d），无法枚举" % (len(free_cols), MAX_FREE_BITS)
        )

    def solve(free_bits):
        solution = 0
        for col, bit in zip(free_cols, free_bits):
            if bit:
                solution |= 1 << col
        # 主元行只含不低于主元列的位，自高向低回代即可。
        for col in sorted(pivots, reverse=True):
            mask, rhs = pivots[col]
            rest = mask & ~(1 << col)
            parity = bin(rest & solution).count("1") & 1
            if parity ^ rhs:
                solution |= 1 << col
        return solution

    solutions = []
    for combo in range(1 << len(free_cols)):
        free_bits = [(combo >> k) & 1 for k in range(len(free_cols))]
        solution = solve(free_bits)
        solutions.append({axis: (solution >> index[axis]) & 1 for axis in axes})
    return solutions
