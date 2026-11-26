"""本地合成交换样本的读取入口。

作为命令运行时，先列出各顶层输入项目与 NumPy 版本，再用
sectornet 引擎执行记录中的查询、反向与参数源替换，并核对
输出证明可用旧输入复得。
"""
import argparse
import json
from pathlib import Path

import numpy as np


SAMPLE = Path(__file__).resolve().parent / "data" / "record.json"
FIELDS = ('axes', 'owner_axes', 'nodes', 'blocks', 'replacements', 'query', 'scale')


def load_record(path=SAMPLE):
    """读取原始数据并检查顶层交换字段。"""
    with Path(path).open(encoding="utf-8") as handle:
        record = json.load(handle)
    if not isinstance(record, dict):
        raise ValueError("输入应为一个 JSON 对象")
    missing = set(FIELDS).difference(record)
    if missing:
        raise ValueError("缺少交换字段：" + "、".join(sorted(missing)))
    return record


def _demo(record):
    """在样本记录上跑一遍查询、反向、替换与证明复得。"""
    from sectornet import Engine, Query

    engine = Engine.from_record(record)
    query = Query.from_record(record["query"])
    result = engine.run(query)
    print("== 查询 ==")
    for comp in result.components:
        line = "分量 %d 扇区 %s 坐标 %s：%s" % (
            comp.index, comp.sector_key, list(comp.coordinates), comp.status)
        if comp.value is not None:
            line += " 值=%g" % comp.value
        if comp.reason:
            line += "（%s）" % comp.reason
        print(line)
    print("标量损失（可用分量之和）：%g" % result.loss)

    grads = engine.backward(result)
    print("== 反向（基础块原坐标梯度） ==")
    for (block, sector), grad in sorted(grads.grads.items()):
        print("%s/%s：%s" % (block, sector, np.array2string(grad, precision=4)))
    report = grads.eligibility()
    for param in report["parameters"]:
        if not param["received_gradient"]:
            print("未收到梯度：%(block)s/%(sector)s —— %(reason)s" % param)

    print("== 替换参数源并重查 ==")
    engine2 = engine
    for replacement in record.get("replacements", []):
        engine2 = engine2.with_replacement(
            replacement["source"], replacement["sector"], replacement["values"])
    result2 = engine2.run(query)
    for comp in result2.components:
        if comp.value is not None:
            print("分量 %d 扇区 %s：值=%g" % (comp.index, comp.sector_key, comp.value))
    print("替换后标量损失：%g" % result2.loss)

    print("== 输出证明复得 ==")
    proof = result.proof()
    print("旧输入复得旧证明：%s" % engine.verify(proof).ok)
    changed = [b for b in engine2.verify(proof).blocks if not b["match"]]
    for block in changed:
        print("新输入下块身份变化：%(block)s/%(sector)s" % block)


def main():
    parser = argparse.ArgumentParser(description="查看本地交换样本的项目")
    parser.add_argument("path", nargs="?", type=Path, default=SAMPLE, help="JSON 记录路径")
    parser.add_argument("--demo", action="store_true", help="在样本上运行查询/反向/替换演示")
    args = parser.parse_args()
    record = load_record(args.path)
    print(json.dumps({"记录": args.path.name, "输入项目": sorted(record), "NumPy": np.__version__},
                     ensure_ascii=False))
    if args.demo:
        _demo(record)


if __name__ == "__main__":
    main()
