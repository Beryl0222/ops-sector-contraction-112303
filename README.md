# 对称分块张量网络的受限收缩与反传

`data/record.json` 是 对称分块张量网络的受限收缩与反传 使用的本地合成交换样本。数组按 JSON 的下标顺序排列。本文的坐标、单位和枚举约定与该记录配套。记录中如果有 `scale` 或 `limits`，对应同一输入协议的部署规格。

axes 为轴在扇区 0、1 的长度。连接同名的两条轴，各轴最多出现两次。每个块以其轴顺序编码扇区，如 010。节点扇区异或须等于 parity，其他扇区为结构零。blocks 以 owner_axes 的顺序存储基础块。用 null 声明未到的数据，用字符串 zero 声明固定的结构零。节点的 own 指向基础块，axis_permutation 按输出轴位置给出原轴下标，indices 以节点轴顺序写扇区键，值为重排之后各轴要读取的坐标列表，可重复取位。未提供视图变换时使用原布局。查询的坐标按 open_axes 和 sectors 的对应位置解释，长度必须符合所查扇区。

## 读取样本

使用 Python 3.11 或 3.12，并在虚拟环境安装锁定依赖。读取入口返回原始记录。作为命令运行时，它会列出各顶层输入项目以及 NumPy 版本；加 `--demo` 会在样本上完整跑一遍查询、反向、替换与证明复得。

```sh
python -m pip install -r requirements.txt
python -m sector_contraction_input --demo
python -m unittest discover -s tests -v
```

测试读取交换样本，核对其中的张量尺寸、标识与数据约定，并在样本记录、
特制记录与部署规格（18 节点、扇区长度 32、24MiB 临时预算）的合成记录
上验证引擎行为。

## sectornet：对称分块收缩与反向算子

`sectornet` 把模型训练和局部探针放在同一套轴命名、块等价及归属契约上运行：

- **轴与扇区**：每条轴两个扇区（0/1）各有长度；同名轴收缩，每轴至多出现两次。
- **块等价（三态）**：每个基础块扇区或为数据、或为 `null` 声明的缺失块、
  或为 `"zero"` 声明的固定结构零；奇偶不允许的扇区是不存储的结构零。
- **归属契约**：节点经 `own` 引用基础块，可带 `axis_permutation` 与
  `indices` 视图变换（可重复取位）；同一块可被多个节点、多个扇区、
  同一扇区的多个位置重复引用。

### 查询与标量损失

```python
from sector_contraction_input import load_record
from sectornet import Engine, Query

record = load_record()
engine = Engine.from_record(record)          # 预算取 record["scale"]["scratch_bytes"]
query = Query.from_record(record["query"])   # open_axes + 逐分量扇区键与局部坐标
result = engine.run(query)

for comp in result.components:               # 逐分量：available / undetermined / restricted
    print(comp.status, comp.value, comp.missing, comp.reason)
print(result.loss)                           # 可用分量加权和（Query.weights 可选）
result.available()                           # 只含可用扇区，可直接接下游
```

- 同一合法输出分量的值不随收缩路径或分段切法变化（浮点结合律误差之内；
  整数数据在 float64 下完全一致）。
- 合法结构零会消掉整条乘积依赖；缺失块若实际参与，对应输出标记为
  `undetermined` 并列出缺块，不会把满零结果交给训练。某在场因子恰好
  全零时，缺失块不参与取值，输出仍为确定值。
- 预算不足时对应输出标记为 `restricted`，原因给出所需字节与预算。

### 反向

```python
grads = engine.backward(result)              # 只从真正采用的输出回传
grads.for_block("S", "00")                   # 基础块原坐标下的梯度；未收到为 None
grads.status("S", "11")                      # (状态, 原因)
report = grads.eligibility()                 # 复查：输出可否训练、参数为何无梯度
```

- 梯度归还到每个原始参数块及数据块的原坐标：重排、切片、重复取位产生的
  全部贡献都加回基础块原坐标；两个节点引用同一块不同扇区、或反复引用
  同一扇区位置时贡献累加。
- 声明为结构零的位置是固定约束，没有可训练的参数，不收梯度；普通参数
  恰好为零仍可接收梯度。若某参数的梯度项取决于缺块（如所在乘积含缺失
  块而另一因子恰好全零），状态标记为 `undetermined`/`partial` 并说明。
- `eligibility()` 在换网络或改变请求的轴范围后回答：哪些输出可用于训练、
  哪些失去资格，以及某个基础参数为什么没有收到梯度。

### 临时张量预算

查询与反向都在进程实际临时张量预算内完成。计入预算的是执行中物化的
视图副本与中间结果；持久块存储与交付给调用者的结果（输出、梯度数组）
不计入。被请求输出很小、中间扇区却很大时，规划器沿轴分段（`Slice`），
分段后逐段收缩/反传，而不是以整表构造不起为由放弃；实在不足时把请求
标记为受限并给出所需字节。

### 替换参数源与输出证明

```python
engine2 = engine.with_replacement("B", "111", values)  # 新引擎；旧引擎不变
result2 = engine2.run(query)                           # 用新块身份与真实占用重新规划
proof = result.proof()                                 # 可 JSON 序列化的输出证明
engine.verify(proof).ok                                # 同一节点图下用旧输入复得
engine2.verify(proof).summary()                        # 报告哪些块身份已变化
```

证明记录扇区赋值、收缩计划与所用块身份；同一节点图下保存的证明仍可用
旧输入复得，换入新参数源后复得报告会指出身份变化的分量与块。

### 部署规格自查

`sectornet.testing.chain_record` 按 `scale` 规格（节点数、扇区数、扇区
长度、临时预算）生成链式合成记录，可用于在真实预算下验证分段收缩与
反向：`tests/test_scale.py` 即以此核对 18 节点、24MiB 预算下的数值
与稠密参照一致，以及大中间张量在分段前后结果一致。
