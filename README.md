# 对称分块张量网络的受限收缩与反传

`data/record.json` 是 对称分块张量网络的受限收缩与反传 使用的本地合成交换样本。数组按 JSON 的下标顺序排列。本文的坐标、单位和枚举约定与该记录配套。记录中如果有 `scale` 或 `limits`，对应同一输入协议的部署规格。

axes 为轴在扇区 0、1 的长度。连接同名的两条轴，各轴最多出现两次。每个块以其轴顺序编码扇区，如 010。节点扇区异或须等于 parity，其他扇区为结构零。blocks 以 owner_axes 的顺序存储基础块。用 null 声明未到的数据，用字符串 zero 声明固定的结构零。节点的 own 指向基础块，axis_permutation 按输出轴位置给出原轴下标，indices 以节点轴顺序写扇区键，值为重排之后各轴要读取的坐标列表，可重复取位。未提供视图变换时使用原布局。查询的坐标按 open_axes 和 sectors 的对应位置解释，长度必须符合所查扇区。

## 读取样本

使用 Python 3.11 或 3.12，并在虚拟环境安装锁定依赖。读取入口返回原始记录。作为命令运行时，它会列出各顶层输入项目以及 NumPy 版本。

```sh
python -m pip install -r requirements.txt
python -m sector_contraction_input
python -m unittest discover -s tests -v
```

测试读取交换样本，核对其中的张量尺寸、标识与数据约定。
