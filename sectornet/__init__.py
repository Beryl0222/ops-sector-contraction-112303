"""对称分块张量网络的受限收缩与反传。

可复用的对称分块收缩和反向算子：模型训练和局部探针在同一套
轴命名、块等价（数据 / 缺失 / 结构零）及归属契约（own 与视图）
上运行；查询与反向都在进程实际临时张量预算内完成。
"""
from .engine import (
    ComponentResult,
    Engine,
    GradResult,
    Query,
    QueryResult,
    VerifyReport,
)
from .gf2 import AssignmentError
from .model import (
    KIND_DATA,
    KIND_MISSING,
    KIND_ZERO,
    BlockStore,
    Network,
    RecordError,
    network_from_record,
)
from .runtime import BudgetExceeded

__all__ = [
    "AssignmentError",
    "BlockStore",
    "BudgetExceeded",
    "ComponentResult",
    "Engine",
    "GradResult",
    "KIND_DATA",
    "KIND_MISSING",
    "KIND_ZERO",
    "Network",
    "Query",
    "QueryResult",
    "RecordError",
    "VerifyReport",
    "network_from_record",
]
