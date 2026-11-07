"""Project-wide invariants for the daily price-volume factor replication.

复现对象：华泰证券《多因子系列之四：华泰单因子测试之动量类因子》(2016-12-20)
本文件是项目参数的唯一来源；任何与华泰原文不一致处在此显式记录。
"""

PROJECT_NAME = "A股日频量价选股因子复现"

REPORT = {
    "institution": "华泰证券",
    "title": "华泰多因子系列之四：华泰单因子测试之动量类因子",
    "date": "2016-12-20",
    "authors": "林晓明、陈烨",
}

# ---- 数据边界（不可变）----
DATA_FREQUENCY = "1d"
ALLOW_INTRADAY_DATA = False
PRICE_FQ = "pre"  # 前复权（华泰原文用 Wind 后复权，收益口径等价）

DAILY_PRICE_FIELDS = (
    "open", "high", "low", "close", "volume", "money",
    "high_limit", "low_limit", "paused",
)

# ---- 样本区间 ----
SAMPLE_START = "2005-04-29"   # 华泰原文回溯起点
SAMPLE_END = "2016-11-30"     # 华泰原文回溯终点
OOS_START = "2017-01-01"      # 样本外（项目自加，稳健性检验）
OOS_END = "2024-12-31"

# ---- 股票池与过滤（华泰原文口径）----
UNIVERSE = "all_a_shares"     # 全 A 股
BENCHMARK_INDEX = "000300.XSHG"  # 沪深 300
EXCLUDE_ST = True             # 剔除 ST
EXCLUDE_PT = True             # 剔除 PT
EXCLUDE_NEXT_DAY_SUSPENDED = True  # 剔除下期停牌
EXCLUDE_LIMIT_UP_DOWN = False      # 华泰原文不剔除涨跌停（忠实复现）
MIN_LISTING_DAYS = None            # 华泰原文不剔除次新

# ---- 调仓与收益标签（月频）----
REBALANCE_FREQUENCY = "monthly"           # 月末最后一个交易日核算因子
EXECUTION = "next_first_trading_day_close"  # 下月首个交易日收盘换仓
FORWARD_RETURN_WINDOW = "next_natural_month"  # 下一整个自然月超额收益(相对沪深300)

# ---- 预处理口径（华泰原文）----
MAD_MULTIPLE = 5.0            # 去极值：中位数 ± 5×MAD
STANDARDIZE = True            # Z-score 标准化
MISSING_FILL = 0.0            # 标准化后缺失值置 0
NEUTRALIZE = True             # IC 前对市值+行业回归取残差
INDUSTRY_LEVEL = "sw_l1"      # 申万一级（聚宽无中信一级，替代并记录）

# ---- 回归法（WLS）----
REGRESSION_WEIGHT = "sqrt_circulating_market_cap"  # 权重=流通市值平方根(Barra)

# ---- 分层回测 ----
GROUP_COUNTS = (5, 10)        # 分 5 层与 10 层
LONG_SHORT = (1, 5)           # 多空 = 买组合1、卖组合5

# ---- 因子（之四，13 个）----
FACTOR_WINDOWS_MONTHS = (1, 3, 6, 12)  # return_Nm 的 N（自然月，月末对月末）

# 换手率构造：聚宽无直接字段，用 volume×close/(流通市值×1e8)
TURNOVER_FORMULA = "volume * close / (circulating_market_cap * 1e8)"
