"""Project-wide invariants for the daily price-volume factor replication."""

PROJECT_NAME = "A股日频量价选股因子复现"
DATA_FREQUENCY = "1d"
ALLOW_INTRADAY_DATA = False

DAILY_PRICE_FIELDS = (
    "open",
    "high",
    "low",
    "close",
    "volume",
    "money",
    "high_limit",
    "low_limit",
    "paused",
)

# The report-specific sample, universe, factor windows and trading assumptions
# must be filled only after docs/factor_spec.md is confirmed.
START_DATE = None
END_DATE = None
UNIVERSE = None
REBALANCE_FREQUENCY = None
FACTOR_NAMES = ()

