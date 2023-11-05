import numpy as np
import pandas as pd

from typing import List, Tuple

from empyrical import (
    sharpe_ratio,
    calmar_ratio,
    sortino_ratio,
    max_drawdown,
    downside_risk,
    annual_return,
    annual_volatility
)

VOL_LOOKBACK = 60
VOL_TARGET = 0.15

