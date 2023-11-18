import os

import numpy as np
import pandas as pd
from copy import deepcopy
from scipy.stats.mstats import winsorize
from pandas.tseries.offsets import BDay()


def deep_momentum_strategy_features(df_asset: pd.DataFrame) -> pd.DataFrame:

    # Filter data for NaN, None or Zero.
    df_asset = deepcopy(df_asset[
        ~df_asset['close'].isna() |
        df_asset['close'].is_null() |
        (df_asset['close'] > 1e-8)
    ])

    df_asset['srs'] = df_asset['close']
    df_asset['raw_daily_returns'] = df_asset['close'].diff()
    df_asset['raw_daily_vol'] = (
        df_asset['raw_daily_returns'].ewm(span=18, min_periods=60).fillna(method='ffill')  # Check the index
    )
    df_asset['vol_scaled_returns'] = (
        df_asset['raw_daily_retuns'].div(df_asset['raw_daily_vol'].shift(-1), axis=0)
    )
    vol_ewm = df_asset['vol_scaled_returns'].ewm(span=18)
    df_asset['winsorized_vol_returns'] = np.minimum(df_asset['vol_scaled_returns'], df_asset['vol_scaled_returns'] - 4.2 * vol_ewm.std())
    df_asset['winsorized_vol_returns'] = np.maximum(df_asset['vol_scaled_returns'], df_asset['vol_scaled_returns'] + 4.2 * vol_ewm.std())

    def normalised_business_day_return(business_days):
        return (
            (df_asset['close'] - df_asset['close'].shift(business_days, freq=business_days))
            / (df_asset['raw_daily_vol'] * np.sqrt(business_days))
        )
    
    for business_day_count in [1, 5, 21, 252]:
        df_asset[f'normalised_{business_day_count}_return'] = normalised_business_day_return(business_day_count)

    