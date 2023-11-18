import os
import sys
sys.path.append(
    '/Users/anthonybaston/Rogue Projects/trading-momentum-transformer'
)
import numpy as np  # noqa
import pandas as pd  # noqa
from copy import deepcopy  # noqa
from pandas.tseries.offsets import BDay  # noqa
from anthony_settings.default import QUANDL_TICKERS  # noqa

def pull_quandl_ticker_data(ticker: str) -> pd.DataFrame:
    cwd = os.getcwd()
    parent_path = os.path.dirname(cwd)
    data = pd.read_csv(os.path.join(parent_path, 'data', 'quandl', f'{ticker}.csv'), parse_dates=[0])
    # data = pd.read_csv(os.path.join('data', 'quandl', f'{ticker}.csv'), parse_dates=[0])

    data.rename(columns={"Trade Date": 'Date', 'Date': 'Date', 'Settle': 'close'}, inplace=True)
    data.set_index('Date', inplace=True)
    data.replace(0.0, np.nan, inplace=True)
    return data

def calc_macd_signal(price: pd.Series, short_timescale: int, long_timescale: int) -> float:
    """Calculate MACD signal for a signal short/long timescale combination

    Args:
        price ([type]): series of prices
        short_timescale ([type]): short timescale
        long_timescale ([type]): long timescale

    Returns:
        float: MACD signal
    """

    def _calc_halflife(timescale):
        return np.log(0.5) / np.log(1 - 1 / timescale)

    macd = (
        price.ewm(halflife=_calc_halflife(short_timescale)).mean()
        - price.ewm(halflife=_calc_halflife(long_timescale)).mean()
    )
    # Need to think about oscnorm and changing this...
    q = macd / price.rolling(63).std().fillna(method="bfill")
    return q / q.rolling(252).std().fillna(method="bfill")

def compute_features_from_price(df_asset: pd.DataFrame) -> pd.DataFrame:

    # Filter data for NaN, None or Zero.
    df_asset = deepcopy(df_asset[
        ~df_asset['close'].isna() |
        df_asset['close'].isnull() |
        (df_asset['close'] > 1e-8)
    ])

    df_asset['raw_daily_returns'] = df_asset['close'].diff()
    df_asset['raw_daily_vol'] = (
        df_asset['raw_daily_returns'].ewm(span=18, min_periods=60).std().fillna(method='ffill')  # Check the index
    )
    df_asset['vol_scaled_returns'] = (
        df_asset['raw_daily_returns'].div(df_asset['raw_daily_vol'].shift(-1), axis=0)
    )
    vol_ewm = df_asset['vol_scaled_returns'].ewm(span=18)
    df_asset['winsorized_vol_returns'] = np.minimum(
        df_asset['vol_scaled_returns'], df_asset['vol_scaled_returns'] + 4.2 * vol_ewm.std()
        )
    df_asset['winsorized_vol_returns'] = np.maximum(
        df_asset['vol_scaled_returns'], df_asset['vol_scaled_returns'] - 4.2 * vol_ewm.std()
        )

    def normalised_business_day_return(business_days):
        return (
            (df_asset['close'] - df_asset['close'].shift(business_days))  # ideally would use freq = BDay() but this runs into problems
            / (df_asset['raw_daily_vol'] * np.sqrt(business_days))
        )
    
    for business_day_count in [1, 5, 21, 252]:
        df_asset[f'normalised_{business_day_count}_return'] = normalised_business_day_return(business_day_count)

    trend_combinations = [(8, 24), (16, 48), (32, 96)]
    for short_window, long_window in trend_combinations:
        df_asset[f'macd_{short_window}_{long_window}'] = calc_macd_signal(
            df_asset['close'], short_window, long_window
        )

    # date features
    if len(df_asset):
        df_asset["day_of_week"] = df_asset.index.dayofweek
        df_asset["day_of_month"] = df_asset.index.day
        df_asset["week_of_year"] = df_asset.index.isocalendar().week
        df_asset["month_of_year"] = df_asset.index.month
        df_asset["year"] = df_asset.index.year
        df_asset["date"] = df_asset.index  # duplication but sometimes makes life easier
    else:
        df_asset["day_of_week"] = []
        df_asset["day_of_month"] = []
        df_asset["week_of_year"] = []
        df_asset["month_of_year"] = []
        df_asset["year"] = []
        df_asset["date"] = []
        
    return df_asset.dropna()

def calc_perc_returns(price: pd.Series, day_offset: int = 1) -> pd.Series:
    
    rets = price / price.shift(day_offset) - 1.0
    return rets

# This amalgamates all the data into the same, single, large DataFrame
all_asset_features = pd.concat(
    [
        compute_features_from_price(
        pull_quandl_ticker_data(ticker)
        ).assign(ticker=ticker) for ticker in QUANDL_TICKERS
        ]
)


