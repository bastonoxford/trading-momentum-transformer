import numpy as np
import pandas as pd

from typing import List, Tuple, Dict

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

def calc_preformance_metrics(data: pd.DataFrame, metric_suffix="", num_identifiers = None) -> Dict:
    """
    Calculate performance metrics for evaluating a strategy
    
    Parameters
    ----------
    data: pd.DataFrame
        dataframe containing returns, indexed by date

    Returns
    ---------
    Dict
        a dictionary of performance metrics
    """
    if not num_identifiers:
        num_identifiers = len(data.dropna()["identifier"].unique())
    srs = data.dropna().groupby(level=0)["captured_returns"].sum()/num_identifiers
    return {
        f"annual_return{metric_suffix}": annual_return(srs),
        f"annual_volatility{metric_suffix}": annual_volatility(srs),
        f"sharpe_ratio{metric_suffix}": sharpe_ratio(srs),
        f"downside_risk{metric_suffix}": downside_risk(srs),
        f"sortino_ratio{metric_suffix}": sortino_ratio(srs),
        f"max_drawdown{metric_suffix}": -max_drawdown(srs),
        f"calmar_ratio{metric_suffix}": calmar_ratio(srs),
        f"perc_pos_return{metric_suffix}": len(srs[srs > 0.0]) / len(srs),
        f"profit_loss_ratio{metric_suffix}": np.mean(srs[srs > 0.0])
        / np.mean(np.abs(srs[srs < 0.0])),
    }

def calc_performance_metrics_subset(srs: pd.Series, metric_suffix="") -> dict:
    """Performance metrics for evaluating strategy

    Args:
        captured_returns (pd.Series): series containing captured returns, aggregated by date

    Returns:
        dict: dictionary of performance metrics
    """
    return {
        f"annual_return{metric_suffix}": annual_return(srs),
        f"annual_volatility{metric_suffix}": annual_volatility(srs),
        f"downside_risk{metric_suffix}": downside_risk(srs),
        f"max_drawdown{metric_suffix}": -max_drawdown(srs),
    }

def cal_net_returns(data: pd.DataFrame, list_basis_points: List[float], identifiers=None):

    if not identifiers:
        identifiers = data['identifier'].unique().tolist()
    cost = np.atleast_2d(list_basis_points) * 1e-4

    dfs = []
    for i in identifiers:
        data_slice = data[data['identifiers'] == i].reset_index(drop=True)
        annualised_vol = data_slice["daily_vol"] * np.sqrt(252)
        scaled_position = VOL_TARGET * data_slice["position"] / annualised_vol
        transaction_costs = scaled_position.diff().abs().fillna(0.0).to_frame().to_numpy() * cost
        net_captured_returns = data_slice[["captured_returns"]].to_numpy() - transaction_costs
        columns = list(map(lambda c: "captured_returns_"+str(c).replace(".", '_') + "bps", list_basis_points))
        dfs.append(pd.concat([data_slice, pd.DataFrame(net_captured_returns, columns=columns)], axis=1))
    return pd.concat(dfs).reset_index(drop=True)

def calc_sharpe_by_year(data: pd.DataFrame, suffixL str = None) -> Dict:
    """Sharpe ratio for each year in dataframe

    Parameters
    ----------
        data (pd.DataFrame): dataframe containing captured returns, indexed by date

    Returns
    -------
        dict: dictionary of Sharpe by year
    """
    if not suffix:
        suffix = ""
    
    data = data.copy()
    data["year"] = data.index.year

    sharpes = (
        data.dropna()[["year", "captured_returns"]]
        .groupby(level=0)
        .mean()
        .groupby("year")
        .apply(lambda y: sharpe_ratio(y["captured_returns"]))
    )

    sharpes.index = "shrpe_ratio_" + sharpes.index.map(int).map(str) + suffix

    return sharpes.to_dict()

def calc_returns(srs: pd.Series, day_offset: int = 1) -> pd.Series:
    """
    For each element of a pandas time-series srs:
        1) calculates the returns over the past number of days
        2) specified by offset

    Parameters
    ----------
        srs (pd.Series): time-series of prices
        day_offset (int, optional): number of days to calculate returns over. Defaults to 1.

    Returns
    -------
        pd.Series: series of returns
    """
    returns = srs / srs.shift(day_offset) - 1.0
    return returns

def calc_daily_vol(daily_returns):
    return(
        daily_returns.ewm(span=VOL_LOOKBACK, min_period=VOL_LOOKBACK).std().fillna(method='ffill')  # KW has as Backfill
    )


