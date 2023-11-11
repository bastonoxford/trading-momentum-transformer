import numpy as np
import pandas as pd
import datetime as dt
import enum

from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler

class DataTypes(enum.IntEnum):

    REAL_VALUED = 0
    CATEGORICAL = 1
    DATE = 2

class InputTypes(enum.IntEnum):

    TARGET = 0
    OBSERVED_INPUT = 1
    KNOWN_INPUT = 2
    STATIC_INPUT = 3
    ID = 4
    TIME = 5

def get_single_column_by_input_type(input_type, column_definition):


    l_var = [tup[0] for tup in column_definition if tup[2] == input_type]

    if len(l_var) != 1:
        raise ValueError(f"Invalid number of columns for {input_type}")
    
    return l_var[0]


def extract_cols_from_data_type(data_type, column_definition, excluded_input_types):
    return [
        tup[0] for tup in column_definition 
        if tup[1] == data_type and tup[2] not in excluded_input_types
        ]


class ModelFeatures:

    def __init__(
            self,
            df,
            total_time_steps,
            start_boundary=1990,
            test_boundary=2020,
            test_end=2021,
            changepoint_lbws=None,
            train_valid_sliding=False,
            transform_real_inputs=False,
            train_valid_ratio=0.9,
            split_tickers_individually=True,
            add_ticker_as_static=False,
            time_features=False,
            lags=None,
            asset_class_dictionary=None,
            static_ticker_type_feature=False,
    ):

        self._column_definition = [
                ("ticker", DataTypes.CATEGORICAL, InputTypes.ID),
                ("date", DataTypes.DATE, InputTypes.TIME),
                ("target_returns", DataTypes.REAL_VALUED, InputTypes.TARGET),
                ("norm_daily_return", DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
                ("norm_monthly_return", DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
                ("norm_quarterly_return", DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
                ("norm_biannual_return", DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
                ("norm_annual_return", DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
                ("macd_8_24", DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
                ("macd_16_48", DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
                ("macd_32_96", DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
        ]

        df = df.dropna()
        df = df[df["year"] >= start_boundary].copy()
        years = df["years"]

        self.identifiers = None
        self._real_scalers = None
        self._cat_scalers = None
        self._target_scaler = None
        self._num_classes_per_cat_input = None
        self.total_time_steps = total_time_steps
        self.lags = lags


        if changepoint_lbws:
            for lbw in changepoint_lbws:
                self._column_definition.append(
                    (f"cp_score_{lbw}", DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT)
                )
                self._column_definition.append(
                    (f"cp_rl_{lbw}", DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT)
                )


        if time_features:
            self._column_definition.append(
                ("days_from_start", DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT)
            )
            self._column_definition.append(
                ("day_of_week", DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT)
            )
            self._column_definition.append(
                ("day_of_month", DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT)
            )
            self._column_definition.append(
                ("week_of_year", DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT)
            )
            self._column_definition.append(
                ("month_of_year", DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT)
            )

            # dataframe could have later years
            start_date = dt.datetime(start_boundary, 1, 1)
            days_from_start_max = (dt.datetime(test_end - 1, 12, 31) - start_date).days
            df["days_from_start"] = (df.index - start_date).days
            df["days_from_start"] = np.minimum(
                df["days_from_start"], days_from_start_max
            )

            df["days_from_start"] = (
                MinMaxScaler().fit_transform(df[["days_from_start"]].values).flatten()
            )
            df["day_of_week"] = (
                MinMaxScaler().fit_transform(df[["day_of_week"]].values).flatten()
            )
            df["day_of_month"] = (
                MinMaxScaler().fit_transform(df[["day_of_month"]].values).flatten()
            )
            df["week_of_year"] = (
                MinMaxScaler().fit_transform(df[["week_of_year"]].values).flatten()
            )
            df["month_of_year"] = (
                MinMaxScaler().fit_transform(df[["month_of_year"]].values).flatten()
            )
        
        if add_ticker_as_static:
            self._column_definition.append(
                ("static_ticker", DataTypes.CATEGORICAL, InputTypes.STATIC_INPUT)
            )
            df["static_ticker"] = df["ticker"]
            if static_ticker_type_feature:
                df["static_ticker_type"] = df["ticker"].map(
                    lambda t: asset_class_dictionary[t]
                )
                self._column_definition.append(
                    ("static_ticker_type",  DataTypes.CATEGORICAL, InputTypes.STATIC_INPUT)
                )

        self.transform_real_inputs = transform_real_inputs
        
        test = df.loc[years >= test_boundary]

        if split_tickers_individually:
            trainvalid = df.loc[years < test_boundary]
            if lags:
                tickers = (
                    trainvalid.groupby("ticker")["ticker"].count() * (1.0 - train_valid_ratio)
                ) >= total_time_steps
                tickers = tickers[tickers].index.tolist()

            else:
                tickers = list(trainvalid.ticker.unique())

            train, valid = [], []

            for ticker in tickers:
                calib_data = trainvalid[trainvalid.ticker == ticker]
                T = len(calib_data)
                train_valid_split = int(train_valid_ratio * T)
                train.append(calib_data.iloc[:train_valid_split].copy())
                valid.append(calib_data.iloc[train_valid_split:].copy())
            
            train = pd.concat(train)
            valid = pd.concat(valid)

            test = test[test.ticker.isin(tickers)]

        else:
            trainvalid = df.loc[years < test_boundary]
            dates = np.sort(trainvalid.index.unique())
            split_index = int(train_valid_ratio * len(dates))
            train_dates = pd.DataFrame({"date": dates[:split_index]})
            valid_dates = pd.DataFrame({"date": dates[split_index:]})

            train = (
                trainvalid.reset_index()
                .merge(train_dates, on="date")
                .set_index("date").copy()
            )

            valid = (
                trainvalid.reset_index()
                .merge(valid_dates, on="date")
                .set_index("date").copy()
            )

            if lags:
                tickers = (
                    valid.groupby("ticker")["ticker"].count() > self.total_time_steps
                )
                tickers = tickers[tickers].index.tolist()
                train =train[train.ticker.isin(tickers)]

            else:
                tickers = list(train.ticker.unique())
            
            valid = valid[valid.ticker.isin(tickers)]
            test = test[test.tickers.isin(tickers)]

        # don't think this is needed...
        if test_end:
            test = test[test["year"] < test_end]

        test_with_buffer = pd.concat(
            [
                pd.concat(
                    [
                        trainvalid[trainvalid.ticker == t].iloc[
                            -(self.total_time_steps - 1) :
                        ],  # TODO this
                        test[test.ticker == t],
                    ]
                ).sort_index()
                for t in tickers
            ]
        )

        # to deal with case where fixed window did not have a full sequence
        if lags:
            for t in tickers:
                test_ticker = test[test["ticker"] == t]
                diff = len(test_ticker) - self.total_time_steps
                if diff < 0:
                    test = pd.concat(
                        [trainvalid[trainvalid["ticker"] == t][diff:], test]
                    )
                    # maybe should sort here but probably not needed

        self.tickers = tickers
        self.num_tickers = len(tickers)
        self.set_scalers(train)

        train, valid, test, test_with_buffer = [
            self.transform_inputs(data)
            for data in [train, valid, test, test_with_buffer]
        ]

        if lags:
            self.train = self._batch_data_smaller_output(
                train, train_valid_sliding, self.lags
            )
            self.valid = self._batch_data_smaller_output(
                valid, train_valid_sliding, self.lags
            )
            self.test_fixed = self._batch_data_smaller_output(test, False, self.lags)
            self.test_sliding = self._batch_data_smaller_output(
                test_with_buffer, True, self.lags
            )
        else:
            self.train = self._batch_data(train, train_valid_sliding)
            self.valid = self._batch_data(valid, train_valid_sliding)
            self.test_fixed = self._batch_data(test, False)
            self.test_sliding = self._batch_data(test_with_buffer, True)


