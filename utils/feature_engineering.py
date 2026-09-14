import numpy as np
import pandas as pd


def consecutive_days(condition: pd.Series) -> pd.Series:
    condition = pd.Series(
        condition,
        index=condition.index
    ).astype(bool)

    groups = (~condition).cumsum()

    return (
        condition.astype(int)
        .groupby(groups)
        .cumsum()
    )


def get_season(month: int) -> str:
    if month in (12, 1, 2):
        return "겨울"
    elif month in (3, 4, 5):
        return "봄"
    elif month in (6, 7, 8):
        return "여름"
    else:
        return "가을"


def create_environment_features(
    frame: pd.DataFrame,
    *,
    fill_remaining_numeric: bool = False
):

    df_feat = frame.copy()

    df_feat["date"] = pd.to_datetime(
        df_feat["date"]
    )

    df_feat = (
        df_feat
        .sort_values("date")
        .reset_index(drop=True)
    )

    # ------------------------------------------------
    # 기본 기상 파생변수
    # ------------------------------------------------

    df_feat["temp_range"] = (
        df_feat["temp_max"]
        - df_feat["temp_min"]
    )

    df_feat["temp_change"] = (
        df_feat["temp_avg"]
        .diff()
        .abs()
        .fillna(0)
    )

    df_feat["humidity_change"] = (
        df_feat["humidity"]
        .diff()
        .abs()
        .fillna(0)
    )

    df_feat["humidity_std3"] = (
        df_feat["humidity"]
        .rolling(
            window=3,
            min_periods=1
        )
        .std()
        .fillna(0)
    )

    df_feat["rainfall_7d"] = (
        df_feat["rainfall"]
        .rolling(
            window=7,
            min_periods=1
        )
        .sum()
    )

    # ------------------------------------------------
    # 일반 고습
    # ------------------------------------------------

    df_feat["rh_over_60"] = (
        df_feat["humidity"] > 60
    ).astype(int)

    df_feat["temp_over_20"] = (
        df_feat["temp_avg"] > 20
    ).astype(int)

    df_feat["rh60_days_7"] = (
        df_feat["rh_over_60"]
        .rolling(7, min_periods=1)
        .sum()
    )

    df_feat["rh60_days_28"] = (
        df_feat["rh_over_60"]
        .rolling(28, min_periods=1)
        .sum()
    )

    # ------------------------------------------------
    # 목조
    # ------------------------------------------------

    df_feat["rh75"] = (
        df_feat["humidity"] >= 75
    ).astype(int)

    df_feat["rh95"] = (
        df_feat["humidity"] >= 95
    ).astype(int)

    df_feat["rh75_days_7"] = (
        df_feat["rh75"]
        .rolling(7, min_periods=1)
        .sum()
    )

    df_feat["rh75_days_28"] = (
        df_feat["rh75"]
        .rolling(28, min_periods=1)
        .sum()
    )

    df_feat["rh95_days_7"] = (
        df_feat["rh95"]
        .rolling(7, min_periods=1)
        .sum()
    )

    df_feat["rh95_days_28"] = (
        df_feat["rh95"]
        .rolling(28, min_periods=1)
        .sum()
    )

    df_feat["rh75_consecutive_days"] = (
        consecutive_days(
            df_feat["humidity"] >= 75
        )
    )

    df_feat["rh95_consecutive_days"] = (
        consecutive_days(
            df_feat["humidity"] >= 95
        )
    )

    df_feat["wood_mold_condition"] = (
        (df_feat["humidity"] >= 75)
        & (df_feat["temp_avg"] >= 20)
        & (df_feat["temp_avg"] <= 30)
    ).astype(int)

    df_feat["wood_mold_days_7"] = (
        df_feat["wood_mold_condition"]
        .rolling(7, min_periods=1)
        .sum()
    )

    df_feat["wood_mold_days_28"] = (
        df_feat["wood_mold_condition"]
        .rolling(28, min_periods=1)
        .sum()
    )

    # ------------------------------------------------
    # 금속
    # ------------------------------------------------

    df_feat["rh70_metal"] = (
        df_feat["humidity"] >= 70
    ).astype(int)

    df_feat["rh70_days_7"] = (
        df_feat["rh70_metal"]
        .rolling(7, min_periods=1)
        .sum()
    )

    df_feat["rh70_days_28"] = (
        df_feat["rh70_metal"]
        .rolling(28, min_periods=1)
        .sum()
    )

    df_feat["rh70_consecutive_days"] = (
        consecutive_days(
            df_feat["humidity"] >= 70
        )
    )

    df_feat["metal_so2_humidity"] = (
        df_feat["rh70_metal"]
        * df_feat["so2"]
    )

    # ------------------------------------------------
    # 대기오염
    # ------------------------------------------------

    df_feat["pm_total"] = (
        df_feat["pm10"]
        + df_feat["pm25"]
    )

    df_feat["pm_load_3d"] = (
        df_feat["pm_total"]
        .rolling(3, min_periods=1)
        .sum()
    )

    df_feat["pm_load_7d"] = (
        df_feat["pm_total"]
        .rolling(7, min_periods=1)
        .sum()
    )

    df_feat["so2_ma7"] = (
        df_feat["so2"]
        .rolling(7, min_periods=1)
        .mean()
    )

    df_feat["no2_ma7"] = (
        df_feat["no2"]
        .rolling(7, min_periods=1)
        .mean()
    )

    df_feat["o3_ma7"] = (
        df_feat["o3"]
        .rolling(7, min_periods=1)
        .mean()
    )

    # ------------------------------------------------
    # 시간 특성
    # ------------------------------------------------

    df_feat["month"] = (
        df_feat["date"].dt.month
    )

    df_feat["season"] = (
        df_feat["month"]
        .apply(get_season)
    )

    # ------------------------------------------------
    # 결측값
    # ------------------------------------------------

    numeric_cols = (
        df_feat
        .select_dtypes(
            include=[np.number]
        )
        .columns
    )

    df_feat[numeric_cols] = (
        df_feat[numeric_cols]
        .ffill()
    )

    if fill_remaining_numeric:
        df_feat[numeric_cols] = (
            df_feat[numeric_cols]
            .fillna(0)
        )

    if (
        df_feat[numeric_cols]
        .isna()
        .sum()
        .sum()
        > 0
    ):
        raise ValueError(
            "파생변수에 결측값이 남아 있습니다."
        )

    return df_feat
