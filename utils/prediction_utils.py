# utils/prediction_utils.py

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from utils.feature_engineering import create_environment_features


# ============================================================
# 1. 모델 및 Feature 불러오기
# ============================================================

def load_prediction_model(
    model_path: str | Path,
    feature_path: str | Path,
):
    """
    저장된 최종 분류모델과
    학습에 사용한 Feature 목록을 불러온다.
    """

    model_path = Path(model_path)
    feature_path = Path(feature_path)

    if not model_path.exists():
        raise FileNotFoundError(
            f"모델 파일이 없습니다: {model_path}"
        )

    if not feature_path.exists():
        raise FileNotFoundError(
            f"Feature 파일이 없습니다: {feature_path}"
        )

    model = joblib.load(model_path)
    feature_cols = joblib.load(feature_path)

    if not isinstance(feature_cols, (list, tuple)):
        raise TypeError(
            "feature_cols 파일의 형식이 올바르지 않습니다."
        )

    feature_cols = list(feature_cols)

    return model, feature_cols


# ============================================================
# 2. 실시간 환경 데이터 전처리
# ============================================================

def prepare_realtime_data(
    weather: pd.DataFrame,
    air: pd.DataFrame,
) -> pd.DataFrame:
    """
    최근 기상자료와 대기환경 자료를 병합하고,
    학습 때와 동일한 방식으로 전처리·파생변수 생성.

    Parameters
    ----------
    weather
        최근 기상 데이터

    air
        최근 대기오염 데이터

    Returns
    -------
    realtime_df
        파생변수가 포함된 최근 환경 데이터
    """

    if weather is None or weather.empty:
        raise ValueError(
            "기상 데이터가 없습니다."
        )

    if air is None:
        air = pd.DataFrame()

    weather = weather.copy()
    air = air.copy()

    # --------------------------------------------------------
    # 날짜 형식 통일
    # --------------------------------------------------------

    weather["date"] = pd.to_datetime(
        weather["date"],
        errors="coerce",
    ).dt.floor("D")

    weather = (
        weather
        .dropna(subset=["date"])
        .sort_values("date")
        .reset_index(drop=True)
    )

    if not air.empty:

        air["date"] = pd.to_datetime(
            air["date"],
            errors="coerce",
        ).dt.floor("D")

        air = (
            air
            .dropna(subset=["date"])
            .sort_values("date")
            .reset_index(drop=True)
        )

    # --------------------------------------------------------
    # 기상 + 대기오염 병합
    # --------------------------------------------------------

    realtime_df = pd.merge(
        weather,
        air,
        on="date",
        how="left",
    )

    realtime_df = (
        realtime_df
        .sort_values("date")
        .reset_index(drop=True)
    )

    if realtime_df.empty:
        raise ValueError(
            "기상·대기환경 병합 데이터가 없습니다."
        )

    # ========================================================
    # 이상치 처리
    # ========================================================

    non_negative_cols = [
        "rainfall",
        "wind_speed",
        "sunshine_hours",
        "pm10",
        "pm25",
        "humidity",
        "o3",
        "no2",
        "co",
        "so2",
    ]

    for col in non_negative_cols:

        if col in realtime_df.columns:

            realtime_df.loc[
                realtime_df[col] < 0,
                col,
            ] = np.nan

    # --------------------------------------------------------
    # 상대습도 범위
    # --------------------------------------------------------

    if "humidity" in realtime_df.columns:

        realtime_df.loc[
            (
                realtime_df["humidity"] < 0
            )
            |
            (
                realtime_df["humidity"] > 100
            ),
            "humidity",
        ] = np.nan

    # --------------------------------------------------------
    # PM 극단값
    # --------------------------------------------------------

    if "pm10" in realtime_df.columns:

        realtime_df.loc[
            realtime_df["pm10"] > 1000,
            "pm10",
        ] = np.nan

    if "pm25" in realtime_df.columns:

        realtime_df.loc[
            realtime_df["pm25"] > 500,
            "pm25",
        ] = np.nan

    # --------------------------------------------------------
    # 강수량 결측값
    # --------------------------------------------------------

    if "rainfall" in realtime_df.columns:

        realtime_df["rainfall"] = (
            realtime_df["rainfall"]
            .fillna(0)
        )

    # ========================================================
    # 과거값 기반 Forward Fill
    # ========================================================

    impute_cols = [
        "temp_avg",
        "temp_max",
        "temp_min",
        "humidity",
        "wind_speed",
        "sunshine_hours",
        "ground_temp",
        "pm10",
        "pm25",
        "o3",
        "no2",
        "co",
        "so2",
    ]

    cols_to_fill = [
        col
        for col in impute_cols
        if col in realtime_df.columns
    ]

    realtime_df[cols_to_fill] = (
        realtime_df[
            cols_to_fill
        ]
        .ffill()
    )

    # --------------------------------------------------------
    # 시작구간 결측 행 제거
    # --------------------------------------------------------

    realtime_df = (
        realtime_df
        .dropna(
            subset=cols_to_fill
        )
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # rolling(28) 계산에 필요한 최소 기간 확인
    # --------------------------------------------------------

    if len(realtime_df) < 28:

        raise ValueError(
            "사용 가능한 최근 환경 데이터가 "
            "28일 미만입니다. "
            "28일 파생변수를 계산할 수 없습니다."
        )

    # ========================================================
    # 학습과 동일한 파생변수 생성
    # ========================================================

    realtime_df = create_environment_features(
        realtime_df,
        fill_remaining_numeric=False,
    )

    return realtime_df


# ============================================================
# 3. 지정 분석일 환경 데이터 선택
# ============================================================

def select_analysis_environment(
    realtime_df: pd.DataFrame,
    target_date,
) -> pd.DataFrame:
    """
    사용자가 선택한 날짜의 환경 데이터 1행을 선택한다.

    지정일 자료가 없다고 해서
    임의로 이전 날짜를 대신 사용하지 않는다.
    """

    analysis_date = pd.Timestamp(
        target_date
    ).floor("D")

    realtime_df = realtime_df.copy()

    realtime_df["date"] = (
        pd.to_datetime(
            realtime_df["date"]
        )
        .dt.floor("D")
    )

    latest_environment = (
        realtime_df.loc[
            realtime_df["date"]
            == analysis_date
        ]
        .copy()
    )

    if latest_environment.empty:

        available_min = (
            realtime_df["date"]
            .min()
            .strftime("%Y-%m-%d")
        )

        available_max = (
            realtime_df["date"]
            .max()
            .strftime("%Y-%m-%d")
        )

        raise ValueError(
            f"지정한 분석일 "
            f"{analysis_date:%Y-%m-%d}의 "
            f"환경 데이터가 없습니다.\n"
            f"현재 확보된 데이터 기간: "
            f"{available_min} ~ {available_max}"
        )

    return latest_environment


# ============================================================
# 4. 문화재 특성 데이터 정리
# ============================================================

def prepare_heritage_data(
    heritage_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    문화재 특성 데이터 컬럼명을 통일하고 검증한다.
    """

    heritage_df = heritage_df.copy()

    rename_map = {
        "재질": "material",
        "노출형태": "exposure",
        "문화재명(국문)": "heritage_name",
        "문화재명": "heritage_name",
    }

    heritage_df = heritage_df.rename(
        columns=rename_map
    )

    required_cols = [
        "heritage_name",
        "material",
        "exposure",
    ]

    missing_cols = [
        col
        for col in required_cols
        if col not in heritage_df.columns
    ]

    if missing_cols:

        raise ValueError(
            "문화재 특성 데이터에 "
            "필요한 컬럼이 없습니다: "
            f"{missing_cols}"
        )

    return heritage_df


# ============================================================
# 5. 지정일 환경자료 × 실제 문화재 결합
# ============================================================

def combine_environment_with_heritage(
    latest_environment: pd.DataFrame,
    heritage_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    동일 날짜의 영천 환경정보를
    각 문화재의 재질·노출환경 정보와 결합한다.
    """

    if latest_environment.empty:

        raise ValueError(
            "분석일 환경 데이터가 없습니다."
        )

    heritage_df = prepare_heritage_data(
        heritage_df
    )

    environment_row = (
        latest_environment.iloc[0]
    )

    predict_df = (
        heritage_df.copy()
    )

    for col in latest_environment.columns:

        # 문화재별 material / exposure는 유지
        if col not in [
            "material",
            "exposure",
        ]:

            predict_df[col] = (
                environment_row[col]
            )

    return predict_df


# ============================================================
# 6. 머신러닝 예측 입력 Feature 생성
# ============================================================

def build_inference_features(
    predict_df: pd.DataFrame,
    feature_cols: list[str],
) -> pd.DataFrame:
    """
    범주형 변수를 One-Hot Encoding한 후,
    학습 때 사용한 Feature 순서에 맞춘다.
    """

    data = predict_df.copy()

    categorical_cols = []

    for col in [
        "material",
        "exposure",
        "season",
    ]:

        if col in data.columns:
            categorical_cols.append(col)

    data_encoded = pd.get_dummies(
        data,
        columns=categorical_cols,
        dtype=int,
    )

    # --------------------------------------------------------
    # 학습 Feature 순서에 정확히 맞춤
    # --------------------------------------------------------

    X_inference = (
        data_encoded
        .reindex(
            columns=feature_cols,
            fill_value=0,
        )
        .copy()
    )

    # --------------------------------------------------------
    # 수치형 검증
    # --------------------------------------------------------

    for col in X_inference.columns:

        X_inference[col] = (
            pd.to_numeric(
                X_inference[col],
                errors="coerce",
            )
        )

    remaining_nan = int(
        X_inference
        .isna()
        .sum()
        .sum()
    )

    if remaining_nan > 0:

        raise ValueError(
            "예측 입력 데이터에 "
            f"{remaining_nan}개의 결측값이 남아 있습니다."
        )

    if list(X_inference.columns) != list(feature_cols):

        raise ValueError(
            "학습 Feature와 "
            "예측 Feature 순서가 일치하지 않습니다."
        )

    return X_inference


# ============================================================
# 7. 최종 예측
# ============================================================

def run_model_prediction(
    model,
    X_inference: pd.DataFrame,
    predict_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    안전 / 주의 / 위험 등급과
    예측 확률을 계산한다.
    """

    result_df = predict_df.copy()

    pred_target = model.predict(
        X_inference
    )

    pred_proba = model.predict_proba(
        X_inference
    )

    result_df["pred_target"] = (
        pred_target
    )

    # --------------------------------------------------------
    # 모델 클래스 순서
    # --------------------------------------------------------

    class_to_index = {
        class_name: index
        for index, class_name
        in enumerate(
            model.classes_
        )
    }

    # --------------------------------------------------------
    # 실제 예측된 등급의 확률
    # --------------------------------------------------------

    pred_probability = []

    for row_idx, label in enumerate(
        pred_target
    ):

        class_idx = (
            class_to_index[label]
        )

        probability = (
            pred_proba[
                row_idx,
                class_idx,
            ]
        )

        pred_probability.append(
            probability
        )

    result_df[
        "pred_probability"
    ] = pred_probability

    # --------------------------------------------------------
    # 각 클래스 확률 저장
    # --------------------------------------------------------

    for class_name in model.classes_:

        class_idx = (
            class_to_index[
                class_name
            ]
        )

        result_df[
            f"prob_{class_name}"
        ] = (
            pred_proba[
                :,
                class_idx,
            ]
        )

    return result_df


# ============================================================
# 8. 결과 화면용 데이터 정리
# ============================================================

def make_result_view(
    prediction_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Streamlit 화면에서 보여줄 컬럼을 정리한다.
    """

    result_df = prediction_df.copy()

    # 확률 → %
    result_df["예측확률"] = (
        result_df[
            "pred_probability"
        ]
        * 100
    ).round(1)

    rename_map = {
        "heritage_name": "문화재명",
        "material": "재질",
        "exposure": "노출환경",
        "pred_target": "예측 위험도",
    }

    result_df = (
        result_df.rename(
            columns=rename_map
        )
    )

    display_cols = [
        col
        for col in [
            "문화재명",
            "재질",
            "노출환경",
            "예측 위험도",
            "예측확률",
        ]
        if col in result_df.columns
    ]

    return result_df[
        display_cols
    ].copy()


# ============================================================
# 9. 안전 / 주의 / 위험 건수
# ============================================================

def summarize_prediction(
    prediction_df: pd.DataFrame,
) -> dict:
    """
    안전·주의·위험 개수를 반환한다.
    """

    if "pred_target" not in prediction_df.columns:

        raise ValueError(
            "pred_target 컬럼이 없습니다."
        )

    counts = (
        prediction_df[
            "pred_target"
        ]
        .value_counts()
    )

    return {
        "안전": int(
            counts.get(
                "안전",
                0,
            )
        ),
        "주의": int(
            counts.get(
                "주의",
                0,
            )
        ),
        "위험": int(
            counts.get(
                "위험",
                0,
            )
        ),
        "전체": int(
            len(prediction_df)
        ),
    }


# ============================================================
# 10. 우선점검 문화재
# ============================================================

def get_attention_heritage(
    prediction_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    주의 또는 위험으로 예측된 문화재만 추출한다.
    """

    if "pred_target" not in prediction_df.columns:

        raise ValueError(
            "pred_target 컬럼이 없습니다."
        )

    attention_df = (
        prediction_df[
            prediction_df[
                "pred_target"
            ].isin(
                [
                    "주의",
                    "위험",
                ]
            )
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    return attention_df


# ============================================================
# 11. 예측 전체 과정 실행
# ============================================================

def predict_heritage_risk(
    model,
    feature_cols: list[str],
    weather: pd.DataFrame,
    air: pd.DataFrame,
    heritage_df: pd.DataFrame,
    target_date,
):
    """
    웹앱에서 가장 간단하게 호출하기 위한 통합 함수.

    Returns
    -------
    prediction_df
        전체 예측 결과

    result_view
        화면 표시용 결과

    summary
        안전/주의/위험 개수

    attention_df
        주의/위험 문화재
    """

    # 1. 최근 환경 데이터 전처리
    realtime_df = prepare_realtime_data(
        weather,
        air,
    )

    # 2. 선택 날짜 환경정보
    latest_environment = (
        select_analysis_environment(
            realtime_df,
            target_date,
        )
    )

    # 3. 문화재와 결합
    predict_df = (
        combine_environment_with_heritage(
            latest_environment,
            heritage_df,
        )
    )

    # 4. 학습 Feature와 구조 맞춤
    X_inference = (
        build_inference_features(
            predict_df,
            feature_cols,
        )
    )

    # 5. AI 예측
    prediction_df = (
        run_model_prediction(
            model,
            X_inference,
            predict_df,
        )
    )

    # 6. 화면 출력용
    result_view = (
        make_result_view(
            prediction_df
        )
    )

    # 7. 안전/주의/위험 개수
    summary = (
        summarize_prediction(
            prediction_df
        )
    )

    # 8. 우선 점검 대상
    attention_df = (
        get_attention_heritage(
            prediction_df
        )
    )

    return (
        prediction_df,
        result_view,
        summary,
        attention_df,
    )
