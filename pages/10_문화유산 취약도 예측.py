from __future__ import annotations

from pathlib import Path
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import json
import time
import urllib.parse

import joblib
from github import Github
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

from utils.feature_engineering import create_environment_features


# ============================================================
# 1. 페이지 설정
# ============================================================

st.set_page_config(
    page_title="문화유산 환경 취약도 예측",
    page_icon="🏛️",
    layout="wide",
)

st.title("🏛️ 영천 문화유산 환경 취약도 예측")
st.caption(
    "예측 버튼을 누르면 전일 기준 최근 40일 기상·대기환경 데이터를 자동 수집하고, "
    "파생변수 생성 후 학습된 분류모델로 문화유산별 환경 취약도를 "
    "안전·주의·위험으로 바로 예측합니다."
)

st.info(
    "📌 이 페이지의 '위험'은 실제 문화재 훼손 발생을 의미하지 않습니다. "
    "문헌 기반 환경조건과 프로젝트에서 정의한 상대 가중치를 이용해 만든 "
    "환경 취약도 등급을 분류한 결과입니다."
)


# ============================================================
# 2. 화면 디자인
# ============================================================

st.markdown(
    """
    <style>
    .prediction-hero {
        padding: 18px 22px;
        border-radius: 18px;
        border: 1px solid rgba(128,128,128,0.22);
        background: linear-gradient(
            135deg,
            rgba(52,152,219,0.08),
            rgba(155,89,182,0.07)
        );
        margin-bottom: 14px;
    }

    .prediction-hero h3 {
        margin: 0 0 6px 0;
        font-size: 1.25rem;
    }

    .prediction-hero p {
        margin: 0;
        opacity: 0.82;
    }

    .risk-safe {
        border-left: 6px solid #2ecc71;
        padding: 10px 14px;
        border-radius: 10px;
        background: rgba(46,204,113,0.08);
    }

    .risk-caution {
        border-left: 6px solid #f39c12;
        padding: 10px 14px;
        border-radius: 10px;
        background: rgba(243,156,18,0.08);
    }

    .risk-danger {
        border-left: 6px solid #e74c3c;
        padding: 10px 14px;
        border-radius: 10px;
        background: rgba(231,76,60,0.08);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 3. 파일 경로
# ============================================================

MODEL_DIR = Path("models")
DATA_DIR = Path("data/processed")

MODEL_PATH = MODEL_DIR / "best_model.pkl"
FEATURE_COLS_PATH = MODEL_DIR / "feature_cols.pkl"
TRAIN_MEDIANS_PATH = MODEL_DIR / "train_medians.pkl"
BUNDLE_PATH = MODEL_DIR / "heritage_risk_bundle.pkl"
MODEL_META_PATH = MODEL_DIR / "model_metadata.json"

# 실시간 대시보드가 항상 읽는 마지막 예측 결과 파일
LATEST_PREDICTION_PATH = DATA_DIR / "latest_prediction.csv"
LATEST_ENVIRONMENT_PATH = DATA_DIR / "latest_40_environment.csv"

HERITAGE_CANDIDATES = [
    DATA_DIR / "yc_heritage_feature.csv",
    DATA_DIR / "yc_heritage_detail_enriched.csv",
    DATA_DIR / "yc_heritage_features.csv",
]


# ============================================================
# 4. 상수
# ============================================================

GRADE_ORDER = ["안전", "주의", "위험"]

GRADE_COLOR = {
    "안전": "#2ECC71",
    "주의": "#F39C12",
    "위험": "#E74C3C",
}

MATERIAL_ORDER = [
    "석조",
    "목조",
    "금속",
    "회화",
    "기타",
]

EXPOSURE_ORDER = [
    "실외",
    "반실외",
    "실내",
]


# ============================================================
# 4-1. 최근 40일 환경 데이터 API 설정
# ============================================================

ASOS_URL = (
    "https://apis.data.go.kr/"
    "1360000/AsosDalyInfoService/getWthrDataList"
)

AIR_URL = (
    "https://apis.data.go.kr/"
    "B552584/ArpltnStatsSvc/getMsrstnAcctoRDyrg"
)

STN_ID = "281"  # 영천 ASOS

ASOS_SERVICE_KEY = st.secrets.get(
    "ASOS_SERVICE_KEY",
    st.secrets.get("SERVICE_KEY", ""),
)

AIR_SERVICE_KEY = st.secrets.get(
    "AIR_SERVICE_KEY",
    st.secrets.get("SERVICE_KEY", ""),
)

AIR_STATION_NAME = st.secrets.get(
    "AIR_STATION_NAME",
    "영천",
)

KST = ZoneInfo("Asia/Seoul")

DEFAULT_TARGET_DATE = (
    datetime.now(KST).date()
    - timedelta(days=1)
)


# ============================================================
# 5. 유틸리티
# ============================================================

def find_heritage_path() -> Path:
    for path in HERITAGE_CANDIDATES:
        if path.exists():
            return path

    raise FileNotFoundError(
        "문화유산 특성 데이터 파일을 찾을 수 없습니다.\n"
        "다음 중 하나의 파일이 필요합니다:\n"
        + "\n".join(
            f"- {path}"
            for path in HERITAGE_CANDIDATES
        )
    )


@st.cache_resource(show_spinner=False)
def load_model_assets():
    """
    최신 학습 결과 Bundle을 우선 사용한다.

    Bundle 구성:
    - model
    - features
    - train_medians
    - metadata

    이전 버전과의 호환을 위해 Bundle이 없으면
    best_model.pkl / feature_cols.pkl / train_medians.pkl /
    model_metadata.json을 각각 읽는다.
    """

    # --------------------------------------------------------
    # 1순위: 통합 Bundle
    # --------------------------------------------------------
    if BUNDLE_PATH.exists():
        try:
            bundle = joblib.load(
                BUNDLE_PATH
            )

            if not isinstance(
                bundle,
                dict,
            ):
                raise ValueError(
                    "모델 Bundle 형식이 올바르지 않습니다."
                )

            model = bundle.get(
                "model"
            )

            feature_cols = bundle.get(
                "features"
            )

            train_medians = bundle.get(
                "train_medians",
                {},
            )

            metadata = bundle.get(
                "metadata",
                {},
            )

            if model is None:
                raise ValueError(
                    "Bundle에 model이 없습니다."
                )

            if not feature_cols:
                raise ValueError(
                    "Bundle에 features가 없습니다."
                )

            if not isinstance(
                train_medians,
                dict,
            ):
                train_medians = {}

            if not isinstance(
                metadata,
                dict,
            ):
                metadata = {}

            return (
                model,
                list(feature_cols),
                train_medians,
                metadata,
            )

        except Exception as e:
            raise RuntimeError(
                f"통합 모델 Bundle 로드 실패: {e}"
            ) from e

    # --------------------------------------------------------
    # 2순위: 이전 버전 개별 파일
    # --------------------------------------------------------
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"최종 모델 파일이 없습니다: {MODEL_PATH}\n"
            "'환경 취약도 분류 모델 학습' 페이지에서 "
            "먼저 모델을 학습하세요."
        )

    if not FEATURE_COLS_PATH.exists():
        raise FileNotFoundError(
            f"Feature 목록 파일이 없습니다: "
            f"{FEATURE_COLS_PATH}"
        )

    model = joblib.load(
        MODEL_PATH
    )

    feature_cols = joblib.load(
        FEATURE_COLS_PATH
    )

    train_medians = {}

    if TRAIN_MEDIANS_PATH.exists():
        try:
            loaded_medians = joblib.load(
                TRAIN_MEDIANS_PATH
            )

            if isinstance(
                loaded_medians,
                dict,
            ):
                train_medians = loaded_medians

        except Exception:
            train_medians = {}

    metadata = {}

    if MODEL_META_PATH.exists():
        try:
            metadata = json.loads(
                MODEL_META_PATH.read_text(
                    encoding="utf-8"
                )
            )
        except Exception:
            metadata = {}

    return (
        model,
        feature_cols,
        train_medians,
        metadata,
    )


@st.cache_data(show_spinner=False)
def load_heritage_data(
    heritage_path_str: str,
) -> pd.DataFrame:

    heritage = pd.read_csv(
        heritage_path_str,
        encoding="utf-8-sig",
    )

    rename_candidates = {
        "문화재명(국문)": "heritage_name",
        "문화재명": "heritage_name",
        "국가유산명": "heritage_name",
        "유산명": "heritage_name",

        "재질": "material",
        "재질분류": "material",

        "노출형태": "exposure",
        "노출환경": "exposure",

        "위도": "latitude",
        "위도(latitude)": "latitude",
        "latitude": "latitude",
        "lat": "latitude",

        "경도": "longitude",
        "경도(longitude)": "longitude",
        "longitude": "longitude",
        "lon": "longitude",
        "lng": "longitude",

        "소재지": "address",
        "주소": "address",
        "소재지도로명주소": "address",

        "종목": "heritage_type",
        "문화재종목": "heritage_type",
        "국가유산종목": "heritage_type",
    }

    for old_col, new_col in rename_candidates.items():
        if (
            old_col in heritage.columns
            and new_col not in heritage.columns
        ):
            heritage = heritage.rename(
                columns={
                    old_col: new_col
                }
            )

    required = [
        "heritage_name",
        "material",
        "exposure",
    ]

    missing = [
        col
        for col in required
        if col not in heritage.columns
    ]

    if missing:
        raise ValueError(
            "문화유산 데이터에 필요한 컬럼이 없습니다: "
            f"{missing}\n\n"
            "필요 컬럼: 문화재명, 재질, 노출형태"
        )

    heritage["heritage_name"] = (
        heritage["heritage_name"]
        .astype(str)
        .str.strip()
    )

    heritage["material"] = (
        heritage["material"]
        .astype(str)
        .str.strip()
        .replace(
            {
                "벽화": "회화",
                "그림": "회화",
                "회화류": "회화",
            }
        )
    )

    heritage["material"] = heritage[
        "material"
    ].where(
        heritage["material"].isin(
            MATERIAL_ORDER
        ),
        "기타",
    )

    heritage["exposure"] = (
        heritage["exposure"]
        .astype(str)
        .str.strip()
        .replace(
            {
                "옥외": "실외",
                "야외": "실외",
                "반옥외": "반실외",
                "옥내": "실내",
            }
        )
    )

    heritage["exposure"] = heritage[
        "exposure"
    ].where(
        heritage["exposure"].isin(
            EXPOSURE_ORDER
        ),
        "실외",
    )

    if "latitude" in heritage.columns:
        heritage["latitude"] = pd.to_numeric(
            heritage["latitude"],
            errors="coerce",
        )

    if "longitude" in heritage.columns:
        heritage["longitude"] = pd.to_numeric(
            heritage["longitude"],
            errors="coerce",
        )

    return (
        heritage
        .drop_duplicates(
            subset=["heritage_name"],
            keep="first",
        )
        .reset_index(drop=True)
    )


def get_recent_environment():
    """
    바로 앞 '전일~40일전 데이터' 페이지에서 만든
    session_state 데이터를 읽는다.
    """

    realtime_df = st.session_state.get(
        "recent_40_environment"
    )

    target_date = st.session_state.get(
        "recent_40_target_date"
    )

    if (
        realtime_df is None
        or target_date is None
    ):
        return None, None

    if not isinstance(
        realtime_df,
        pd.DataFrame,
    ):
        return None, None

    realtime_df = realtime_df.copy()

    realtime_df["date"] = pd.to_datetime(
        realtime_df["date"],
        errors="coerce",
    )

    return realtime_df, target_date


def select_target_environment(
    realtime_df: pd.DataFrame,
    target_date,
) -> pd.DataFrame:
    """
    지정한 기준일의 환경 데이터만 정확히 선택.
    이전 날짜를 임의로 대신 사용하지 않는다.
    """

    target_ts = pd.Timestamp(
        target_date
    ).floor("D")

    target_rows = realtime_df.loc[
        realtime_df["date"].dt.floor("D")
        == target_ts
    ]

    if target_rows.empty:
        raise ValueError(
            f"{target_ts:%Y-%m-%d} 기준일의 "
            "환경 Feature가 없습니다."
        )

    return (
        target_rows
        .tail(1)
        .reset_index(drop=True)
    )


def combine_environment_and_heritage(
    target_environment: pd.DataFrame,
    heritage_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    기준일 환경 1행 × 문화유산 전체.
    """

    env_row = (
        target_environment
        .iloc[0]
        .to_dict()
    )

    rows = []

    for _, heritage_row in heritage_df.iterrows():
        combined = dict(
            env_row
        )

        for col in heritage_df.columns:
            combined[col] = heritage_row[
                col
            ]

        rows.append(
            combined
        )

    return pd.DataFrame(
        rows
    )


def build_inference_features(
    prediction_df: pd.DataFrame,
    feature_cols: list[str],
    train_medians: dict,
) -> pd.DataFrame:
    """
    학습 시 저장한 feature_cols와 완전히 동일한 열 순서로 변환.
    """

    work = prediction_df.copy()

    # 범주형 Feature를 학습과 동일하게 One-Hot
    categorical_cols = [
        col
        for col in [
            "material",
            "exposure",
            "season",
        ]
        if col in work.columns
    ]

    work = pd.get_dummies(
        work,
        columns=categorical_cols,
        dtype=int,
    )

    # 모델이 요구하는 열만 정확히 맞춤
    X = work.reindex(
        columns=feature_cols,
        fill_value=0,
    )

    # 모든 열을 숫자형으로 보장
    for col in X.columns:
        X[col] = pd.to_numeric(
            X[col],
            errors="coerce",
        )

    missing_cols = (
        X.columns[
            X.isna().any()
        ]
        .tolist()
    )

    if missing_cols:
        # ----------------------------------------------------
        # 학습 시 2019~2024 최종 학습 구간에서 계산한
        # Feature 중앙값으로 실시간 API 결측값 보정
        # ----------------------------------------------------
        for col in missing_cols:
            median_value = (
                train_medians.get(col)
                if isinstance(
                    train_medians,
                    dict,
                )
                else None
            )

            if pd.notna(
                median_value
            ):
                X[col] = X[col].fillna(
                    median_value
                )

        remaining_missing = (
            X.columns[
                X.isna().any()
            ]
            .tolist()
        )

        if remaining_missing:
            raise ValueError(
                "예측 Feature에 결측값이 남아 있습니다: "
                f"{remaining_missing}. "
                "모델 학습 페이지에서 다시 학습하여 "
                "heritage_risk_bundle.pkl을 생성해주세요."
            )

    return X


def get_class_probability(
    model,
    probability_matrix: np.ndarray,
    class_name: str,
) -> np.ndarray:
    """
    model.classes_ 순서와 무관하게 원하는 등급 확률 반환.
    해당 등급이 모델에 없다면 0 반환.
    """

    classes = list(
        model.classes_
    )

    if class_name not in classes:
        return np.zeros(
            probability_matrix.shape[0]
        )

    class_idx = classes.index(
        class_name
    )

    return probability_matrix[
        :,
        class_idx,
    ]


def run_prediction(
    model,
    feature_cols: list[str],
    train_medians: dict,
    prediction_df: pd.DataFrame,
) -> pd.DataFrame:

    X = build_inference_features(
        prediction_df,
        feature_cols,
        train_medians,
    )

    predicted = model.predict(
        X
    )

    result = prediction_df.copy()

    result[
        "predicted_grade"
    ] = predicted

    if hasattr(
        model,
        "predict_proba",
    ):
        probabilities = model.predict_proba(
            X
        )

        result[
            "safe_probability"
        ] = (
            get_class_probability(
                model,
                probabilities,
                "안전",
            )
            * 100
        )

        result[
            "caution_probability"
        ] = (
            get_class_probability(
                model,
                probabilities,
                "주의",
            )
            * 100
        )

        result[
            "danger_probability"
        ] = (
            get_class_probability(
                model,
                probabilities,
                "위험",
            )
            * 100
        )

    else:
        result["safe_probability"] = np.where(
            result["predicted_grade"] == "안전",
            100.0,
            0.0,
        )

        result["caution_probability"] = np.where(
            result["predicted_grade"] == "주의",
            100.0,
            0.0,
        )

        result["danger_probability"] = np.where(
            result["predicted_grade"] == "위험",
            100.0,
            0.0,
        )

    # "주의 이상 가능성"은 관심 순위를 위한 보조 지표
    result[
        "attention_probability"
    ] = (
        result[
            "caution_probability"
        ]
        + result[
            "danger_probability"
        ]
    )

    # 0~100의 시각화용 종합 점수
    # 안전=0, 주의=50, 위험=100으로 확률 가중
    result[
        "risk_index"
    ] = (
        result[
            "caution_probability"
        ]
        * 0.5
        + result[
            "danger_probability"
        ]
    ).clip(
        0,
        100,
    )

    return result


def make_display_result(
    result_df: pd.DataFrame,
) -> pd.DataFrame:

    columns = [
        "heritage_name",
        "material",
        "exposure",
        "predicted_grade",
        "risk_index",
        "safe_probability",
        "caution_probability",
        "danger_probability",
        "attention_probability",
    ]

    for optional in [
        "heritage_type",
        "address",
        "latitude",
        "longitude",
    ]:
        if optional in result_df.columns:
            columns.append(
                optional
            )

    display = (
        result_df[
            columns
        ]
        .copy()
        .sort_values(
            [
                "risk_index",
                "danger_probability",
            ],
            ascending=False,
        )
        .reset_index(drop=True)
    )

    return display



# ============================================================
# 5-1. 전일~40일 환경 데이터 자동 수집 / 전처리
# ============================================================

def to_float(value):
    """
    API의 '', '-', None 등을 NaN으로 안전하게 변환.
    """
    if value in ("", "-", None):
        return np.nan

    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


# ============================================================
# 4. ASOS 최근 40일 수집
# ============================================================

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_recent_weather(
    target_date: date,
) -> pd.DataFrame:
    """
    예측 기준일 포함 최근 40일 ASOS 일자료 수집.
    """

    if not ASOS_SERVICE_KEY:
        raise ValueError(
            "Streamlit Secrets에 ASOS_SERVICE_KEY "
            "또는 SERVICE_KEY가 없습니다."
        )

    start_date = target_date - timedelta(days=39)

    params = {
        "serviceKey": ASOS_SERVICE_KEY,
        "numOfRows": "100",
        "pageNo": "1",
        "dataType": "JSON",
        "dataCd": "ASOS",
        "dateCd": "DAY",
        "startDt": start_date.strftime("%Y%m%d"),
        "endDt": target_date.strftime("%Y%m%d"),
        "stnIds": STN_ID,
    }

    response = requests.get(
        ASOS_URL,
        params=params,
        timeout=40,
    )

    response.raise_for_status()

    try:
        result = response.json()
    except Exception as e:
        raise RuntimeError(
            "ASOS API 응답을 JSON으로 해석할 수 없습니다."
        ) from e

    items = (
        result.get("response", {})
        .get("body", {})
        .get("items", {})
        .get("item", [])
    )

    if not items:
        raise RuntimeError(
            f"ASOS 데이터가 없습니다: "
            f"{start_date} ~ {target_date}"
        )

    weather = pd.DataFrame(items)

    required_cols = [
        "tm",
        "avgTa",
        "maxTa",
        "minTa",
        "avgRhm",
        "sumRn",
        "avgWs",
        "sumSsHr",
        "avgTs",
    ]

    missing_cols = [
        col
        for col in required_cols
        if col not in weather.columns
    ]

    if missing_cols:
        raise ValueError(
            "ASOS 응답에 필요한 컬럼이 없습니다: "
            f"{missing_cols}"
        )

    weather = weather[
        required_cols
    ].copy()

    # sumSsHr = 일조시간
    weather.columns = [
        "date",
        "temp_avg",
        "temp_max",
        "temp_min",
        "humidity",
        "rainfall",
        "wind_speed",
        "sunshine_hours",
        "ground_temp",
    ]

    weather["date"] = pd.to_datetime(
        weather["date"],
        errors="coerce",
    ).dt.floor("D")

    for col in [
        "temp_avg",
        "temp_max",
        "temp_min",
        "humidity",
        "rainfall",
        "wind_speed",
        "sunshine_hours",
        "ground_temp",
    ]:
        weather[col] = pd.to_numeric(
            weather[col],
            errors="coerce",
        )

    # 강수량 공백은 무강수 0 mm로 처리
    weather["rainfall"] = (
        weather["rainfall"]
        .fillna(0)
    )

    weather = (
        weather
        .dropna(subset=["date"])
        .sort_values("date")
        .drop_duplicates(
            subset=["date"],
            keep="last",
        )
        .reset_index(drop=True)
    )

    return weather


# ============================================================
# 5. AirKorea 최근 40일 수집
#    7일 단위로 나누어 요청 + 재시도
# ============================================================

def _request_air_chunk(
    start_date: date,
    end_date: date,
    station_name: str,
) -> list[dict]:
    """
    AirKorea API를 한 구간에 대해 호출.
    """

    safe_key = urllib.parse.unquote(
        AIR_SERVICE_KEY
    )

    params = {
        "serviceKey": safe_key,
        "returnType": "json",
        "numOfRows": "200",
        "pageNo": "1",
        "inqBginDt": start_date.strftime("%Y%m%d"),
        "inqEndDt": end_date.strftime("%Y%m%d"),
        "msrstnName": station_name,
    }

    last_error = None

    # 총 4회 시도
    for retry_no, wait_seconds in enumerate(
        [0, 3, 6, 10],
        start=1,
    ):
        if wait_seconds:
            time.sleep(wait_seconds)

        try:
            response = requests.get(
                AIR_URL,
                params=params,
                timeout=60,
            )

            response.raise_for_status()

            if not response.text.strip().startswith("{"):
                raise RuntimeError(
                    "AirKorea API가 JSON이 아닌 응답을 반환했습니다."
                )

            data = response.json()

            items = (
                data.get("response", {})
                .get("body", {})
                .get("items", [])
            )

            return items or []

        except Exception as e:
            last_error = e

    raise RuntimeError(
        f"AirKorea 수집 실패 "
        f"{start_date}~{end_date}: {last_error}"
    )


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_recent_air(
    target_date: date,
) -> tuple[pd.DataFrame, str]:
    """
    최근 40일 AirKorea 자료를 7일 단위로 수집.
    기본 측정소에서 결과가 없으면 "영천시"를 한 번 더 시도한다.
    """

    if not AIR_SERVICE_KEY:
        raise ValueError(
            "Streamlit Secrets에 AIR_SERVICE_KEY "
            "또는 SERVICE_KEY가 없습니다."
        )

    start_date = target_date - timedelta(days=39)

    station_candidates = [
        AIR_STATION_NAME,
    ]

    if AIR_STATION_NAME != "영천시":
        station_candidates.append("영천시")

    if AIR_STATION_NAME != "영천":
        station_candidates.append("영천")

    for station_name in station_candidates:
        all_items = []

        chunk_start = start_date

        while chunk_start <= target_date:
            chunk_end = min(
                chunk_start + timedelta(days=6),
                target_date,
            )

            items = _request_air_chunk(
                chunk_start,
                chunk_end,
                station_name,
            )

            all_items.extend(items)

            chunk_start = (
                chunk_end
                + timedelta(days=1)
            )

        if not all_items:
            continue

        air = pd.DataFrame(
            all_items
        )

        rename_map = {
            "msurDt": "date",
            "pm10Value": "pm10",
            "pm25Value": "pm25",
            "o3Value": "o3",
            "no2Value": "no2",
            "coValue": "co",
            "so2Value": "so2",
        }

        air = air.rename(
            columns=rename_map
        )

        required_cols = [
            "date",
            "pm10",
            "pm25",
            "o3",
            "no2",
            "co",
            "so2",
        ]

        for col in required_cols:
            if col not in air.columns:
                air[col] = np.nan

        air = air[
            required_cols
        ].copy()

        air["date"] = pd.to_datetime(
            air["date"],
            errors="coerce",
        ).dt.floor("D")

        for col in [
            "pm10",
            "pm25",
            "o3",
            "no2",
            "co",
            "so2",
        ]:
            air[col] = (
                air[col]
                .replace(
                    ["-", "", "null", "None"],
                    np.nan,
                )
            )

            air[col] = pd.to_numeric(
                air[col],
                errors="coerce",
            )

        air = (
            air
            .dropna(subset=["date"])
            .groupby(
                "date",
                as_index=False,
            )
            .mean(
                numeric_only=True
            )
            .sort_values("date")
            .reset_index(drop=True)
        )

        if not air.empty:
            return air, station_name

    raise RuntimeError(
        "영천 대기환경 자료를 찾지 못했습니다. "
        "AIR_STATION_NAME 설정을 확인하세요."
    )


# ============================================================
# 6. 기상 + 대기환경 전처리 및 파생변수
# ============================================================

def prepare_recent_environment(
    weather: pd.DataFrame,
    air: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    """
    학습 데이터와 동일한 방식으로 최근 환경자료를 정리하고
    create_environment_features()를 적용한다.
    """

    if weather.empty:
        raise ValueError(
            "기상 데이터가 없습니다."
        )

    if air.empty:
        raise ValueError(
            "대기환경 데이터가 없습니다."
        )

    merged = pd.merge(
        weather,
        air,
        on="date",
        how="left",
    )

    merged = (
        merged
        .sort_values("date")
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # 병합 전 품질 정보
    # --------------------------------------------------------

    quality = {
        "weather_days": len(weather),
        "air_days": len(air),
        "merged_days": len(merged),
        "air_missing_before_ffill": int(
            merged[
                [
                    "pm10",
                    "pm25",
                    "o3",
                    "no2",
                    "co",
                    "so2",
                ]
            ]
            .isna()
            .any(axis=1)
            .sum()
        ),
    }

    # --------------------------------------------------------
    # 이상치 처리
    # --------------------------------------------------------

    non_negative_cols = [
        "rainfall",
        "wind_speed",
        "sunshine_hours",
        "humidity",
        "pm10",
        "pm25",
        "o3",
        "no2",
        "co",
        "so2",
    ]

    for col in non_negative_cols:
        if col in merged.columns:
            merged.loc[
                merged[col] < 0,
                col,
            ] = np.nan

    merged.loc[
        (merged["humidity"] < 0)
        | (merged["humidity"] > 100),
        "humidity",
    ] = np.nan

    merged.loc[
        merged["pm10"] > 1000,
        "pm10",
    ] = np.nan

    merged.loc[
        merged["pm25"] > 500,
        "pm25",
    ] = np.nan

    merged["rainfall"] = (
        merged["rainfall"]
        .fillna(0)
    )

    # --------------------------------------------------------
    # 과거값 기반 ffill
    # bfill 사용 안 함: 미래 데이터 누출 방지
    # --------------------------------------------------------

    fill_cols = [
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

    fill_cols = [
        col
        for col in fill_cols
        if col in merged.columns
    ]

    merged[fill_cols] = (
        merged[fill_cols]
        .ffill()
    )

    before_drop = len(
        merged
    )

    # 시작 구간에서 과거값이 없어 채우지 못한 행 제거
    merged = (
        merged
        .dropna(
            subset=fill_cols
        )
        .reset_index(drop=True)
    )

    quality["initial_rows_removed"] = (
        before_drop - len(merged)
    )

    # --------------------------------------------------------
    # 최소 길이
    # --------------------------------------------------------

    if len(merged) < 28:
        raise ValueError(
            "전처리 후 사용 가능한 데이터가 "
            f"{len(merged)}일뿐입니다. "
            "28일 파생변수를 계산하려면 최소 28일이 필요합니다."
        )

    # --------------------------------------------------------
    # 날짜 연속성 확인
    # --------------------------------------------------------

    expected_dates = pd.date_range(
        merged["date"].min(),
        merged["date"].max(),
        freq="D",
    )

    missing_dates = expected_dates.difference(
        merged["date"]
    )

    quality["missing_calendar_days"] = (
        len(missing_dates)
    )

    # 실제 날짜가 누락되면 rolling 일수가 관측행 기준으로 바뀌므로 중단
    if len(missing_dates) > 0:
        raise ValueError(
            "최근 환경 데이터에 날짜 누락이 있습니다. "
            f"{len(missing_dates)}일 누락: "
            f"{list(missing_dates[:10])}"
        )

    # --------------------------------------------------------
    # 학습과 동일한 공통 파생변수
    # --------------------------------------------------------

    realtime_df = create_environment_features(
        merged.copy(),
        fill_remaining_numeric=False,
    )

    quality["usable_days"] = len(
        realtime_df
    )

    return realtime_df, quality


# ============================================================
# 6. 마지막 예측 결과 저장 / GitHub 자동 업로드
# ============================================================

def get_github_repo():
    """
    Streamlit Secrets에서 GitHub 인증정보를 읽고
    Repository 객체와 기본 브랜치명을 반환한다.

    필수 Secrets
    ------------
    GITHUB_TOKEN = "..."
    GITHUB_REPO = "jiwonshin84/yc_heritage_project"
    """

    try:
        token = st.secrets["GITHUB_TOKEN"]
        repo_name = st.secrets["GITHUB_REPO"]

    except KeyError as e:
        raise RuntimeError(
            f"Streamlit Secrets에 GitHub 설정이 없습니다: {e}"
        ) from e

    if not str(token).strip():
        raise RuntimeError(
            "GITHUB_TOKEN 값이 비어 있습니다."
        )

    if not str(repo_name).strip():
        raise RuntimeError(
            "GITHUB_REPO 값이 비어 있습니다."
        )

    try:
        github = Github(
            str(token).strip(),
            timeout=30,
        )

        repo = github.get_repo(
            str(repo_name).strip()
        )

        branch = repo.default_branch

        return repo, branch

    except Exception as e:
        raise RuntimeError(
            f"GitHub 저장소 연결 실패: {e}"
        ) from e


def upload_local_file_to_github(
    local_path: Path | str,
    git_file_path: str,
    commit_message: str,
) -> dict:
    """
    로컬 파일을 GitHub 저장소에 생성 또는 업데이트한다.
    """

    local_path = Path(
        local_path
    )

    if not local_path.exists():
        raise FileNotFoundError(
            f"업로드할 로컬 파일이 없습니다: {local_path}"
        )

    repo, branch = get_github_repo()

    file_bytes = (
        local_path
        .read_bytes()
    )

    git_file_path = (
        str(git_file_path)
        .replace("\\", "/")
        .lstrip("/")
    )

    try:
        existing = repo.get_contents(
            git_file_path,
            ref=branch,
        )

        repo.update_file(
            path=git_file_path,
            message=commit_message,
            content=file_bytes,
            sha=existing.sha,
            branch=branch,
        )

        return {
            "path": git_file_path,
            "status": "업데이트",
            "branch": branch,
        }

    except Exception as get_error:
        status_code = getattr(
            get_error,
            "status",
            None,
        )

        if status_code == 404:
            repo.create_file(
                path=git_file_path,
                message=commit_message,
                content=file_bytes,
                branch=branch,
            )

            return {
                "path": git_file_path,
                "status": "신규 생성",
                "branch": branch,
            }

        raise RuntimeError(
            f"GitHub 기존 파일 확인 실패 "
            f"[{git_file_path}]: {get_error}"
        ) from get_error


def save_latest_prediction(
    result_df: pd.DataFrame,
    prediction_date,
) -> tuple[pd.DataFrame, dict]:
    """
    마지막 예측 결과를
    data/processed/latest_prediction.csv 로 저장하고
    GitHub에도 자동 업로드한다.

    실시간 대시보드와의 호환을 위해:
    - prediction_date
    - risk_label
    컬럼을 함께 저장한다.
    """

    save_df = result_df.copy()

    prediction_ts = pd.Timestamp(
        prediction_date
    ).floor("D")

    # 대시보드가 최근 예측일과 등급을 안정적으로 읽을 수 있도록
    # 명확한 공통 컬럼을 추가
    save_df.insert(
        0,
        "prediction_date",
        prediction_ts.strftime("%Y-%m-%d"),
    )

    save_df["risk_label"] = (
        save_df["predicted_grade"]
    )

    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_df.to_csv(
        LATEST_PREDICTION_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    github_result = (
        upload_local_file_to_github(
            local_path=LATEST_PREDICTION_PATH,
            git_file_path=(
                "data/processed/latest_prediction.csv"
            ),
            commit_message=(
                "chore: latest heritage prediction "
                f"{prediction_ts:%Y-%m-%d}"
            ),
        )
    )

    return save_df, github_result


# ============================================================
# 8. 모델 / 문화유산 데이터 준비
# ============================================================

try:
    (
        model,
        feature_cols,
        train_medians,
        metadata,
    ) = load_model_assets()
except Exception as e:
    st.error(
        f"❌ 모델 로드 실패: {e}"
    )
    st.stop()


try:
    heritage_path = find_heritage_path()
    heritage_df = load_heritage_data(
        str(heritage_path)
    )
except Exception as e:
    st.error(
        f"❌ 문화유산 데이터 로드 실패: {e}"
    )
    st.stop()


# ============================================================
# 9. 전일~40일 자동 수집 + 바로 취약도 예측
# ============================================================

target_date = DEFAULT_TARGET_DATE
target_ts = pd.Timestamp(
    target_date
)

start_target_date = (
    target_date
    - timedelta(days=39)
)

model_name = metadata.get(
    "model_name",
    type(model).__name__,
)


# ------------------------------------------------------------
# 세션 상태
# ------------------------------------------------------------

session_defaults = {
    "heritage_prediction_result": None,
    "heritage_prediction_date": None,
    "heritage_prediction_github": None,
    "heritage_prediction_github_error": None,
    "recent_40_weather": None,
    "recent_40_air": None,
    "recent_40_environment": None,
    "recent_40_quality": None,
    "recent_40_target_date": None,
    "recent_40_air_station": None,
}

for key, default_value in session_defaults.items():
    if key not in st.session_state:
        st.session_state[key] = default_value


# rerun 후에도 상세 화면에서 사용할 수 있도록 세션에서 복원
realtime_df = st.session_state.get(
    "recent_40_environment"
)


st.markdown(
    f"""
    <div class="prediction-hero">
        <h3>🤖 전일 기준 자동 수집·예측</h3>
        <p>
            <b>{start_target_date:%Y-%m-%d}</b> ~
            <b>{target_ts:%Y-%m-%d}</b> 최근 40일 환경자료를 자동 수집하고,
            7일·28일 파생변수 생성 → 문화유산 재질·노출환경 결합 →
            환경 취약도 예측까지 한 번에 실행합니다.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


top1, top2, top3, top4 = st.columns(
    4
)

top1.metric(
    "📅 예측 기준일",
    f"{target_ts:%Y-%m-%d}",
)

top2.metric(
    "🏛️ 분석 문화유산",
    f"{len(heritage_df):,}개",
)

top3.metric(
    "🤖 적용 모델",
    model_name,
)

top4.metric(
    "🧩 모델 Feature",
    f"{len(feature_cols):,}개",
)

st.caption(
    f"결측 보정용 학습 중앙값: {len(train_medians):,}개 Feature · "
    "2019~2024 최종 학습 구간 기준"
)

st.caption(
    "※ 별도의 '최근 40일 환경 데이터' 페이지를 먼저 실행할 필요가 없습니다. "
    "아래 버튼 한 번으로 전일 기준 최근 40일 데이터 수집부터 예측까지 처리합니다."
)


run_clicked = st.button(
    "🚀 전일~40일 자동 수집 후 문화유산 환경 취약도 예측",
    type="primary",
    use_container_width=True,
)


if run_clicked:
    try:
        status = st.status(
            "환경 데이터 수집 및 문화유산 취약도 예측을 준비하고 있습니다...",
            expanded=True,
        )

        # 1. ASOS
        status.update(
            label="🌦 기상청 ASOS 최근 40일 수집 중...",
            state="running",
        )

        weather = fetch_recent_weather(
            target_date
        )

        # 2. AirKorea
        status.update(
            label="🌫 AirKorea 최근 40일 수집 중...",
            state="running",
        )

        air, used_station = fetch_recent_air(
            target_date
        )

        # 3. merge + features
        status.update(
            label="🧮 기상·대기환경 병합 및 7일·28일 파생변수 생성 중...",
            state="running",
        )

        realtime_df, quality = (
            prepare_recent_environment(
                weather,
                air,
            )
        )

        # 기준일 정확히 존재하는지 확인
        target_rows = realtime_df.loc[
            realtime_df["date"].dt.floor("D")
            == pd.Timestamp(
                target_date
            ).floor("D")
        ]

        if target_rows.empty:
            raise ValueError(
                f"{target_date:%Y-%m-%d} 기준일의 "
                "최종 환경 Feature가 없습니다. "
                "이전 날짜를 임의로 대신 사용하지 않습니다."
            )

        # 수집 결과 세션 저장
        st.session_state.recent_40_weather = weather
        st.session_state.recent_40_air = air
        st.session_state.recent_40_environment = realtime_df
        st.session_state.recent_40_quality = quality
        st.session_state.recent_40_target_date = target_date
        st.session_state.recent_40_air_station = used_station

        # 최근 40일 환경·파생변수를 파일로도 저장하여
        # 페이지 이동/재배포 이후 시각화 페이지에서 fallback으로 사용
        DATA_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        realtime_df.to_csv(
            LATEST_ENVIRONMENT_PATH,
            index=False,
            encoding="utf-8-sig",
        )

        # 4. exact target feature
        status.update(
            label="📌 전일 기준 환경 Feature 선택 중...",
            state="running",
        )

        target_environment = (
            select_target_environment(
                realtime_df,
                target_date,
            )
        )

        # 5. combine
        status.update(
            label="🏛️ 환경 데이터와 문화유산 특성 결합 중...",
            state="running",
        )

        prediction_input = (
            combine_environment_and_heritage(
                target_environment,
                heritage_df,
            )
        )

        # 6. prediction
        status.update(
            label="🤖 학습 모델로 문화유산별 환경 취약도 예측 중...",
            state="running",
        )

        result_df = run_prediction(
            model,
            feature_cols,
            train_medians,
            prediction_input,
        )

        result_df = make_display_result(
            result_df
        )

        # 예측 결과 세션 저장
        st.session_state.heritage_prediction_result = (
            result_df
        )

        st.session_state.heritage_prediction_date = (
            target_date
        )

        # 실시간 대시보드 호환 세션 키
        st.session_state["prediction_result"] = (
            result_df.copy()
        )
        st.session_state["prediction_df"] = (
            result_df.copy()
        )
        st.session_state["latest_prediction"] = (
            result_df.copy()
        )
        st.session_state["danger_count"] = int(
            (
                result_df["predicted_grade"]
                == "위험"
            ).sum()
        )

        # 7. latest_prediction.csv + GitHub
        status.update(
            label="☁️ 마지막 예측 결과 저장 중...",
            state="running",
        )

        st.session_state[
            "heritage_prediction_github"
        ] = None

        st.session_state[
            "heritage_prediction_github_error"
        ] = None

        try:
            # 최근 40일 환경·파생변수도 GitHub에 저장하여
            # Streamlit 재배포 후에도 09 페이지가 읽을 수 있게 한다.
            upload_local_file_to_github(
                local_path=LATEST_ENVIRONMENT_PATH,
                git_file_path=(
                    "data/processed/latest_40_environment.csv"
                ),
                commit_message=(
                    "chore: latest 40-day environment "
                    f"{pd.Timestamp(target_date):%Y-%m-%d}"
                ),
            )

            (
                latest_saved_df,
                github_result,
            ) = save_latest_prediction(
                result_df=result_df,
                prediction_date=target_date,
            )

            st.session_state[
                "heritage_prediction_github"
            ] = github_result

            st.session_state[
                "latest_prediction_df"
            ] = latest_saved_df

        except Exception as save_error:
            # GitHub 업로드 오류가 발생해도 예측 자체는 유지
            st.session_state[
                "heritage_prediction_github_error"
            ] = str(save_error)

            # 로컬 latest_prediction.csv는 별도로 보장
            try:
                local_save_df = result_df.copy()

                local_save_df.insert(
                    0,
                    "prediction_date",
                    pd.Timestamp(
                        target_date
                    ).strftime("%Y-%m-%d"),
                )

                local_save_df["risk_label"] = (
                    local_save_df[
                        "predicted_grade"
                    ]
                )

                DATA_DIR.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                local_save_df.to_csv(
                    LATEST_PREDICTION_PATH,
                    index=False,
                    encoding="utf-8-sig",
                )

                st.session_state[
                    "latest_prediction_df"
                ] = local_save_df

            except Exception:
                pass

        status.update(
            label=(
                "✅ 최근 40일 자동 수집 · "
                "파생변수 생성 · 환경 취약도 예측 완료"
            ),
            state="complete",
            expanded=False,
        )

        st.rerun()

    except Exception as e:
        st.error(
            f"❌ 자동 수집 및 예측 실행 실패: {e}"
        )


# ============================================================
# 10. 예측 결과
# ============================================================

result_df = (
    st.session_state
    .heritage_prediction_result
)

prediction_date = (
    st.session_state
    .heritage_prediction_date
)


if result_df is None:

    st.markdown("---")

    st.markdown(
        """
        <div class="prediction-hero">
            <h3>위의 자동 수집·예측 버튼을 눌러주세요.</h3>
            <p>
                별도의 최근 40일 데이터 페이지를 먼저 실행할 필요 없이,
                전일 기준 최근 40일 자료 수집부터 파생변수 생성과
                문화유산별 안전·주의·위험 예측까지 한 번에 처리합니다.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.stop()


# ============================================================
# 11. 결과 요약
# ============================================================

st.markdown("---")

st.subheader(
    f"📊 {pd.Timestamp(prediction_date):%Y-%m-%d} "
    "문화유산 환경 취약도 예측 결과"
)


github_save_result = st.session_state.get(
    "heritage_prediction_github"
)

github_save_error = st.session_state.get(
    "heritage_prediction_github_error"
)

if github_save_result:
    st.success(
        "☁️ 마지막 예측 결과 저장 완료 · "
        f"{github_save_result['path']} · "
        f"{github_save_result['status']} · "
        f"{github_save_result['branch']} 브랜치"
    )

elif github_save_error:
    st.warning(
        "⚠️ 환경 취약도 예측은 정상 완료되었지만 "
        "GitHub 자동 저장에는 실패했습니다. "
        "현재 세션과 로컬 latest_prediction.csv 결과는 유지됩니다.\n\n"
        f"GitHub 저장 오류: {github_save_error}"
    )

else:
    st.info(
        "ℹ️ 예측을 실행하면 최근 40일 환경 데이터를 자동 수집한 뒤 "
        "결과를 data/processed/latest_prediction.csv 로 저장하고 "
        "GitHub에도 자동 업로드합니다."
    )

total_count = len(
    result_df
)

safe_count = int(
    (
        result_df[
            "predicted_grade"
        ]
        == "안전"
    )
    .sum()
)

caution_count = int(
    (
        result_df[
            "predicted_grade"
        ]
        == "주의"
    )
    .sum()
)

danger_count = int(
    (
        result_df[
            "predicted_grade"
        ]
        == "위험"
    )
    .sum()
)

attention_count = (
    caution_count
    + danger_count
)


k1, k2, k3, k4 = st.columns(
    4
)

k1.metric(
    "🏛️ 전체 분석",
    f"{total_count:,}개",
)

k2.metric(
    "✅ 안전",
    f"{safe_count:,}개",
    (
        f"{safe_count / total_count * 100:.1f}%"
        if total_count
        else "0%"
    ),
)

k3.metric(
    "⚠️ 주의",
    f"{caution_count:,}개",
    (
        f"{caution_count / total_count * 100:.1f}%"
        if total_count
        else "0%"
    ),
)

k4.metric(
    "🚨 위험",
    f"{danger_count:,}개",
    (
        f"{danger_count / total_count * 100:.1f}%"
        if total_count
        else "0%"
    ),
)


if danger_count > 0:

    st.markdown(
        f"""
        <div class="risk-danger">
            <b>🚨 위험 등급 {danger_count}개</b><br>
            현재 모델에서 위험 등급으로 분류된 문화유산을
            우선 점검 대상으로 확인할 수 있습니다.
        </div>
        """,
        unsafe_allow_html=True,
    )

elif caution_count > 0:

    st.markdown(
        f"""
        <div class="risk-caution">
            <b>⚠️ 주의 등급 {caution_count}개</b><br>
            위험 등급은 없지만 주의 등급 문화유산이 있습니다.
            상위 순위 문화유산의 환경조건을 함께 확인하세요.
        </div>
        """,
        unsafe_allow_html=True,
    )

else:

    st.markdown(
        """
        <div class="risk-safe">
            <b>✅ 현재 모델에서 모두 안전 등급으로 분류되었습니다.</b><br>
            이는 실제 훼손 가능성이 없다는 의미가 아니라,
            현재 학습된 분류모델의 환경 취약도 기준에 따른 결과입니다.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# 12. 메인 시각화
# ============================================================

st.markdown("---")
st.subheader("🎨 한눈에 보는 예측 결과")

left_chart, right_chart = st.columns(
    [0.9, 1.3]
)

# ------------------------------------------------------------
# 11-1. 등급 분포 Donut
# ------------------------------------------------------------

with left_chart:

    grade_counts = (
        result_df[
            "predicted_grade"
        ]
        .value_counts()
        .reindex(
            GRADE_ORDER,
            fill_value=0,
        )
        .rename_axis("등급")
        .reset_index(
            name="문화유산 수"
        )
    )

    fig_grade = px.pie(
        grade_counts,
        names="등급",
        values="문화유산 수",
        color="등급",
        color_discrete_map=GRADE_COLOR,
        hole=0.62,
        title="안전·주의·위험 등급 분포",
    )

    fig_grade.update_traces(
        textposition="inside",
        textinfo="label+percent",
        hovertemplate=(
            "<b>%{label}</b><br>"
            "문화유산 %{value}개<br>"
            "%{percent}<extra></extra>"
        ),
    )

    fig_grade.add_annotation(
        text=f"<b>{total_count}</b><br>문화유산",
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(
            size=20
        ),
    )

    fig_grade.update_layout(
        height=470,
        margin=dict(
            t=70,
            b=20,
            l=20,
            r=20,
        ),
        legend=dict(
            orientation="h",
            y=-0.08,
        ),
    )

    st.plotly_chart(
        fig_grade,
        use_container_width=True,
    )


# ------------------------------------------------------------
# 11-2. 관심도 TOP 15
# ------------------------------------------------------------

with right_chart:

    top_n = min(
        15,
        len(result_df),
    )

    top_attention = (
        result_df
        .nlargest(
            top_n,
            "risk_index",
        )
        .sort_values(
            "risk_index",
            ascending=True,
        )
        .copy()
    )

    fig_top = px.bar(
        top_attention,
        x="risk_index",
        y="heritage_name",
        orientation="h",
        color="predicted_grade",
        color_discrete_map=GRADE_COLOR,
        text="risk_index",
        title=f"환경 취약도 관심순위 TOP {top_n}",
        labels={
            "risk_index": "환경 취약도 지수",
            "heritage_name": "",
            "predicted_grade": "예측 등급",
        },
    )

    fig_top.update_traces(
        texttemplate="%{text:.1f}",
        textposition="outside",
        cliponaxis=False,
        hovertemplate=(
            "<b>%{y}</b><br>"
            "환경 취약도 지수 %{x:.1f}<extra></extra>"
        ),
    )

    fig_top.update_layout(
        height=470,
        xaxis_range=[0, 105],
        margin=dict(
            t=70,
            b=30,
            l=20,
            r=40,
        ),
    )

    st.plotly_chart(
        fig_top,
        use_container_width=True,
    )


# ============================================================
# 13. 재질별 / 노출환경별 분석
# ============================================================

st.markdown("---")
st.subheader("🏛️ 재질·노출환경별 취약도 비교")

material_col, exposure_col = st.columns(
    2
)

# ------------------------------------------------------------
# 12-1. 재질별 등급
# ------------------------------------------------------------

with material_col:

    material_summary = (
        result_df
        .groupby(
            [
                "material",
                "predicted_grade",
            ]
        )
        .size()
        .reset_index(
            name="문화유산 수"
        )
    )

    fig_material = px.bar(
        material_summary,
        x="material",
        y="문화유산 수",
        color="predicted_grade",
        color_discrete_map=GRADE_COLOR,
        category_orders={
            "material": MATERIAL_ORDER,
            "predicted_grade": GRADE_ORDER,
        },
        barmode="stack",
        title="재질별 안전·주의·위험 분포",
        labels={
            "material": "재질",
            "predicted_grade": "예측 등급",
        },
    )

    fig_material.update_layout(
        height=420,
        legend_title_text="예측 등급",
    )

    st.plotly_chart(
        fig_material,
        use_container_width=True,
    )


# ------------------------------------------------------------
# 12-2. 노출환경별 평균 취약도
# ------------------------------------------------------------

with exposure_col:

    exposure_summary = (
        result_df
        .groupby(
            "exposure",
            as_index=False,
        )
        .agg(
            평균_취약도=(
                "risk_index",
                "mean",
            ),
            문화유산_수=(
                "heritage_name",
                "count",
            ),
        )
    )

    fig_exposure = px.bar(
        exposure_summary,
        x="exposure",
        y="평균_취약도",
        text="평균_취약도",
        category_orders={
            "exposure": EXPOSURE_ORDER,
        },
        title="노출환경별 평균 환경 취약도",
        labels={
            "exposure": "노출환경",
            "평균_취약도": "평균 환경 취약도 지수",
        },
    )

    fig_exposure.update_traces(
        texttemplate="%{text:.1f}",
        textposition="outside",
    )

    fig_exposure.update_layout(
        height=420,
        yaxis_range=[0, 100],
    )

    st.plotly_chart(
        fig_exposure,
        use_container_width=True,
    )


# ============================================================
# 14. 지도 시각화
# ============================================================

if (
    "latitude" in result_df.columns
    and "longitude" in result_df.columns
):

    map_df = result_df.dropna(
        subset=[
            "latitude",
            "longitude",
        ]
    ).copy()

    if not map_df.empty:

        st.markdown("---")
        st.subheader("🗺️ 문화유산 환경 취약도 공간 분포")

        map_df[
            "표시크기"
        ] = (
            map_df[
                "risk_index"
            ]
            .clip(
                lower=5,
                upper=100,
            )
        )

        fig_map = px.scatter_map(
            map_df,
            lat="latitude",
            lon="longitude",
            color="predicted_grade",
            size="표시크기",
            color_discrete_map=GRADE_COLOR,
            hover_name="heritage_name",
            hover_data={
                "material": True,
                "exposure": True,
                "risk_index": ":.1f",
                "danger_probability": ":.1f",
                "latitude": False,
                "longitude": False,
                "표시크기": False,
            },
            zoom=9,
            height=600,
            labels={
                "material": "재질",
                "exposure": "노출환경",
                "risk_index": "환경 취약도 지수",
                "danger_probability": "위험 확률(%)",
                "predicted_grade": "예측 등급",
            },
        )

        fig_map.update_layout(
            map_style="open-street-map",
            margin=dict(
                t=10,
                b=10,
                l=10,
                r=10,
            ),
            legend=dict(
                orientation="h",
                y=1.02,
            ),
        )

        st.plotly_chart(
            fig_map,
            use_container_width=True,
        )


# ============================================================
# 15. 문화유산별 상세 결과
# ============================================================

st.markdown("---")
st.subheader("🔎 문화유산별 예측 결과 상세")

filter1, filter2, filter3 = st.columns(
    [1, 1, 1.4]
)

with filter1:
    selected_grade = st.multiselect(
        "예측 등급",
        options=GRADE_ORDER,
        default=GRADE_ORDER,
    )

with filter2:
    selected_materials = st.multiselect(
        "재질",
        options=[
            x
            for x in MATERIAL_ORDER
            if x in result_df[
                "material"
            ].unique()
        ],
        default=[
            x
            for x in MATERIAL_ORDER
            if x in result_df[
                "material"
            ].unique()
        ],
    )

with filter3:
    keyword = st.text_input(
        "문화유산명 검색",
        placeholder="문화유산 이름 일부를 입력하세요.",
    )


filtered_result = result_df[
    result_df[
        "predicted_grade"
    ].isin(
        selected_grade
    )
    &
    result_df[
        "material"
    ].isin(
        selected_materials
    )
].copy()


if keyword.strip():
    filtered_result = filtered_result[
        filtered_result[
            "heritage_name"
        ]
        .str.contains(
            keyword.strip(),
            case=False,
            na=False,
        )
    ]


table_cols = [
    "heritage_name",
    "material",
    "exposure",
    "predicted_grade",
    "risk_index",
    "attention_probability",
    "safe_probability",
    "caution_probability",
    "danger_probability",
]

for optional in [
    "heritage_type",
    "address",
]:
    if optional in filtered_result.columns:
        table_cols.append(
            optional
        )


table_df = filtered_result[
    table_cols
].copy()


rename_map = {
    "heritage_name": "문화유산명",
    "material": "재질",
    "exposure": "노출환경",
    "predicted_grade": "예측 등급",
    "risk_index": "환경 취약도 지수",
    "attention_probability": "주의 이상 확률(%)",
    "safe_probability": "안전 확률(%)",
    "caution_probability": "주의 확률(%)",
    "danger_probability": "위험 확률(%)",
    "heritage_type": "국가유산 종목",
    "address": "소재지",
}

table_df = table_df.rename(
    columns=rename_map
)


st.dataframe(
    table_df,
    use_container_width=True,
    height=560,
    hide_index=True,
    column_config={
        "환경 취약도 지수": st.column_config.ProgressColumn(
            "환경 취약도 지수",
            min_value=0,
            max_value=100,
            format="%.1f",
        ),
        "주의 이상 확률(%)": st.column_config.ProgressColumn(
            "주의 이상 확률(%)",
            min_value=0,
            max_value=100,
            format="%.1f%%",
        ),
        "위험 확률(%)": st.column_config.ProgressColumn(
            "위험 확률(%)",
            min_value=0,
            max_value=100,
            format="%.1f%%",
        ),
    },
)


# ============================================================
# 16. 우선 확인 문화유산
# ============================================================

st.markdown("---")
st.subheader("🚨 우선 확인 문화유산")

attention_view = (
    result_df[
        result_df[
            "predicted_grade"
        ].isin(
            [
                "주의",
                "위험",
            ]
        )
    ]
    .sort_values(
        [
            "predicted_grade",
            "risk_index",
        ],
        ascending=[
            False,
            False,
        ],
    )
    .head(20)
    .copy()
)


if attention_view.empty:

    attention_view = (
        result_df
        .nlargest(
            min(
                10,
                len(result_df),
            ),
            "risk_index",
        )
        .copy()
    )

    st.info(
        "현재 주의·위험 등급 문화유산이 없어 "
        "환경 취약도 지수가 높은 문화유산을 대신 표시합니다."
    )


for rank, (_, row) in enumerate(
    attention_view.iterrows(),
    start=1,
):

    grade = row[
        "predicted_grade"
    ]

    icon = {
        "안전": "✅",
        "주의": "⚠️",
        "위험": "🚨",
    }.get(
        grade,
        "•",
    )

    with st.expander(
        (
            f"{rank}. {icon} "
            f"{row['heritage_name']} "
            f"— {grade} "
            f"(취약도 {row['risk_index']:.1f})"
        ),
        expanded=(
            rank <= 3
        ),
    ):

        a1, a2, a3, a4 = st.columns(
            4
        )

        a1.metric(
            "재질",
            row[
                "material"
            ],
        )

        a2.metric(
            "노출환경",
            row[
                "exposure"
            ],
        )

        a3.metric(
            "주의 이상 확률",
            f"{row['attention_probability']:.1f}%",
        )

        a4.metric(
            "위험 확률",
            f"{row['danger_probability']:.1f}%",
        )

        if (
            "address" in row.index
            and pd.notna(
                row[
                    "address"
                ]
            )
        ):
            st.caption(
                f"📍 소재지: {row['address']}"
            )


# ============================================================
# 17. 예측 기준일 환경 Feature 확인
# ============================================================

st.markdown("---")

with st.expander(
    "🧪 이번 예측에 사용된 기준일 환경 Feature",
    expanded=False,
):

    realtime_for_view = st.session_state.get(
        "recent_40_environment"
    )

    if realtime_for_view is None:
        st.info(
            "현재 세션에 최근 40일 환경 Feature가 없습니다. "
            "상단의 자동 수집·예측 버튼을 다시 실행해주세요."
        )

    else:
        target_environment = (
            select_target_environment(
                realtime_for_view,
                prediction_date,
            )
        )

        selected_feature_view = (
            target_environment
            .T
            .reset_index()
        )

        selected_feature_view.columns = [
            "Feature",
            "값",
        ]

        st.dataframe(
            selected_feature_view,
            use_container_width=True,
            hide_index=True,
            height=520,
        )


# ============================================================
# 18. CSV 다운로드
# ============================================================

st.markdown("---")

download_df = (
    result_df
    .rename(
        columns=rename_map
    )
    .copy()
)

csv_bytes = (
    download_df
    .to_csv(
        index=False
    )
    .encode(
        "utf-8-sig"
    )
)

st.download_button(
    "📥 문화유산별 예측 결과 CSV 다운로드",
    data=csv_bytes,
    file_name=(
        f"영천_문화유산_환경취약도_예측_"
        f"{pd.Timestamp(prediction_date):%Y%m%d}.csv"
    ),
    mime="text/csv",
    type="primary",
    use_container_width=True,
)


st.caption(
    "☁️ 예측 실행 시 동일 결과가 "
    "`data/processed/latest_prediction.csv` 파일로 GitHub에 자동 저장되어 "
    "실시간 대시보드의 안전·주의·위험 현황에 사용됩니다."
)


# ============================================================
# 19. 해석 안내
# ============================================================

st.caption(
    "※ 환경 취약도 지수는 모델의 안전·주의·위험 예측 확률을 "
    "시각화하기 위해 0~100 범위로 환산한 보조지표입니다. "
    "실제 문화재의 물리적 훼손 정도를 나타내는 측정값이 아닙니다."
)
