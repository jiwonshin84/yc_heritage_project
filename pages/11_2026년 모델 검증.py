from __future__ import annotations

from pathlib import Path
from datetime import datetime, date, timedelta
from zoneinfo import ZoneInfo
import time
import urllib.parse

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import requests
import streamlit as st

from utils.feature_engineering import create_environment_features


# ============================================================
# 1. 페이지 설정
# ============================================================

st.set_page_config(
    page_title="영천 2026년 미학습 기간 환경 취약도 예측",
    page_icon="🧪",
    layout="wide",
)

st.title("🧪 영천 지역 2026년 미학습 기간 환경 취약도 예측")
st.caption(
    "2025년까지의 자료로 구축한 모델을 학습에 사용하지 않은 2026년 환경자료에 적용하여 "
    "날짜별 환경 취약도를 예측합니다. 실제 예측 기간은 ASOS와 AirKorea에서 실제 확보된 "
    "공공데이터의 공통 가용기간을 기준으로 자동 결정됩니다."
)

st.info(
    "📌 날짜별 대표 등급은 해당 날짜에 영천 문화유산 전체를 예측한 뒤 "
    "가장 높은 위험 등급을 대표값으로 표시합니다. "
    "따라서 실제 훼손 발생 여부가 아니라 프로젝트 모델의 상대적 환경 취약도입니다."
)


# ============================================================
# 2. 경로 / 상수
# ============================================================

MODEL_DIR = Path("models")
DATA_DIR = Path("data/processed")

MODEL_PATH = MODEL_DIR / "best_model.pkl"
FEATURE_COLS_PATH = MODEL_DIR / "feature_cols.pkl"
TRAIN_MEDIANS_PATH = MODEL_DIR / "train_medians.pkl"
BUNDLE_PATH = MODEL_DIR / "heritage_risk_bundle.pkl"
MODEL_META_PATH = MODEL_DIR / "model_metadata.json"

HERITAGE_CANDIDATES = [
    DATA_DIR / "yc_heritage_feature.csv",
    DATA_DIR / "yc_heritage_detail_enriched.csv",
    DATA_DIR / "yc_heritage_features.csv",
]

MATERIAL_ORDER = ["석조", "목조", "금속", "회화", "기타"]
EXPOSURE_ORDER = ["실외", "반실외", "실내"]
GRADE_ORDER = ["안전", "주의", "위험"]
GRADE_RANK = {"안전": 0, "주의": 1, "위험": 2}
GRADE_COLOR = {"안전": "#2ECC71", "주의": "#F39C12", "위험": "#E74C3C"}

ASOS_URL = (
    "https://apis.data.go.kr/"
    "1360000/AsosDalyInfoService/getWthrDataList"
)
AIR_URL = (
    "https://apis.data.go.kr/"
    "B552584/ArpltnStatsSvc/getMsrstnAcctoRDyrg"
)

STN_ID = "281"  # 영천 ASOS
KST = ZoneInfo("Asia/Seoul")

ASOS_SERVICE_KEY = st.secrets.get(
    "ASOS_SERVICE_KEY",
    st.secrets.get("SERVICE_KEY", ""),
)
AIR_SERVICE_KEY = st.secrets.get(
    "AIR_SERVICE_KEY",
    st.secrets.get("SERVICE_KEY", ""),
)
AIR_STATION_NAME = st.secrets.get("AIR_STATION_NAME", "영천")

DISPLAY_START_DATE = date(2026, 1, 1)
# 28일 rolling 파생변수를 1월 1일부터 계산하려면 이전 데이터가 필요함.
COLLECT_START_DATE = DISPLAY_START_DATE - timedelta(days=35)
TODAY = datetime.now(KST).date()
REQUEST_END_DATE = TODAY - timedelta(days=1)


def find_heritage_path() -> Path:
    for path in HERITAGE_CANDIDATES:
        if path.exists():
            return path
    raise FileNotFoundError(
        "문화유산 특성 데이터 파일을 찾을 수 없습니다.\n"
        + "\n".join(f"- {p}" for p in HERITAGE_CANDIDATES)
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


# ============================================================
# 3. 기간형 ASOS 수집
# ============================================================

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_weather_range(
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    """
    ASOS 장기간 수집용.
    전체 기간을 한 번에 요청하지 않고 30일 단위로 나누어 수집한다.
    한 구간이 실패하면 3회 재시도하고, 그래도 실패하면 그 구간을
    더 작은 기간으로 자동 분할한다.
    """

    if not ASOS_SERVICE_KEY:
        raise ValueError(
            "Streamlit Secrets에 ASOS_SERVICE_KEY 또는 SERVICE_KEY가 없습니다."
        )

    def request_once(req_start: date, req_end: date):
        params = {
            "serviceKey": ASOS_SERVICE_KEY,
            "numOfRows": "999",
            "pageNo": "1",
            "dataType": "JSON",
            "dataCd": "ASOS",
            "dateCd": "DAY",
            "startDt": req_start.strftime("%Y%m%d"),
            "endDt": req_end.strftime("%Y%m%d"),
            "stnIds": STN_ID,
        }

        for wait_seconds in [0, 2, 5]:
            if wait_seconds:
                time.sleep(wait_seconds)

            try:
                response = requests.get(
                    ASOS_URL,
                    params=params,
                    timeout=60,
                )
                response.raise_for_status()

                if not response.text.strip().startswith("{"):
                    continue

                result = response.json()

                header = (
                    result.get("response", {})
                    .get("header", {})
                )

                result_code = str(header.get("resultCode", ""))
                result_msg = header.get("resultMsg", "")

                if result_code not in ["00", "0", ""]:
                    # 인증 오류 등은 분할해도 해결되지 않으므로
                    # 마지막 재시도까지 진행 후 None 처리
                    continue

                items_obj = (
                    result.get("response", {})
                    .get("body", {})
                    .get("items", {})
                )

                if isinstance(items_obj, dict):
                    items = items_obj.get("item", [])
                else:
                    items = []

                if isinstance(items, dict):
                    items = [items]

                # 정상 응답이지만 해당 기간에 자료가 없는 경우도 []
                return items or []

            except Exception:
                continue

        return None

    def collect_period(req_start: date, req_end: date):
        """
        요청 자체가 계속 실패하면 기간을 절반으로 나눈다.
        하루까지 실패하면 해당 날짜를 failed_dates에 남긴다.
        """
        items = request_once(req_start, req_end)

        if items is not None:
            return items, []

        days = (req_end - req_start).days + 1

        if days <= 1:
            return [], [req_start]

        left_days = days // 2
        midpoint = req_start + timedelta(days=left_days - 1)
        right_start = midpoint + timedelta(days=1)

        left_items, left_failed = collect_period(
            req_start,
            midpoint,
        )
        right_items, right_failed = collect_period(
            right_start,
            req_end,
        )

        return (
            left_items + right_items,
            left_failed + right_failed,
        )

    all_items = []
    failed_dates = []

    # 30일씩 나누어 요청
    chunk_start = start_date

    while chunk_start <= end_date:
        chunk_end = min(
            chunk_start + timedelta(days=29),
            end_date,
        )

        items, failed = collect_period(
            chunk_start,
            chunk_end,
        )

        all_items.extend(items)
        failed_dates.extend(failed)

        chunk_start = chunk_end + timedelta(days=1)
        time.sleep(0.15)

    if not all_items:
        raise RuntimeError(
            f"ASOS 데이터를 전혀 수집하지 못했습니다: "
            f"{start_date} ~ {end_date}. "
            "ASOS_SERVICE_KEY와 공공데이터포털 API 활용신청 상태를 확인하세요."
        )

    weather = pd.DataFrame(all_items)

    required = [
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

    missing = [
        col for col in required
        if col not in weather.columns
    ]

    if missing:
        raise ValueError(
            f"ASOS 응답에 필요한 컬럼이 없습니다: {missing}"
        )

    weather = weather[required].copy()

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

    numeric_cols = [
        col for col in weather.columns
        if col != "date"
    ]

    for col in numeric_cols:
        weather[col] = pd.to_numeric(
            weather[col],
            errors="coerce",
        )

    weather["rainfall"] = weather["rainfall"].fillna(0)

    weather = (
        weather
        .dropna(subset=["date"])
        .sort_values("date")
        .drop_duplicates("date", keep="last")
        .reset_index(drop=True)
    )

    if weather.empty:
        raise RuntimeError(
            "ASOS 응답은 받았지만 변환 후 사용 가능한 날짜 자료가 없습니다."
        )

    # 디버깅/화면 안내용
    weather.attrs["failed_dates"] = sorted(set(failed_dates))

    return weather


# ============================================================
# 4. 기간형 AirKorea 수집
# ============================================================

def _request_air_once(
    start_date: date,
    end_date: date,
    station_name: str,
):
    """AirKorea 한 구간 요청. 반복 실패 시 None 반환."""
    safe_key = urllib.parse.unquote(AIR_SERVICE_KEY)

    params = {
        "serviceKey": safe_key,
        "returnType": "json",
        "numOfRows": "200",
        "pageNo": "1",
        "inqBginDt": start_date.strftime("%Y%m%d"),
        "inqEndDt": end_date.strftime("%Y%m%d"),
        "msrstnName": station_name,
    }

    for wait_seconds in [0, 2, 5]:
        if wait_seconds:
            time.sleep(wait_seconds)

        try:
            response = requests.get(
                AIR_URL,
                params=params,
                timeout=40,
            )
            response.raise_for_status()

            if not response.text.strip().startswith("{"):
                raise RuntimeError(
                    "AirKorea API가 JSON이 아닌 응답을 반환했습니다."
                )

            data = response.json()

            header = (
                data.get("response", {})
                .get("header", {})
            )
            result_code = str(header.get("resultCode", ""))
            result_msg = header.get("resultMsg", "")

            if result_code not in ["00", "0", ""]:
                raise RuntimeError(
                    f"AirKorea API 오류: {result_code} / {result_msg}"
                )

            items = (
                data.get("response", {})
                .get("body", {})
                .get("items", [])
            )

            if isinstance(items, dict):
                items = [items]

            return items or []

        except Exception:
            # 이 함수에서는 전체 앱을 중단시키지 않고
            # 재시도 후 None을 반환하여 상위 함수가 더 작은 기간으로 분할함.
            continue

    return None


def _collect_air_period(
    start_date: date,
    end_date: date,
    station_name: str,
):
    """
    요청 실패 시 기간을 절반씩 재귀 분할.
    1일 요청까지 실패하면 해당 날짜만 failed_dates에 기록.
    """
    items = _request_air_once(
        start_date,
        end_date,
        station_name,
    )

    if items is not None:
        return items, []

    days = (end_date - start_date).days + 1

    if days <= 1:
        return [], [start_date]

    left_days = days // 2
    midpoint = start_date + timedelta(days=left_days - 1)
    second_start = midpoint + timedelta(days=1)

    left_items, left_failed = _collect_air_period(
        start_date,
        midpoint,
        station_name,
    )
    right_items, right_failed = _collect_air_period(
        second_start,
        end_date,
        station_name,
    )

    return (
        left_items + right_items,
        left_failed + right_failed,
    )


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_air_range(
    start_date: date,
    end_date: date,
):
    """
    AirKorea 장기간 수집.
    기본 7일 단위 요청 → 504/timeout 발생 시 자동 분할 →
    최종 실패 날짜만 기록하고 나머지 기간은 계속 수집.

    반환:
        air, station_name, failed_dates
    """
    if not AIR_SERVICE_KEY:
        raise ValueError(
            "Streamlit Secrets에 AIR_SERVICE_KEY 또는 SERVICE_KEY가 없습니다."
        )

    candidates = []
    for name in [AIR_STATION_NAME, "영천시", "영천"]:
        if name and name not in candidates:
            candidates.append(name)

    for station_name in candidates:
        all_items = []
        failed_dates = []
        chunk_start = start_date

        while chunk_start <= end_date:
            chunk_end = min(
                chunk_start + timedelta(days=6),
                end_date,
            )

            items, failed = _collect_air_period(
                chunk_start,
                chunk_end,
                station_name,
            )

            all_items.extend(items)
            failed_dates.extend(failed)

            chunk_start = chunk_end + timedelta(days=1)
            time.sleep(0.15)

        if not all_items:
            continue

        air = pd.DataFrame(all_items).rename(columns={
            "msurDt": "date",
            "pm10Value": "pm10",
            "pm25Value": "pm25",
            "o3Value": "o3",
            "no2Value": "no2",
            "coValue": "co",
            "so2Value": "so2",
        })

        cols = [
            "date", "pm10", "pm25",
            "o3", "no2", "co", "so2",
        ]

        for col in cols:
            if col not in air.columns:
                air[col] = np.nan

        air = air[cols].copy()

        air["date"] = pd.to_datetime(
            air["date"],
            errors="coerce",
        ).dt.floor("D")

        for col in cols[1:]:
            air[col] = pd.to_numeric(
                air[col].replace(
                    ["-", "", "null", "None"],
                    np.nan,
                ),
                errors="coerce",
            )

        air = (
            air.dropna(subset=["date"])
            .groupby("date", as_index=False)
            .mean(numeric_only=True)
            .sort_values("date")
            .reset_index(drop=True)
        )

        if not air.empty:
            # 중복 제거 및 정렬
            failed_dates = sorted(set(failed_dates))
            return air, station_name, failed_dates

    raise RuntimeError(
        "영천 대기환경 자료를 한 건도 수집하지 못했습니다. "
        "AirKorea API 상태와 AIR_STATION_NAME 설정을 확인하세요."
    )


# ============================================================
# 5. 전처리 + 파생변수
# ============================================================

def prepare_environment(weather: pd.DataFrame, air: pd.DataFrame):
    merged = pd.merge(weather, air, on="date", how="left")
    merged = merged.sort_values("date").reset_index(drop=True)

    non_negative = [
        "rainfall", "wind_speed", "sunshine_hours", "humidity",
        "pm10", "pm25", "o3", "no2", "co", "so2",
    ]
    for col in non_negative:
        if col in merged.columns:
            merged.loc[merged[col] < 0, col] = np.nan

    merged.loc[
        (merged["humidity"] < 0) | (merged["humidity"] > 100),
        "humidity",
    ] = np.nan
    merged.loc[merged["pm10"] > 1000, "pm10"] = np.nan
    merged.loc[merged["pm25"] > 500, "pm25"] = np.nan
    merged["rainfall"] = merged["rainfall"].fillna(0)

    fill_cols = [
        "temp_avg", "temp_max", "temp_min", "humidity",
        "wind_speed", "sunshine_hours", "ground_temp",
        "pm10", "pm25", "o3", "no2", "co", "so2",
    ]
    fill_cols = [c for c in fill_cols if c in merged.columns]
    merged[fill_cols] = merged[fill_cols].ffill()
    merged = merged.dropna(subset=fill_cols).reset_index(drop=True)

    if len(merged) < 28:
        raise ValueError(
            f"전처리 후 {len(merged)}일만 남았습니다. 28일 파생변수 계산이 불가능합니다."
        )

    # 날짜 누락 확인
    expected = pd.date_range(merged["date"].min(), merged["date"].max(), freq="D")
    missing_dates = expected.difference(merged["date"])
    if len(missing_dates):
        raise ValueError(
            "ASOS 기준 날짜가 누락되어 rolling 파생변수를 정확히 계산할 수 없습니다. "
            f"누락 {len(missing_dates)}일: {list(missing_dates[:10])}"
        )

    features = create_environment_features(
        merged.copy(),
        fill_remaining_numeric=False,
    )

    features["date"] = pd.to_datetime(features["date"], errors="coerce").dt.floor("D")
    return features.sort_values("date").reset_index(drop=True)


# ============================================================
# 6. 모든 날짜 × 모든 문화유산 예측
# ============================================================

def predict_all_dates(
    environment_df: pd.DataFrame,
    heritage_df: pd.DataFrame,
    model,
    feature_cols,
    train_medians,
):
    env = environment_df.loc[
        environment_df["date"] >= pd.Timestamp(DISPLAY_START_DATE)
    ].copy()

    if env.empty:
        raise ValueError("2026-01-01 이후 예측 가능한 환경 Feature가 없습니다.")

    # 날짜 환경 × 영천 문화유산 전체 Cross Join
    prediction_df = (
        env.assign(_key=1)
        .merge(heritage_df.assign(_key=1), on="_key", how="inner")
        .drop(columns="_key")
    )

    result = run_prediction(
        model,
        feature_cols,
        train_medians,
        prediction_df,
    )

    result["grade_rank"] = (
        result["predicted_grade"].map(GRADE_RANK).fillna(-1).astype(int)
    )

    # 날짜별 대표 등급 = 해당 날짜 문화유산 중 가장 높은 등급
    daily_grade = (
        result.groupby("date", as_index=False)["grade_rank"]
        .max()
    )
    reverse_grade = {v: k for k, v in GRADE_RANK.items()}
    daily_grade["daily_grade"] = daily_grade["grade_rank"].map(reverse_grade)

    counts = (
        result.groupby(["date", "predicted_grade"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=GRADE_ORDER, fill_value=0)
        .reset_index()
        .rename(columns={
            "안전": "safe_count",
            "주의": "caution_count",
            "위험": "danger_count",
        })
    )

    risk_stats = (
        result.groupby("date", as_index=False)
        .agg(
            max_risk_index=("risk_index", "max"),
            avg_risk_index=("risk_index", "mean"),
            max_danger_probability=("danger_probability", "max"),
            heritage_count=("heritage_name", "nunique"),
        )
    )

    daily = (
        daily_grade
        .merge(counts, on="date", how="left")
        .merge(risk_stats, on="date", how="left")
        .drop(columns="grade_rank")
        .sort_values("date")
        .reset_index(drop=True)
    )

    return result, daily


# ============================================================
# 7. 실행 화면
# ============================================================

try:
    heritage_path = find_heritage_path()
    model, feature_cols, train_medians, metadata = load_model_assets()
    heritage_df = load_heritage_data(str(heritage_path))

    model_name = (
        metadata.get("best_model_name")
        or metadata.get("model_name")
        or metadata.get("best_model")
        or type(model).__name__
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("분석 시작일", "2026-01-01")
    c2.metric("수집 시작일", COLLECT_START_DATE.strftime("%Y-%m-%d"))
    c3.metric("요청 종료일", REQUEST_END_DATE.strftime("%Y-%m-%d"))
    c4.metric("최고 학습 모델", str(model_name))

    st.caption(
        "※ 1월 1일부터 28일 rolling 파생변수를 사용할 수 있도록 "
        f"실제 API 수집은 {COLLECT_START_DATE:%Y-%m-%d}부터 시작합니다."
    )

    run = st.button(
        "🚀 2026년 공공데이터 수집 · 파생변수 생성 · 환경 취약도 예측",
        type="primary",
        use_container_width=True,
    )

    if run:
        progress = st.progress(0, text="영천 ASOS 일자료 수집 중...")

        weather = fetch_weather_range(
            COLLECT_START_DATE,
            REQUEST_END_DATE,
        )

        failed_weather_dates = weather.attrs.get("failed_dates", [])

        st.caption(
            f"🌦 ASOS 실제 수집 범위: "
            f"{pd.to_datetime(weather['date']).min():%Y-%m-%d} ~ "
            f"{pd.to_datetime(weather['date']).max():%Y-%m-%d}"
        )

        if failed_weather_dates:
            st.warning(
                f"⚠️ ASOS 서버 응답 실패로 "
                f"{len(failed_weather_dates)}일을 직접 수집하지 못했습니다."
            )
            with st.expander("🌦 ASOS 수집 실패 날짜 확인"):
                st.write([
                    d.strftime("%Y-%m-%d")
                    for d in failed_weather_dates
                ])

        progress.progress(25, text="영천 AirKorea 일자료 수집 중...")

        # ASOS는 전일 일자료가 당일 늦게 공개될 수 있으므로
        # 실제 ASOS 최신 날짜까지만 대기자료를 요청/사용
        actual_end_ts = pd.to_datetime(weather["date"]).max()
        actual_end_date = actual_end_ts.date()

        air, used_station, failed_air_dates = fetch_air_range(
            COLLECT_START_DATE,
            actual_end_date,
        )

        air_min_date = pd.to_datetime(air["date"]).min()
        air_max_date = pd.to_datetime(air["date"]).max()

        st.caption(
            f"🌫 AirKorea 실제 수집 범위({used_station}): "
            f"{air_min_date:%Y-%m-%d} → {air_max_date:%Y-%m-%d}"
        )

        if failed_air_dates:
            st.warning(
                f"⚠️ AirKorea 서버 오류 등으로 "
                f"{len(failed_air_dates)}일의 자료를 직접 수집하지 못했습니다. "
                "해당 날짜는 기상자료와 결합한 뒤 기존 전처리 방식으로 보완합니다."
            )
            with st.expander("🌫 AirKorea 수집 실패 날짜 확인"):
                st.write([
                    d.strftime("%Y-%m-%d")
                    for d in failed_air_dates
                ])

        progress.progress(55, text="기상·대기자료 결합 및 파생변수 생성 중...")

        environment_df = prepare_environment(weather, air)

        env_min_date = pd.to_datetime(environment_df["date"]).min()
        env_max_date = pd.to_datetime(environment_df["date"]).max()

        prediction_env = environment_df.loc[
            environment_df["date"] >= pd.Timestamp(DISPLAY_START_DATE)
        ].copy()

        if prediction_env.empty:
            raise ValueError(
                "2026-01-01 이후 ASOS와 AirKorea가 함께 확보된 예측 가능 기간이 없습니다."
            )

        actual_prediction_start = pd.to_datetime(prediction_env["date"]).min()
        actual_prediction_end = pd.to_datetime(prediction_env["date"]).max()
        actual_prediction_days = int(prediction_env["date"].nunique())

        st.info(
            "📅 공공데이터 실제 가용기간\n\n"
            f"- ASOS: {pd.to_datetime(weather['date']).min():%Y-%m-%d} → "
            f"{pd.to_datetime(weather['date']).max():%Y-%m-%d}\n"
            f"- AirKorea({used_station}): {air_min_date:%Y-%m-%d} → {air_max_date:%Y-%m-%d}\n"
            f"- 기상·대기 결합 후 사용 가능기간: {env_min_date:%Y-%m-%d} → {env_max_date:%Y-%m-%d}\n"
            f"- **최종 예측 가능기간: {actual_prediction_start:%Y-%m-%d} → "
            f"{actual_prediction_end:%Y-%m-%d} ({actual_prediction_days:,}일)**"
        )

        if actual_prediction_start.date() > DISPLAY_START_DATE:
            st.warning(
                f"⚠️ 2026-01-01부터 {actual_prediction_start:%Y-%m-%d} 직전까지는 "
                "예측에 필요한 기상·대기환경 공공데이터의 공통 가용기간에 포함되지 않아 "
                "이번 분석에서 제외됩니다. 미래 시점의 값으로 과거 결측을 채우지 않고, "
                "실제로 확보된 공통 기간만 예측에 사용합니다."
            )

        progress.progress(75, text="최고 학습 모델로 날짜별 예측 중...")

        all_predictions, daily = predict_all_dates(
            environment_df,
            heritage_df,
            model,
            feature_cols,
            train_medians,
        )
        progress.progress(100, text="완료")

        st.session_state["test_2026_environment"] = environment_df
        st.session_state["test_2026_all_predictions"] = all_predictions
        st.session_state["test_2026_daily"] = daily
        st.session_state["test_2026_station"] = used_station
        st.session_state["test_2026_model_name"] = str(model_name)

    daily = st.session_state.get("test_2026_daily")
    all_predictions = st.session_state.get("test_2026_all_predictions")
    environment_df = st.session_state.get("test_2026_environment")

    if isinstance(daily, pd.DataFrame) and not daily.empty:
        latest = daily.iloc[-1]

        actual_start = pd.to_datetime(daily["date"]).min()
        actual_end = pd.to_datetime(daily["date"]).max()

        st.success(
            f"예측 완료 · 실제 예측기간: {actual_start:%Y-%m-%d} → "
            f"{actual_end:%Y-%m-%d} · 최신 대표 등급: {latest['daily_grade']}"
        )

        p1, p2, p3 = st.columns(3)
        p1.metric("실제 예측 시작일", f"{actual_start:%Y-%m-%d}")
        p2.metric("실제 예측 종료일", f"{actual_end:%Y-%m-%d}")
        p3.metric("예측 일수", f"{len(daily):,}일")

        m1, m2, m3 = st.columns(3)
        m1.metric("최신 대표 등급", latest["daily_grade"])
        m2.metric("최신 위험 문화유산", f"{int(latest['danger_count']):,}개")
        m3.metric("최신 최대 위험지수", f"{latest['max_risk_index']:.1f}")

        st.subheader("📈 2026 날짜별 대표 환경 취약도")

        chart_df = daily.copy()
        chart_df["등급값"] = chart_df["daily_grade"].map({"안전": 0, "주의": 1, "위험": 2})

        fig = px.scatter(
            chart_df,
            x="date",
            y="등급값",
            color="daily_grade",
            color_discrete_map=GRADE_COLOR,
            category_orders={"daily_grade": GRADE_ORDER},
            hover_data={
                "date": True,
                "daily_grade": True,
                "safe_count": True,
                "caution_count": True,
                "danger_count": True,
                "max_risk_index": ":.1f",
                "avg_risk_index": ":.1f",
                "등급값": False,
            },
            labels={
                "date": "날짜",
                "등급값": "위험도",
                "daily_grade": "대표 등급",
            },
        )
        fig.update_yaxes(
            tickmode="array",
            tickvals=[0, 1, 2],
            ticktext=["안전", "주의", "위험"],
            range=[-0.3, 2.3],
        )
        fig.update_traces(marker={"size": 8})
        fig.update_layout(height=430)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("📊 날짜별 안전·주의·위험 문화유산 수")

        count_long = daily.melt(
            id_vars=["date"],
            value_vars=["safe_count", "caution_count", "danger_count"],
            var_name="등급",
            value_name="문화유산 수",
        )
        count_long["등급"] = count_long["등급"].map({
            "safe_count": "안전",
            "caution_count": "주의",
            "danger_count": "위험",
        })

        fig2 = px.area(
            count_long,
            x="date",
            y="문화유산 수",
            color="등급",
            color_discrete_map=GRADE_COLOR,
            category_orders={"등급": GRADE_ORDER},
        )
        fig2.update_layout(height=420)
        st.plotly_chart(fig2, use_container_width=True)

        st.subheader("📋 날짜별 환경 취약도 표")

        display_daily = daily.copy()
        display_daily["date"] = display_daily["date"].dt.strftime("%Y-%m-%d")
        display_daily = display_daily.rename(columns={
            "date": "날짜",
            "daily_grade": "대표 위험도",
            "safe_count": "안전 수",
            "caution_count": "주의 수",
            "danger_count": "위험 수",
            "max_risk_index": "최대 위험지수",
            "avg_risk_index": "평균 위험지수",
            "max_danger_probability": "최대 위험확률(%)",
            "heritage_count": "문화유산 수",
        })

        st.dataframe(
            display_daily,
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "⬇️ 날짜별 위험도 CSV 다운로드",
            data=display_daily.to_csv(index=False).encode("utf-8-sig"),
            file_name="yeongcheon_2026_daily_risk.csv",
            mime="text/csv",
            use_container_width=True,
        )

        with st.expander("🔎 특정 날짜의 문화유산별 예측 상세 보기"):
            selected_date = st.date_input(
                "상세 조회 날짜",
                value=daily["date"].max().date(),
                min_value=daily["date"].min().date(),
                max_value=daily["date"].max().date(),
                key="test_2026_detail_date",
            )

            detail = all_predictions.loc[
                all_predictions["date"].dt.date == selected_date,
                [
                    "heritage_name", "material", "exposure",
                    "predicted_grade", "risk_index",
                    "safe_probability", "caution_probability", "danger_probability",
                ],
            ].copy()

            detail = detail.sort_values(
                ["risk_index", "danger_probability"],
                ascending=False,
            )

            st.dataframe(
                detail.rename(columns={
                    "heritage_name": "문화유산명",
                    "material": "재질",
                    "exposure": "노출",
                    "predicted_grade": "예측 등급",
                    "risk_index": "위험지수",
                    "safe_probability": "안전확률(%)",
                    "caution_probability": "주의확률(%)",
                    "danger_probability": "위험확률(%)",
                }),
                use_container_width=True,
                hide_index=True,
            )

        with st.expander("🧮 생성된 환경 파생변수 확인"):
            st.dataframe(
                environment_df.loc[
                    environment_df["date"] >= pd.Timestamp(DISPLAY_START_DATE)
                ].sort_values("date", ascending=False),
                use_container_width=True,
                hide_index=True,
            )

except Exception as e:
    st.error(f"❌ 2026년 미학습 기간 예측 페이지 실행 실패: {e}")
    st.exception(e)
