from __future__ import annotations

from datetime import date, datetime, timedelta
import time
import urllib.parse

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
    page_title="전일~40일전 환경 데이터",
    page_icon="📅",
    layout="wide",
)

st.title("📅 예측용 전일~40일전 환경 데이터")
st.caption(
    "문화재 환경 취약도 예측에 사용하는 최근 40일의 "
    "기상·대기환경 자료와 7일·28일 파생변수를 확인합니다."
)

st.info(
    "📌 28일 rolling 파생변수를 안정적으로 계산하기 위해 "
    "예측 기준일을 포함하여 최근 40일 데이터를 수집합니다. "
    "이 페이지에서는 모델을 새로 학습하거나 문화재별 예측을 실행하지 않습니다."
)


# ============================================================
# 2. API / 기본 설정
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

# 기존 SERVICE_KEY를 사용 중인 웹앱도 동작하도록 fallback 지원
ASOS_SERVICE_KEY = st.secrets.get(
    "ASOS_SERVICE_KEY",
    st.secrets.get("SERVICE_KEY", ""),
)

AIR_SERVICE_KEY = st.secrets.get(
    "AIR_SERVICE_KEY",
    st.secrets.get("SERVICE_KEY", ""),
)

# 측정소 명칭은 환경에 따라 "영천" / "영천시"가 다를 수 있어
# Secrets로 변경 가능하도록 구성
AIR_STATION_NAME = st.secrets.get(
    "AIR_STATION_NAME",
    "영천",
)

TODAY = date.today()
DEFAULT_TARGET_DATE = TODAY - timedelta(days=1)


# ============================================================
# 3. 안전한 숫자 변환
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
# 7. 화면 입력
# ============================================================

col_date, col_info = st.columns(
    [1, 2],
    vertical_alignment="bottom",
)

with col_date:
    target_date = st.date_input(
        "📌 예측 기준일",
        value=DEFAULT_TARGET_DATE,
        max_value=DEFAULT_TARGET_DATE,
        help="기본값은 어제입니다. 오늘과 미래 날짜는 사용할 수 없습니다.",
    )

with col_info:
    start_target_date = (
        target_date
        - timedelta(days=39)
    )

    st.write(
        f"**수집 범위:** "
        f"{start_target_date:%Y-%m-%d} ~ "
        f"{target_date:%Y-%m-%d} "
        f"(기준일 포함 40일)"
    )


# ============================================================
# 8. 세션 상태
# ============================================================

if "recent_40_weather" not in st.session_state:
    st.session_state.recent_40_weather = None

if "recent_40_air" not in st.session_state:
    st.session_state.recent_40_air = None

if "recent_40_environment" not in st.session_state:
    st.session_state.recent_40_environment = None

if "recent_40_quality" not in st.session_state:
    st.session_state.recent_40_quality = None

if "recent_40_target_date" not in st.session_state:
    st.session_state.recent_40_target_date = None

if "recent_40_air_station" not in st.session_state:
    st.session_state.recent_40_air_station = None


# ============================================================
# 9. 데이터 수집 실행
# ============================================================

if st.button(
    "📡 최근 40일 환경 데이터 불러오기",
    type="primary",
    use_container_width=True,
):
    try:
        status = st.status(
            "최근 40일 환경 데이터 수집 준비 중...",
            expanded=True,
        )

        status.update(
            label="🌦 기상청 ASOS 최근 40일 수집 중...",
            state="running",
        )

        weather = fetch_recent_weather(
            target_date
        )

        status.update(
            label="🌫 AirKorea 최근 40일 수집 중...",
            state="running",
        )

        air, used_station = fetch_recent_air(
            target_date
        )

        status.update(
            label="🔗 기상·대기환경 병합 및 파생변수 생성 중...",
            state="running",
        )

        realtime_df, quality = prepare_recent_environment(
            weather,
            air,
        )

        # 지정일 정확히 존재하는지 검사
        target_rows = realtime_df.loc[
            realtime_df["date"]
            == pd.Timestamp(target_date)
        ]

        if target_rows.empty:
            raise ValueError(
                f"{target_date:%Y-%m-%d} 기준일의 "
                "최종 환경 데이터가 없습니다. "
                "이전 날짜를 임의로 대신 사용하지 않습니다."
            )

        # Prediction 페이지가 바로 재사용할 수 있도록 저장
        st.session_state.recent_40_weather = weather
        st.session_state.recent_40_air = air
        st.session_state.recent_40_environment = realtime_df
        st.session_state.recent_40_quality = quality
        st.session_state.recent_40_target_date = target_date
        st.session_state.recent_40_air_station = used_station

        status.update(
            label="✅ 최근 40일 예측용 환경 데이터 준비 완료",
            state="complete",
            expanded=False,
        )

        st.rerun()

    except Exception as e:
        st.error(
            f"❌ 최근 40일 데이터 구축 실패: {e}"
        )


# ============================================================
# 10. 결과 화면
# ============================================================

realtime_df = (
    st.session_state.recent_40_environment
)

quality = (
    st.session_state.recent_40_quality
)

loaded_target_date = (
    st.session_state.recent_40_target_date
)

used_station = (
    st.session_state.recent_40_air_station
)


if realtime_df is not None:

    # ========================================================
    # 10-1. 수집 상태
    # ========================================================

    st.markdown("---")
    st.subheader("✅ 예측 입력 데이터 준비 현황")

    m1, m2, m3, m4, m5 = st.columns(5)

    m1.metric(
        "사용 가능 일수",
        f"{len(realtime_df)}일",
    )

    m2.metric(
        "시작일",
        realtime_df["date"]
        .min()
        .strftime("%Y-%m-%d"),
    )

    m3.metric(
        "예측 기준일",
        pd.Timestamp(
            loaded_target_date
        ).strftime("%Y-%m-%d"),
    )

    m4.metric(
        "AirKorea 측정소",
        str(used_station),
    )

    m5.metric(
        "날짜 누락",
        f"{quality['missing_calendar_days']}일",
    )

    if len(realtime_df) >= 28:
        st.success(
            "✅ 28일 rolling 파생변수를 계산할 수 있는 "
            "충분한 데이터가 확보되었습니다."
        )

    # ========================================================
    # 10-2. 기준일 환경 요약
    # ========================================================

    target_row_df = realtime_df.loc[
        realtime_df["date"]
        == pd.Timestamp(
            loaded_target_date
        )
    ]

    target_row = (
        target_row_df.iloc[0]
    )

    st.markdown("---")
    st.subheader(
        f"📌 {pd.Timestamp(loaded_target_date):%Y-%m-%d} "
        "기준 환경 요약"
    )

    row1 = st.columns(6)

    row1[0].metric(
        "평균기온",
        f"{target_row['temp_avg']:.1f} ℃",
    )

    row1[1].metric(
        "습도",
        f"{target_row['humidity']:.1f} %",
    )

    row1[2].metric(
        "일 강수량",
        f"{target_row['rainfall']:.1f} mm",
    )

    row1[3].metric(
        "PM10",
        f"{target_row['pm10']:.1f}",
    )

    row1[4].metric(
        "PM2.5",
        f"{target_row['pm25']:.1f}",
    )

    row1[5].metric(
        "SO₂",
        f"{target_row['so2']:.4f}",
    )

    row2 = st.columns(6)

    metric_specs = [
        (
            "최근 7일 강수",
            "rainfall_7d",
            "{:.1f} mm",
        ),
        (
            "28일 RH≥75%",
            "rh75_days_28",
            "{:.0f}일",
        ),
        (
            "RH≥75 연속",
            "rh75_consecutive_days",
            "{:.0f}일",
        ),
        (
            "28일 목조 곰팡이",
            "wood_mold_days_28",
            "{:.0f}일",
        ),
        (
            "28일 RH≥70%",
            "rh70_days_28",
            "{:.0f}일",
        ),
        (
            "7일 PM 누적",
            "pm_load_7d",
            "{:.1f}",
        ),
    ]

    for col, (
        label,
        feature,
        fmt,
    ) in zip(
        row2,
        metric_specs,
    ):
        if feature in target_row.index:
            value = target_row[
                feature
            ]

            col.metric(
                label,
                fmt.format(value),
            )

    # ========================================================
    # 10-3. 40일 전체 원자료
    # ========================================================

    st.markdown("---")
    st.subheader("📋 최근 40일 기상·대기환경 데이터")

    basic_cols = [
        "date",
        "temp_avg",
        "temp_max",
        "temp_min",
        "humidity",
        "rainfall",
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

    basic_cols = [
        col
        for col in basic_cols
        if col in realtime_df.columns
    ]

    basic_view = (
        realtime_df[basic_cols]
        .sort_values(
            "date",
            ascending=False,
        )
        .copy()
    )

    st.dataframe(
        basic_view,
        use_container_width=True,
        height=520,
        hide_index=True,
    )

    # ========================================================
    # 10-4. 예측 핵심 파생변수
    # ========================================================

    st.markdown("---")
    st.subheader("🧮 최근 40일 예측용 파생변수")

    derived_cols = [
        "date",
        "temp_range",
        "temp_change",
        "humidity_change",
        "humidity_std3",
        "rainfall_7d",
        "rh60_days_28",
        "rh75_days_28",
        "rh95_days_28",
        "rh75_consecutive_days",
        "rh95_consecutive_days",
        "wood_mold_days_28",
        "rh70_days_28",
        "rh70_consecutive_days",
        "metal_so2_humidity",
        "pm_total",
        "pm_load_3d",
        "pm_load_7d",
        "so2_ma7",
        "no2_ma7",
        "o3_ma7",
        "season",
    ]

    derived_cols = [
        col
        for col in derived_cols
        if col in realtime_df.columns
    ]

    derived_view = (
        realtime_df[derived_cols]
        .sort_values(
            "date",
            ascending=False,
        )
        .copy()
    )

    st.dataframe(
        derived_view,
        use_container_width=True,
        height=520,
        hide_index=True,
    )

    # ========================================================
    # 10-5. 추이 그래프
    # ========================================================

    st.markdown("---")
    st.subheader("📈 최근 40일 환경 변화")

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        fig_temp_hum = go.Figure()

        fig_temp_hum.add_trace(
            go.Scatter(
                x=realtime_df["date"],
                y=realtime_df["temp_avg"],
                mode="lines+markers",
                name="평균기온(℃)",
                yaxis="y1",
            )
        )

        fig_temp_hum.add_trace(
            go.Scatter(
                x=realtime_df["date"],
                y=realtime_df["humidity"],
                mode="lines+markers",
                name="습도(%)",
                yaxis="y2",
            )
        )

        fig_temp_hum.update_layout(
            title="평균기온·습도 변화",
            height=420,
            hovermode="x unified",
            yaxis=dict(
                title="기온(℃)",
            ),
            yaxis2=dict(
                title="습도(%)",
                overlaying="y",
                side="right",
                range=[0, 100],
            ),
            legend=dict(
                orientation="h",
            ),
        )

        st.plotly_chart(
            fig_temp_hum,
            use_container_width=True,
        )

    with chart_col2:
        air_chart = realtime_df[
            [
                "date",
                "pm10",
                "pm25",
            ]
        ].melt(
            id_vars="date",
            var_name="항목",
            value_name="농도",
        )

        fig_air = px.line(
            air_chart,
            x="date",
            y="농도",
            color="항목",
            markers=True,
            title="PM10·PM2.5 변화",
        )

        fig_air.update_layout(
            height=420,
            hovermode="x unified",
        )

        st.plotly_chart(
            fig_air,
            use_container_width=True,
        )

    chart_col3, chart_col4 = st.columns(2)

    with chart_col3:
        if "rainfall_7d" in realtime_df.columns:
            fig_rain = px.bar(
                realtime_df,
                x="date",
                y="rainfall_7d",
                title="최근 7일 누적 강수량",
                labels={
                    "rainfall_7d": "7일 누적 강수량(mm)",
                    "date": "날짜",
                },
            )

            fig_rain.update_layout(
                height=390,
            )

            st.plotly_chart(
                fig_rain,
                use_container_width=True,
            )

    with chart_col4:
        humid_features = [
            col
            for col in [
                "rh60_days_28",
                "rh70_days_28",
                "rh75_days_28",
                "rh95_days_28",
            ]
            if col in realtime_df.columns
        ]

        if humid_features:
            humid_long = (
                realtime_df[
                    ["date"]
                    + humid_features
                ]
                .melt(
                    id_vars="date",
                    var_name="조건",
                    value_name="최근 28일 일수",
                )
            )

            fig_rh = px.line(
                humid_long,
                x="date",
                y="최근 28일 일수",
                color="조건",
                title="28일 습도 조건 누적일수",
            )

            fig_rh.update_layout(
                height=390,
                yaxis_range=[0, 28],
            )

            st.plotly_chart(
                fig_rh,
                use_container_width=True,
            )

    # ========================================================
    # 10-6. 데이터 품질
    # ========================================================

    st.markdown("---")

    with st.expander(
        "🔍 수집·전처리 품질 확인",
        expanded=False,
    ):
        q1, q2, q3, q4 = st.columns(4)

        q1.metric(
            "기상 API 일수",
            f"{quality['weather_days']}일",
        )

        q2.metric(
            "대기 API 일수",
            f"{quality['air_days']}일",
        )

        q3.metric(
            "대기 결측 발생일",
            f"{quality['air_missing_before_ffill']}일",
        )

        q4.metric(
            "초기 결측 제거",
            f"{quality['initial_rows_removed']}일",
        )

        st.info(
            "결측값은 시간 순서를 유지하기 위해 ffill만 사용합니다. "
            "미래 날짜의 값을 이전 날짜에 채우는 bfill은 사용하지 않습니다."
        )

    # ========================================================
    # 10-7. CSV 다운로드
    # ========================================================

    st.markdown("---")

    csv_bytes = (
        realtime_df
        .to_csv(index=False)
        .encode("utf-8-sig")
    )

    st.download_button(
        "📥 최근 40일 예측용 데이터 CSV 다운로드",
        data=csv_bytes,
        file_name=(
            f"yeongcheon_prediction_input_40days_"
            f"{pd.Timestamp(loaded_target_date):%Y%m%d}.csv"
        ),
        mime="text/csv",
        type="primary",
        use_container_width=True,
    )

    st.caption(
        "이 페이지에서 생성된 recent_40_weather / recent_40_air / "
        "recent_40_environment 세션 데이터는 다음 '예측' 페이지에서 "
        "동일 세션 안에서 재사용할 수 있습니다."
    )
