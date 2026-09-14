from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# ============================================================
# 1. 페이지 설정
# ============================================================

st.set_page_config(
    page_title="최근 40일 환경 데이터",
    page_icon="🔵",
    layout="wide",
)

st.title("🔵 최근 40일 환경 데이터")
st.caption(
    "문화유산 환경 취약도 예측 과정에서 자동 수집된 최근 40일 "
    "기상·대기환경 데이터와 7일·28일 파생변수를 시각화합니다."
)

st.info(
    "📌 이 페이지에서는 데이터를 새로 수집하지 않습니다. "
    "먼저 **문화유산 취약도 예측** 페이지에서 자동 수집·예측을 실행하면 "
    "같은 세션의 데이터를 우선 사용하고, 세션이 초기화된 경우에는 "
    "직전에 저장된 최근 40일 환경 데이터를 자동으로 불러와 분석합니다."
)

DATA_DIR = Path("data/processed")
LATEST_ENVIRONMENT_PATH = (
    DATA_DIR
    / "latest_40_environment.csv"
)


# ============================================================
# 2. 화면 스타일
# ============================================================

st.markdown(
    """
    <style>
        .block-container {
            padding-top: 1rem !important;
            padding-bottom: 2rem !important;
        }

        .data-card {
            padding: 14px 16px;
            border-radius: 14px;
            border: 1px solid rgba(128,128,128,0.22);
            background: rgba(128,128,128,0.04);
            margin-bottom: 10px;
        }

        .data-card h4 {
            margin: 0 0 4px 0;
        }

        .data-card p {
            margin: 0;
            opacity: 0.82;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 3. 한글 표시명
# ============================================================

COLUMN_KR = {
    "date": "날짜",

    # 원자료
    "temp_avg": "평균기온(℃)",
    "temp_max": "최고기온(℃)",
    "temp_min": "최저기온(℃)",
    "humidity": "평균습도(%)",
    "rainfall": "강수량(mm)",
    "wind_speed": "평균풍속(m/s)",
    "sunshine_hours": "일조시간(hr)",
    "ground_temp": "지면온도(℃)",
    "pm10": "PM10(㎍/㎥)",
    "pm25": "PM2.5(㎍/㎥)",
    "o3": "O₃(ppm)",
    "no2": "NO₂(ppm)",
    "co": "CO(ppm)",
    "so2": "SO₂(ppm)",

    # 기본 파생변수
    "temp_range": "일교차(℃)",
    "temp_change": "평균기온 변화량",
    "humidity_change": "습도 변화량",
    "humidity_std3": "3일 습도 변동성",
    "rainfall_7d": "최근 7일 누적 강수량",

    # 습도 지속성
    "rh60_days_7": "최근 7일 RH>60% 일수",
    "rh60_days_28": "최근 28일 RH>60% 일수",
    "rh75_days_7": "최근 7일 RH≥75% 일수",
    "rh75_days_28": "최근 28일 RH≥75% 일수",
    "rh95_days_7": "최근 7일 RH≥95% 일수",
    "rh95_days_28": "최근 28일 RH≥95% 일수",
    "rh75_consecutive_days": "RH≥75% 연속일수",
    "rh95_consecutive_days": "RH≥95% 연속일수",

    # 목조
    "wood_mold_condition": "목조 곰팡이 조건",
    "wood_mold_days_7": "최근 7일 목조 곰팡이 조건 일수",
    "wood_mold_days_28": "최근 28일 목조 곰팡이 조건 일수",

    # 금속
    "rh70_metal": "금속 RH≥70% 여부",
    "rh70_days_7": "최근 7일 RH≥70% 일수",
    "rh70_days_28": "최근 28일 RH≥70% 일수",
    "rh70_consecutive_days": "RH≥70% 연속일수",
    "metal_so2_humidity": "금속 고습·SO₂ 복합지표",

    # 대기오염
    "pm_total": "PM10+PM2.5",
    "pm_load_3d": "최근 3일 미세먼지 부하",
    "pm_load_7d": "최근 7일 미세먼지 부하",
    "so2_ma7": "최근 7일 SO₂ 평균",
    "no2_ma7": "최근 7일 NO₂ 평균",
    "o3_ma7": "최근 7일 O₃ 평균",

    # 기타
    "temp_over_20": "20℃ 초과 여부",
    "rh_over_60": "RH>60% 여부",
    "rh75": "RH≥75% 여부",
    "rh95": "RH≥95% 여부",
    "season": "계절",
    "month": "월",
}


# ============================================================
# 4. 유틸리티
# ============================================================

def safe_metric(value, digits=1, suffix=""):
    if value is None or pd.isna(value):
        return "-"

    try:
        return f"{float(value):.{digits}f}{suffix}"
    except Exception:
        return str(value)


def get_session_environment():
    """
    최근 40일 환경 데이터를 다음 우선순위로 불러온다.

    1순위
    - 문화유산 취약도 예측 페이지가 현재 세션에 저장한 데이터

    2순위
    - 직전 예측 시 저장한
      data/processed/latest_40_environment.csv

    반환값의 data_source는
    "session" 또는 "csv"이다.
    """

    realtime_df = st.session_state.get(
        "recent_40_environment"
    )

    weather_df = st.session_state.get(
        "recent_40_weather"
    )

    air_df = st.session_state.get(
        "recent_40_air"
    )

    quality = st.session_state.get(
        "recent_40_quality"
    )

    target_date = st.session_state.get(
        "recent_40_target_date"
    )

    air_station = st.session_state.get(
        "recent_40_air_station"
    )

    data_source = None

    # --------------------------------------------------------
    # 1순위: 현재 Streamlit 세션 데이터
    # --------------------------------------------------------
    if (
        isinstance(
            realtime_df,
            pd.DataFrame,
        )
        and not realtime_df.empty
    ):
        data_source = "session"

    # --------------------------------------------------------
    # 2순위: 저장된 CSV fallback
    # --------------------------------------------------------
    else:
        realtime_df = None

        if LATEST_ENVIRONMENT_PATH.exists():
            try:
                realtime_df = pd.read_csv(
                    LATEST_ENVIRONMENT_PATH,
                    encoding="utf-8-sig",
                )

                if (
                    isinstance(
                        realtime_df,
                        pd.DataFrame,
                    )
                    and not realtime_df.empty
                ):
                    data_source = "csv"

                    # CSV에는 별도 target_date가 없으므로
                    # 가장 최근 날짜를 기준일로 사용
                    if "date" in realtime_df.columns:
                        parsed_dates = pd.to_datetime(
                            realtime_df["date"],
                            errors="coerce",
                        )

                        if parsed_dates.notna().any():
                            target_date = (
                                parsed_dates.max()
                            )

            except Exception as e:
                st.warning(
                    "저장된 최근 40일 환경 데이터 파일을 "
                    f"불러오지 못했습니다: {e}"
                )
                realtime_df = None

    # --------------------------------------------------------
    # 사용할 수 있는 데이터가 없는 경우
    # --------------------------------------------------------
    if (
        realtime_df is None
        or not isinstance(
            realtime_df,
            pd.DataFrame,
        )
        or realtime_df.empty
    ):
        return (
            None,
            weather_df,
            air_df,
            quality,
            target_date,
            air_station,
            None,
        )

    # --------------------------------------------------------
    # 날짜 정리
    # --------------------------------------------------------
    df = realtime_df.copy()

    if "date" not in df.columns:
        st.error(
            "최근 40일 환경 데이터에 date 컬럼이 없습니다."
        )

        return (
            None,
            weather_df,
            air_df,
            quality,
            target_date,
            air_station,
            data_source,
        )

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce",
    )

    df = (
        df
        .dropna(
            subset=["date"]
        )
        .sort_values("date")
        .reset_index(drop=True)
    )

    if df.empty:
        return (
            None,
            weather_df,
            air_df,
            quality,
            target_date,
            air_station,
            data_source,
        )

    return (
        df,
        weather_df,
        air_df,
        quality,
        target_date,
        air_station,
        data_source,
    )


# ============================================================
# 5. 세션 데이터 불러오기
# ============================================================

(
    realtime_df,
    weather_df,
    air_df,
    quality,
    target_date,
    air_station,
    data_source,
) = get_session_environment()


if realtime_df is None:
    st.warning(
        "⚠️ 최근 40일 환경 데이터를 찾지 못했습니다."
    )

    st.markdown(
        """
        <div class="data-card">
            <h4>먼저 문화유산 취약도 예측을 실행해주세요.</h4>
            <p>
                예측 페이지에서 버튼을 누르면 전일 기준 최근 40일 환경 데이터를
                자동 수집하고 파생변수를 생성합니다. 이 자료는 현재 세션과
                data/processed/latest_40_environment.csv에 저장되며,
                이후 이 페이지에서 바로 시각화할 수 있습니다.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.stop()


# ============================================================
# 6. 상단 수집 현황
# ============================================================

loaded_target_date = pd.Timestamp(
    target_date
) if target_date is not None else realtime_df["date"].max()

start_date = realtime_df["date"].min()
end_date = realtime_df["date"].max()

st.markdown("---")
st.subheader("✅ 최근 40일 데이터 준비 현황")

m1, m2, m3, m4, m5 = st.columns(5)

m1.metric(
    "사용 가능 일수",
    f"{len(realtime_df)}일",
)

m2.metric(
    "시작일",
    start_date.strftime("%Y-%m-%d"),
)

m3.metric(
    "예측 기준일",
    loaded_target_date.strftime("%Y-%m-%d"),
)

m4.metric(
    "AirKorea 측정소",
    str(air_station) if air_station else "-",
)

missing_days = (
    quality.get(
        "missing_calendar_days",
        0,
    )
    if isinstance(
        quality,
        dict,
    )
    else 0
)

m5.metric(
    "날짜 누락",
    f"{missing_days}일",
)

if data_source == "session":
    st.caption(
        "📡 데이터 출처: 현재 예측 세션에서 전달된 최근 40일 환경 데이터"
    )

elif data_source == "csv":
    st.caption(
        "💾 데이터 출처: 직전 예측에서 저장된 "
        "data/processed/latest_40_environment.csv"
    )

if len(realtime_df) >= 28:
    st.success(
        "✅ 28일 rolling 파생변수를 계산할 수 있는 충분한 데이터가 확보되었습니다."
    )


# ============================================================
# 7. 기준일 환경 요약
# ============================================================

target_rows = realtime_df.loc[
    realtime_df["date"].dt.floor("D")
    == loaded_target_date.floor("D")
]

if target_rows.empty:
    target_row = realtime_df.iloc[-1]
    st.warning(
        "지정된 예측 기준일 행을 찾지 못해 현재 세션의 가장 최근 날짜를 표시합니다."
    )
else:
    target_row = target_rows.iloc[-1]

st.markdown("---")
st.subheader(
    f"📌 {pd.Timestamp(target_row['date']):%Y-%m-%d} 기준 환경 요약"
)

row1 = st.columns(6)

row1[0].metric(
    "평균기온",
    safe_metric(
        target_row.get("temp_avg"),
        1,
        " ℃",
    ),
)

row1[1].metric(
    "평균습도",
    safe_metric(
        target_row.get("humidity"),
        1,
        " %",
    ),
)

row1[2].metric(
    "강수량",
    safe_metric(
        target_row.get("rainfall"),
        1,
        " mm",
    ),
)

row1[3].metric(
    "PM10",
    safe_metric(
        target_row.get("pm10"),
        1,
        " ㎍/㎥",
    ),
)

row1[4].metric(
    "PM2.5",
    safe_metric(
        target_row.get("pm25"),
        1,
        " ㎍/㎥",
    ),
)

row1[5].metric(
    "SO₂",
    safe_metric(
        target_row.get("so2"),
        4,
        " ppm",
    ),
)

row2 = st.columns(6)

row2[0].metric(
    "일교차",
    safe_metric(
        target_row.get("temp_range"),
        1,
        " ℃",
    ),
)

row2[1].metric(
    "7일 누적 강수",
    safe_metric(
        target_row.get("rainfall_7d"),
        1,
        " mm",
    ),
)

row2[2].metric(
    "28일 RH>60%",
    safe_metric(
        target_row.get("rh60_days_28"),
        0,
        "일",
    ),
)

row2[3].metric(
    "28일 RH≥75%",
    safe_metric(
        target_row.get("rh75_days_28"),
        0,
        "일",
    ),
)

row2[4].metric(
    "28일 RH≥95%",
    safe_metric(
        target_row.get("rh95_days_28"),
        0,
        "일",
    ),
)

row2[5].metric(
    "미세먼지 7일 부하",
    safe_metric(
        target_row.get("pm_load_7d"),
        1,
    ),
)


# ============================================================
# 8. 기온·습도 추세
# ============================================================

st.markdown("---")
st.subheader("🌡️ 기온·습도 변화")

temp_cols = [
    col
    for col in [
        "temp_avg",
        "temp_max",
        "temp_min",
    ]
    if col in realtime_df.columns
]

left_chart, right_chart = st.columns(2)

with left_chart:
    if temp_cols:
        temp_long = (
            realtime_df[
                ["date"] + temp_cols
            ]
            .rename(
                columns={
                    "date": "날짜",
                    "temp_avg": "평균기온",
                    "temp_max": "최고기온",
                    "temp_min": "최저기온",
                }
            )
            .melt(
                id_vars="날짜",
                var_name="기온 종류",
                value_name="기온(℃)",
            )
        )

        fig_temp = px.line(
            temp_long,
            x="날짜",
            y="기온(℃)",
            color="기온 종류",
            markers=True,
            title="최근 40일 기온 변화",
        )

        fig_temp.update_layout(
            height=420,
            legend_title_text="",
        )

        st.plotly_chart(
            fig_temp,
            use_container_width=True,
        )


with right_chart:
    if "humidity" in realtime_df.columns:
        fig_humidity = px.line(
            realtime_df,
            x="date",
            y="humidity",
            markers=True,
            title="최근 40일 평균습도 변화",
            labels={
                "date": "날짜",
                "humidity": "평균습도(%)",
            },
        )

        fig_humidity.add_hline(
            y=60,
            line_dash="dot",
            annotation_text="RH 60%",
        )

        fig_humidity.add_hline(
            y=75,
            line_dash="dash",
            annotation_text="RH 75%",
        )

        fig_humidity.add_hline(
            y=95,
            line_dash="dashdot",
            annotation_text="RH 95%",
        )

        fig_humidity.update_layout(
            height=420,
            yaxis_range=[0, 100],
        )

        st.plotly_chart(
            fig_humidity,
            use_container_width=True,
        )


# ============================================================
# 9. 강수·일교차
# ============================================================

st.markdown("---")
st.subheader("🌧️ 강수 및 온도 변동")

rain_col, range_col = st.columns(2)

with rain_col:
    if "rainfall" in realtime_df.columns:
        fig_rain = px.bar(
            realtime_df,
            x="date",
            y="rainfall",
            title="일별 강수량",
            labels={
                "date": "날짜",
                "rainfall": "강수량(mm)",
            },
        )

        if "rainfall_7d" in realtime_df.columns:
            fig_rain.add_trace(
                go.Scatter(
                    x=realtime_df["date"],
                    y=realtime_df["rainfall_7d"],
                    mode="lines+markers",
                    name="7일 누적 강수량",
                    yaxis="y2",
                )
            )

            fig_rain.update_layout(
                yaxis2=dict(
                    title="7일 누적 강수량(mm)",
                    overlaying="y",
                    side="right",
                )
            )

        fig_rain.update_layout(
            height=420,
        )

        st.plotly_chart(
            fig_rain,
            use_container_width=True,
        )


with range_col:
    available_range = [
        col
        for col in [
            "temp_range",
            "temp_change",
            "humidity_change",
            "humidity_std3",
        ]
        if col in realtime_df.columns
    ]

    if available_range:
        range_legend = {
            "temp_range": "일교차",
            "temp_change": "기온 변화량",
            "humidity_change": "습도 변화량",
            "humidity_std3": "3일 습도 변동성",
        }

        range_long = (
            realtime_df[
                ["date"] + available_range
            ]
            .rename(
                columns={
                    "date": "날짜",
                    **range_legend,
                }
            )
            .melt(
                id_vars="날짜",
                var_name="변동 지표",
                value_name="값",
            )
        )

        fig_range = px.line(
            range_long,
            x="날짜",
            y="값",
            color="변동 지표",
            markers=True,
            title="온도·습도 변동 파생변수",
        )

        fig_range.update_layout(
            height=420,
            legend_title_text="",
        )

        st.plotly_chart(
            fig_range,
            use_container_width=True,
        )


# ============================================================
# 10. 대기오염 추세
# ============================================================

st.markdown("---")
st.subheader("🌫️ 대기오염 변화")

air_left, air_right = st.columns(2)

with air_left:
    particle_cols = [
        col
        for col in [
            "pm10",
            "pm25",
            "pm_total",
        ]
        if col in realtime_df.columns
    ]

    if particle_cols:
        particle_legend = {
            "pm10": "PM10",
            "pm25": "PM2.5",
            "pm_total": "PM10+PM2.5",
        }

        pm_long = (
            realtime_df[
                ["date"] + particle_cols
            ]
            .rename(
                columns={
                    "date": "날짜",
                    **particle_legend,
                }
            )
            .melt(
                id_vars="날짜",
                var_name="미세먼지 지표",
                value_name="농도/부하",
            )
        )

        fig_pm = px.line(
            pm_long,
            x="날짜",
            y="농도/부하",
            color="미세먼지 지표",
            markers=True,
            title="미세먼지 관련 지표",
        )

        fig_pm.update_layout(
            height=420,
            legend_title_text="",
        )

        st.plotly_chart(
            fig_pm,
            use_container_width=True,
        )


with air_right:
    gas_cols = [
        col
        for col in [
            "o3",
            "no2",
            "so2",
        ]
        if col in realtime_df.columns
    ]

    if gas_cols:
        gas_legend = {
            "o3": "O₃",
            "no2": "NO₂",
            "so2": "SO₂",
        }

        gas_long = (
            realtime_df[
                ["date"] + gas_cols
            ]
            .rename(
                columns={
                    "date": "날짜",
                    **gas_legend,
                }
            )
            .melt(
                id_vars="날짜",
                var_name="대기오염 물질",
                value_name="농도(ppm)",
            )
        )

        fig_gas = px.line(
            gas_long,
            x="날짜",
            y="농도(ppm)",
            color="대기오염 물질",
            markers=True,
            title="O₃ · NO₂ · SO₂ 변화",
        )

        fig_gas.update_layout(
            height=420,
            legend_title_text="",
        )

        st.plotly_chart(
            fig_gas,
            use_container_width=True,
        )


# ============================================================
# 11. 누적 대기오염 파생변수
# ============================================================

derived_air_cols = [
    col
    for col in [
        "pm_load_3d",
        "pm_load_7d",
        "so2_ma7",
        "no2_ma7",
        "o3_ma7",
    ]
    if col in realtime_df.columns
]

if derived_air_cols:
    st.markdown("---")
    st.subheader("🧪 누적·이동평균 대기오염 파생변수")

    air_feature_legend = {
        "pm_load_3d": "3일 미세먼지 부하",
        "pm_load_7d": "7일 미세먼지 부하",
        "so2_ma7": "SO₂ 7일 평균",
        "no2_ma7": "NO₂ 7일 평균",
        "o3_ma7": "O₃ 7일 평균",
    }

    selected_air_features = st.multiselect(
        "표시할 대기오염 파생변수",
        options=derived_air_cols,
        default=derived_air_cols,
        format_func=lambda x: air_feature_legend.get(
            x,
            x,
        ),
    )

    if selected_air_features:
        air_feature_long = (
            realtime_df[
                ["date"] + selected_air_features
            ]
            .rename(
                columns={
                    "date": "날짜",
                    **air_feature_legend,
                }
            )
            .melt(
                id_vars="날짜",
                var_name="파생변수",
                value_name="값",
            )
        )

        fig_air_feature = px.line(
            air_feature_long,
            x="날짜",
            y="값",
            color="파생변수",
            markers=True,
            title="대기오염 누적·이동평균 변화",
        )

        fig_air_feature.update_layout(
            height=450,
            legend_title_text="",
        )

        st.plotly_chart(
            fig_air_feature,
            use_container_width=True,
        )


# ============================================================
# 12. 28일 습도 지속성
# ============================================================

st.markdown("---")
st.subheader("💧 최근 28일 습도 조건 누적")

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
    humid_legend_map = {
        "rh60_days_28": "RH>60%",
        "rh70_days_28": "RH≥70%",
        "rh75_days_28": "RH≥75%",
        "rh95_days_28": "RH≥95%",
    }

    humid_long = (
        realtime_df[
            ["date"] + humid_features
        ]
        .rename(
            columns={
                "date": "날짜",
                **humid_legend_map,
            }
        )
        .melt(
            id_vars="날짜",
            var_name="습도 조건",
            value_name="최근 28일 해당 일수",
        )
    )

    fig_rh = px.line(
        humid_long,
        x="날짜",
        y="최근 28일 해당 일수",
        color="습도 조건",
        markers=True,
        title="최근 28일 고습 조건 누적일수",
    )

    fig_rh.update_layout(
        height=450,
        yaxis_range=[0, 28],
        legend_title_text="",
    )

    st.plotly_chart(
        fig_rh,
        use_container_width=True,
    )


# ============================================================
# 13. 재질 관련 환경지표
# ============================================================

st.markdown("---")
st.subheader("🏛️ 재질 관련 환경 취약 파생변수")

material_left, material_right = st.columns(2)

with material_left:
    wood_cols = [
        col
        for col in [
            "wood_mold_days_7",
            "wood_mold_days_28",
            "rh75_consecutive_days",
            "rh95_consecutive_days",
        ]
        if col in realtime_df.columns
    ]

    if wood_cols:
        wood_legend = {
            "wood_mold_days_7": "7일 목조 곰팡이 조건",
            "wood_mold_days_28": "28일 목조 곰팡이 조건",
            "rh75_consecutive_days": "RH≥75% 연속일수",
            "rh95_consecutive_days": "RH≥95% 연속일수",
        }

        wood_long = (
            realtime_df[
                ["date"] + wood_cols
            ]
            .rename(
                columns={
                    "date": "날짜",
                    **wood_legend,
                }
            )
            .melt(
                id_vars="날짜",
                var_name="목조 관련 지표",
                value_name="값",
            )
        )

        fig_wood = px.line(
            wood_long,
            x="날짜",
            y="값",
            color="목조 관련 지표",
            markers=True,
            title="목조 관련 습도·곰팡이 지표",
        )

        fig_wood.update_layout(
            height=420,
            legend_title_text="",
        )

        st.plotly_chart(
            fig_wood,
            use_container_width=True,
        )


with material_right:
    metal_cols = [
        col
        for col in [
            "rh70_days_28",
            "rh70_consecutive_days",
            "metal_so2_humidity",
        ]
        if col in realtime_df.columns
    ]

    if metal_cols:
        metal_legend = {
            "rh70_days_28": "28일 RH≥70% 일수",
            "rh70_consecutive_days": "RH≥70% 연속일수",
            "metal_so2_humidity": "고습·SO₂ 복합지표",
        }

        metal_long = (
            realtime_df[
                ["date"] + metal_cols
            ]
            .rename(
                columns={
                    "date": "날짜",
                    **metal_legend,
                }
            )
            .melt(
                id_vars="날짜",
                var_name="금속 관련 지표",
                value_name="값",
            )
        )

        fig_metal = px.line(
            metal_long,
            x="날짜",
            y="값",
            color="금속 관련 지표",
            markers=True,
            title="금속 관련 고습·SO₂ 지표",
        )

        fig_metal.update_layout(
            height=420,
            legend_title_text="",
        )

        st.plotly_chart(
            fig_metal,
            use_container_width=True,
        )


# ============================================================
# 14. 전체 파생변수 선택 시각화
# ============================================================

st.markdown("---")
st.subheader("🧮 파생변수 직접 선택 비교")

excluded_cols = {
    "date",
    "season",
    "month",
}

candidate_numeric_features = []

for col in realtime_df.columns:
    if col in excluded_cols:
        continue

    if pd.api.types.is_numeric_dtype(
        realtime_df[col]
    ):
        candidate_numeric_features.append(
            col
        )

default_features = [
    col
    for col in [
        "rainfall_7d",
        "rh75_days_28",
        "wood_mold_days_28",
        "metal_so2_humidity",
        "pm_load_7d",
    ]
    if col in candidate_numeric_features
]

selected_features = st.multiselect(
    "비교할 Feature를 선택하세요",
    options=candidate_numeric_features,
    default=default_features,
    format_func=lambda x: COLUMN_KR.get(
        x,
        x,
    ),
)

if selected_features:
    selected_long = (
        realtime_df[
            ["date"] + selected_features
        ]
        .rename(
            columns={
                "date": "날짜",
                **{
                    col: COLUMN_KR.get(
                        col,
                        col,
                    )
                    for col in selected_features
                },
            }
        )
        .melt(
            id_vars="날짜",
            var_name="Feature",
            value_name="값",
        )
    )

    fig_selected = px.line(
        selected_long,
        x="날짜",
        y="값",
        color="Feature",
        markers=True,
        title="선택한 환경 Feature 변화",
    )

    fig_selected.update_layout(
        height=500,
        legend_title_text="",
    )

    st.plotly_chart(
        fig_selected,
        use_container_width=True,
    )

    st.caption(
        "※ 서로 단위와 값 범위가 다른 Feature를 한 그래프에 표시하므로 "
        "절대 크기보다 시간에 따른 변화 방향을 비교하는 용도로 활용하세요."
    )


# ============================================================
# 15. 최근 40일 기본 환경 데이터 표
# ============================================================

st.markdown("---")
st.subheader("📋 최근 40일 기상·대기환경 원자료")

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
    realtime_df[
        basic_cols
    ]
    .sort_values(
        "date",
        ascending=False,
    )
    .copy()
    .rename(
        columns=COLUMN_KR
    )
)

st.dataframe(
    basic_view,
    use_container_width=True,
    height=520,
    hide_index=True,
)


# ============================================================
# 16. 최근 40일 파생변수 표
# ============================================================

st.markdown("---")
st.subheader("🧪 최근 40일 예측용 파생변수")

raw_cols = {
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
}

derived_cols = [
    col
    for col in realtime_df.columns
    if col not in raw_cols
]

derived_view_cols = (
    ["date"]
    + derived_cols
)

derived_view = (
    realtime_df[
        derived_view_cols
    ]
    .sort_values(
        "date",
        ascending=False,
    )
    .copy()
    .rename(
        columns=COLUMN_KR
    )
)

st.dataframe(
    derived_view,
    use_container_width=True,
    height=600,
    hide_index=True,
)


# ============================================================
# 17. 데이터 품질
# ============================================================

st.markdown("---")

with st.expander(
    "🔍 수집·전처리 품질 확인",
    expanded=False,
):

    if isinstance(
        quality,
        dict,
    ):
        q1, q2, q3, q4 = st.columns(
            4
        )

        q1.metric(
            "기상 API 일수",
            f"{quality.get('weather_days', '-')}일",
        )

        q2.metric(
            "대기 API 일수",
            f"{quality.get('air_days', '-')}일",
        )

        q3.metric(
            "대기 결측 발생일",
            f"{quality.get('air_missing_before_ffill', '-')}일",
        )

        q4.metric(
            "사용 가능 일수",
            f"{quality.get('usable_days', len(realtime_df))}일",
        )

        st.json(
            quality
        )

    else:
        st.info(
            "현재 세션에는 별도의 데이터 품질 요약 정보가 없습니다."
        )


# ============================================================
# 18. CSV 다운로드
# ============================================================

st.markdown("---")

csv_bytes = (
    realtime_df
    .to_csv(
        index=False
    )
    .encode(
        "utf-8-sig"
    )
)

st.download_button(
    "📥 최근 40일 환경·파생변수 전체 CSV 다운로드",
    data=csv_bytes,
    file_name=(
        "영천_최근40일_환경데이터_파생변수_"
        f"{loaded_target_date:%Y%m%d}.csv"
    ),
    mime="text/csv",
    use_container_width=True,
)

st.caption(
    "※ 이 페이지의 파생변수는 문화유산 취약도 예측 모델 입력에 사용된 "
    "환경 Feature를 시각화한 것입니다."
)

st.caption(
    "선화여고 · 영천 헤리티지 AI 탐구단"
)
