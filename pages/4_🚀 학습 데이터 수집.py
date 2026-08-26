from datetime import datetime
import time
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import streamlit as st

# ------------------------------------------------------------
# 1. 페이지 설정 및 동적 연도 계산
# ------------------------------------------------------------
st.set_page_config(
    page_title="10개년 기상·미세먼지 데이터 수집 및 분석", layout="wide"
)

current_year = datetime.now().year
end_year = current_year - 1
start_year = end_year - 9

file_name = f"[{start_year}_{end_year}] yeongcheon.csv"

ASOS_SERVICE_KEY = (
    "feb2bfabd299d5d05e89c7aec49ba7e706112603e76549a92e868bd86ec60323"
)
ASOS_URL = "http://apis.data.go.kr/1360000/AsosDalyInfoService/getWthrDataList"
STN_ID = "281"  # 영천 관측소

st.title(
    f"📊 {start_year}~{end_year}년 ({end_year-start_year+1}개년) 데이터 수집 및 시각화"
)


# ------------------------------------------------------------
# 2. 데이터 수집 및 가공 함수
# ------------------------------------------------------------
def fetch_asos_year(year):
    start_dt = f"{year}0101"
    end_dt = f"{year}1231"
    params = {
        "serviceKey": ASOS_SERVICE_KEY,
        "numOfRows": "400",
        "pageNo": "1",
        "dataType": "JSON",
        "dataCd": "ASOS",
        "dateCd": "DAY",
        "startDt": start_dt,
        "endDt": end_dt,
        "stnIds": STN_ID,
    }
    try:
        response = requests.get(ASOS_URL, params=params, timeout=30)
        result = response.json()
        items = result["response"]["body"]["items"]["item"]
        return pd.DataFrame(items)
    except Exception as e:
        st.error(f"{year}년 수집 실패: {e}")
        return pd.DataFrame()


def get_season(month):
    """월 정보를 바탕으로 계절 파악"""
    if month in [3, 4, 5]:
        return "1. 봄 (3~5월)"
    elif month in [6, 7, 8]:
        return "2. 여름 (6~8월)"
    elif month in [9, 10, 11]:
        return "3. 가을 (9~11월)"
    else:
        return "4. 겨울 (12~2월)"


def collect_and_process_data():
    all_years = []
    for year in range(start_year, end_year + 1):
        df_year = fetch_asos_year(year)
        all_years.append(df_year)
        time.sleep(0.1)

    weather_raw = pd.concat(all_years, ignore_index=True)

    weather = weather_raw[[
        "tm",
        "avgTa",
        "maxTa",
        "minTa",
        "avgRhm",
        "sumRn",
        "avgWs",
        "sumSsHr",
        "avgTs",
    ]].copy()
    weather.columns = [
        "date",
        "temp_avg",
        "temp_max",
        "temp_min",
        "humidity",
        "rainfall",
        "wind_speed",
        "solar_radiation",
        "ground_temp",
    ]

    weather["date"] = pd.to_datetime(weather["date"], errors="coerce")
    numeric_cols = [
        "temp_avg",
        "temp_max",
        "temp_min",
        "humidity",
        "rainfall",
        "wind_speed",
        "solar_radiation",
        "ground_temp",
    ]
    for col in numeric_cols:
        weather[col] = pd.to_numeric(weather[col], errors="coerce")

    weather["rainfall"] = weather["rainfall"].fillna(0)
    weather = (
        weather.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    )

    # 미세먼지 외부 수집 데이터 병합
    air_url = "https://docs.google.com/spreadsheets/d/1fBEnheVOP-23Hmv_5ZJZVy6m9VmNkpVd2XutOdmlYc8/export?format=csv&gid=700055413"
    air = pd.read_csv(air_url)
    air["date"] = pd.to_datetime(air["date"], errors="coerce")

    df = pd.merge(weather, air, on="date", how="left")

    # 파생 변수 추가 (월, 연도, 계절)
    df["month"] = df["date"].dt.month
    df["year"] = df["date"].dt.year
    df["season"] = df["month"].apply(get_season)

    return df


# ------------------------------------------------------------
# 3. 데이터 로드 / 수집 실행 (세션 스테이트 활용)
# ------------------------------------------------------------
if "df_data" not in st.session_state:
    st.session_state.df_data = None

col_btn1, col_btn2 = st.columns([1, 4])
with col_btn1:
    if st.button("🚀 데이터 수집 시작"):
        with st.spinner(f"{start_year}~{end_year}년 데이터를 수집 중입니다..."):
            st.session_state.df_data = collect_and_process_data()

df = st.session_state.df_data

# 데이터 수집 후 화면 렌더링
if df is not None:
    csv_bytes = df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")

    st.success("✅ 데이터 수집이 완료되었습니다!")
    st.download_button(
        label=f"📥 {file_name} 파일 다운로드",
        data=csv_bytes,
        file_name=file_name,
        mime="text/csv",
        type="primary",
    )

    # ------------------------------------------------------------
    # 4. 주요 데이터 지표 (KPI Metrics)
    # ------------------------------------------------------------
    st.markdown("---")
    st.subheader("📌 수집 데이터 주요 요약 지표")

    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    kpi1.metric("총 관측 일수", f"{len(df):,} 일")
    kpi2.metric("평균 기온", f"{df['temp_avg'].mean():.1f} °C")
    kpi3.metric("평균 습도", f"{df['humidity'].mean():.1f} %")
    kpi4.metric("평균 PM10", f"{df['pm10'].mean():.1f} ㎛/㎥")
    kpi5.metric("평균 PM2.5", f"{df['pm25'].mean():.1f} ㎛/㎥")

    # ------------------------------------------------------------
    # 5. 그리드 배치 차트 시각화
    # ------------------------------------------------------------
    st.markdown("---")
    st.subheader("📈 종합 기상 및 미세먼지 데이터 분석")

    # [Row 1] 계절별 기온/습도 현황 & 연도별 계절 기온 변화 추이
    row1_col1, row1_col2 = st.columns(2)

    with row1_col1:
        # 1-1. 계절별 평균 기온 및 습도 비교
        df_season_avg = df.groupby("season").agg({"temp_avg": "mean", "humidity": "mean"}).reset_index()

        fig_season = make_subplots(specs=[[{"secondary_y": True}]])
        fig_season.add_trace(
            go.Bar(
                x=df_season_avg["season"],
                y=df_season_avg["temp_avg"],
                name="평균 기온 (°C)",
                marker_color="#FF6B6B",
                text=df_season_avg["temp_avg"].round(1),
                textposition="auto",
            ),
            secondary_y=False,
        )
        fig_season.add_trace(
            go.Scatter(
                x=df_season_avg["season"],
                y=df_season_avg["humidity"],
                name="평균 습도 (%)",
                line=dict(color="#1C7ED6", width=3, dash="dash"),
                mode="lines+markers+text",
                text=df_season_avg["humidity"].round(1).astype(str) + "%",
                textposition="top center",
            ),
            secondary_y=True,
        )
        fig_season.update_layout(
            title="☀️계절별 평균 기온 및 습도 분포 🌸☀️🍁❄️",
            xaxis_title="계절",
            height=420,
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        fig_season.update_yaxes(title_text="기온 (°C)", secondary_y=False)
        fig_season.update_yaxes(title_text="습도 (%)", secondary_y=True, range=[0, 100])
        st.plotly_chart(fig_season, use_container_width=True)

    with row1_col2:
        # 1-2. 연도별 계절 기온 변화 추이 (라인 차트)
        df_yearly_season = df.groupby(["year", "season"])["temp_avg"].mean().reset_index()

        fig_season_trend = px.line(
            df_yearly_season,
            x="year",
            y="temp_avg",
            color="season",
            markers=True,
            title="📅 연도별 계절 평균 기온 추이",
            labels={"temp_avg": "평균 기온 (°C)", "year": "연도", "season": "계절"},
            color_discrete_map={
                "1. 봄 (3~5월)": "#51CF66",
                "2. 여름 (6~8월)": "#FF6B6B",
                "3. 가을 (9~11월)": "#FCC419",
                "4. 겨울 (12~2월)": "#339AF0",
            },
        )
        fig_season_trend.update_layout(
            height=420,
            hovermode="x unified",
            xaxis=dict(type="category"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_season_trend, use_container_width=True)

    # [Row 2] 계절별/연도별 미세먼지 동향 & 계절구분 월별 강수량/일사량 분석
    row2_col1, row2_col2 = st.columns(2)

    with row2_col1:
        # 2-1. 연도별 계절별 미세먼지 농도 (계절 Tab 추가)
        st.markdown("##### 🌫️ 계절별·연도별 미세먼지 평균 변화")
        seasons = ["1. 봄 (3~5월)", "2. 여름 (6~8월)", "3. 가을 (9~11월)", "4. 겨울 (12~2월)"]
        tab_spring, tab_summer, tab_autumn, tab_winter = st.tabs(["🌸 봄", "☀️ 여름", "🍁 가을", "❄️ 겨울"])
        
        season_tabs = {
            "1. 봄 (3~5월)": tab_spring,
            "2. 여름 (6~8월)": tab_summer,
            "3. 가을 (9~11월)": tab_autumn,
            "4. 겨울 (12~2월)": tab_winter,
        }

        df_air_season_yearly = df.groupby(["season", "year"])[["pm10", "pm25"]].mean().reset_index()

        for season_name, tab in season_tabs.items():
            with tab:
                df_sub = df_air_season_yearly[df_air_season_yearly["season"] == season_name]
                fig_air_season = px.bar(
                    df_sub,
                    x="year",
                    y=["pm10", "pm25"],
                    barmode="group",
                    labels={"value": "농도 (㎛/㎥)", "year": "연도", "variable": "구분"},
                    color_discrete_map={"pm10": "#FFAA00", "pm25": "#FF4444"},
                    title=f"{season_name} 연도별 미세먼지 농도",
                )
                fig_air_season.update_layout(
                    height=350,
                    xaxis=dict(type="category"),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    margin=dict(l=10, r=10, t=40, b=10),
                )
                st.plotly_chart(fig_air_season, use_container_width=True)

    with row2_col2:
        # 2-2. 10년간 월별 통계 (계절 그룹화 적용)
        df_month_agg = (
            df.groupby(["season", "month"])
            .agg({"rainfall": "sum", "solar_radiation": "mean"})
            .reset_index()
            .sort_values("month")
        )

        # 월별 계절 표시형 라벨 생성 (예: "3월 (봄)")
        df_month_agg["month_label"] = df_month_agg.apply(
            lambda r: f"{r['month']}월\n({r['season'].split('.')[1].split('(')[0].strip()})", axis=1
        )

        fig_rain = make_subplots(specs=[[{"secondary_y": True}]])
        fig_rain.add_trace(
            go.Bar(
                x=df_month_agg["month_label"],
                y=df_month_agg["rainfall"],
                name="총 강수량 (mm)",
                marker_color="#29B6F6",
            ),
            secondary_y=False,
        )
        fig_rain.add_trace(
            go.Scatter(
                x=df_month_agg["month_label"],
                y=df_month_agg["solar_radiation"],
                name="평균 일사량 (MJ/㎡)",
                line=dict(color="#FFA726", width=3),
                mode="lines+markers",
            ),
            secondary_y=True,
        )
        fig_rain.update_layout(
            title="🌧️☀️ 계절/월별 통계 (10년 누적 강수량 vs 평균 일사량)",
            xaxis_title="월 (계절구분)",
            height=420,
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        fig_rain.update_yaxes(title_text="강수량 (mm)", secondary_y=False)
        fig_rain.update_yaxes(title_text="일사량 (MJ/㎡)", secondary_y=True)
        st.plotly_chart(fig_rain, use_container_width=True)

    # ------------------------------------------------------------
    # 6. 수집 데이터 미리보기
    # ------------------------------------------------------------
    st.markdown("---")
    with st.expander("📋 수집 데이터 원본 상세보기 (최근 100건)"):
        st.dataframe(
            df.sort_values("date", ascending=False).head(100),
            use_container_width=True,
        )
else:
    st.info(
        "상단의 '🚀 데이터 수집 시작' 버튼을 누르면 최근 10개년 데이터 수집과 함께 2x2 그리드 차트 및 CSV 다운로드 버튼이 제공됩니다."
    )
