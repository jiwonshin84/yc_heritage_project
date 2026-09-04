from datetime import datetime
from github import Github, GithubException
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
# 2. 데이터 수집 및 가공 함수 (진행 상태 콜백 지원)
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


def collect_and_process_data(status_container, progress_bar):
    total_years = end_year - start_year + 1
    all_years = []
    
    for i, year in enumerate(range(start_year, end_year + 1)):
        status_container.update(label=f"📡 [{i+1}/{total_years}] {year}년 기상 공공 API 데이터 수집 중...", state="running")
        df_year = fetch_asos_year(year)
        if not df_year.empty:
            all_years.append(df_year)
        progress_bar.progress((i + 1) / (total_years + 1))
        time.sleep(0.05)

    status_container.update(label="🔄 수집 데이터 통합 및 정제(전처리) 중...", state="running")
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

    # 강수량 결측치는 0mm로 채우기
    weather["rainfall"] = weather["rainfall"].fillna(0)
    weather = (
        weather.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    )

    # 미세먼지 외부 수집 데이터 병합
    status_container.update(label="🔗 미세먼지 학습 데이터셋 외부 링크 병합 중...", state="running")
    air_url = "https://docs.google.com/spreadsheets/d/1fBEnheVOP-23Hmv_5ZJZVy6m9VmNkpVd2XutOdmlYc8/export?format=csv&gid=700055413"
    air = pd.read_csv(air_url)
    air["date"] = pd.to_datetime(air["date"], errors="coerce")

    df = pd.merge(weather, air, on="date", how="left")

    # 파생 변수 추가 (월, 연도, 계절)
    df["month"] = df["date"].dt.month
    df["year"] = df["date"].dt.year
    df["season"] = df["month"].apply(get_season)

    # ------------------------------------------------------------
    # 🚀 GitHub API를 이용해 원격 저장소에 파일 업로드/업데이트
    # ------------------------------------------------------------
    status_container.update(label="☁️ GitHub 저장소로 자동 업로드 중...", state="running")
    try:
        token = st.secrets["GITHUB_TOKEN"]
        repo_name = st.secrets["GITHUB_REPO"]  # 형식: "사용자명/저장소명"
        
        g = Github(token)
        repo = g.get_repo(repo_name)
        
        git_file_path = f"data/processed/{file_name}"
        file_content = df.to_csv(index=False, encoding="utf-8-sig")
        commit_message = f"chore: 웹앱을 통한 {file_name} 자동 데이터 업데이트"
        
        try:
            contents = repo.get_contents(git_file_path)
            repo.update_file(
                path=git_file_path,
                message=commit_message,
                content=file_content,
                sha=contents.sha,
                branch="main"
            )
            st.toast(f"☁️ GitHub [{git_file_path}] 파일이 성공적으로 업데이트되었습니다!", icon="🚀")
        except Exception:
            repo.create_file(
                path=git_file_path,
                message=commit_message,
                content=file_content,
                branch="main"
            )
            st.toast(f"☁️ GitHub [{git_file_path}] 파일이 새로 생성(업로드)되었습니다!", icon="🚀")
            
    except Exception as e:
        st.warning(f"⚠️ GitHub API 업로드 중 오류 발생: {e}")
    # ------------------------------------------------------------

    progress_bar.progress(1.0)
    status_container.update(label="✅ 학습 데이터 구축 및 전처리 완료!", state="complete", expanded=False)

    return df


# ------------------------------------------------------------
# 3. 데이터 로드 / 수집 실행 (세션 스테이트 활용)
# ------------------------------------------------------------
if "df_data" not in st.session_state:
    st.session_state.df_data = None

# 버튼과 안내 박스를 하나의 행(Columns)으로 배치
col_ui1, col_ui2 = st.columns([1, 3], vertical_alignment="center")

with col_ui1:
    collect_clicked = st.button("🚀 데이터 수집 시작", use_container_width=True)

with col_ui2:
    if st.session_state.df_data is None:
        st.info("💡 버튼을 누르면 10개년 기상·미세먼지 학습 데이터 수집 및 전처리가 시작됩니다.")
    else:
        st.success("✅ 학습용 데이터셋이 성공적으로 준비되었습니다!")

if collect_clicked:
    status_box = st.status("데이터 수집 준비 중...", expanded=True)
    prog_bar = st.progress(0)
    st.session_state.df_data = collect_and_process_data(status_box, prog_bar)
    st.rerun()

df = st.session_state.df_data

# 데이터 수집 후 화면 렌더링
if df is not None:
    csv_bytes = df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")

    st.download_button(
        label=f"📥 {file_name} 파일 다운로드",
        data=csv_bytes,
        file_name=file_name,
        mime="text/csv",
        type="primary",
    )

    # ------------------------------------------------------------
    # 3-1. 데이터 전처리 및 품질 요약 (Data Cleansing Report)
    # ------------------------------------------------------------
    st.markdown("---")
    with st.expander("🔍 데이터 전처리 및 품질 리포트 확인하기", expanded=False):
        col_r1, col_r2, col_r3, col_r4 = st.columns(4)
        col_r1.metric("총 수집 행(Row) 수", f"{len(df):,} 개")
        col_r2.metric("날짜 파싱 오류", f"{df['date'].isna().sum()} 건")
        col_r3.metric("강수량 결측치 보정", "0.0 처리 완료")
        col_r4.metric("미세먼지 병합율", f"{(df['pm10'].notna().mean() * 100):.1f}%")
        st.info(
            "💡 **전처리 노트**: 기상청 ASOS 일별 데이터 수집 후 날짜(`date`) 표준화, 결측치 수치 변환, "
            "강수량(`rainfall`) 공백 0 처리 과정을 거쳤으며, 외부 대기오염 시트 데이터와 기준일자(Left Join)로 결합하여 머신러닝 학습셋을 완성했습니다."
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
    # 5. 그리드 배치 차트 시각화 (모두 계절 중심 분석)
    # ------------------------------------------------------------
    st.markdown("---")
    st.subheader("📈 계절별 기상 및 미세먼지 종합 분석")

    row1_col1, row1_col2 = st.columns(2)

    with row1_col1:
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
            title="🌸☀️🍁❄️ 계절별 평균 기온 및 습도 분포",
            xaxis_title="계절",
            height=420,
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        fig_season.update_yaxes(title_text="기온 (°C)", secondary_y=False)
        fig_season.update_yaxes(title_text="습도 (%)", secondary_y=True, range=[0, 100])
        st.plotly_chart(fig_season, use_container_width=True)

    with row1_col2:
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

    row2_col1, row2_col2 = st.columns(2)

    with row2_col1:
        df_season_air = df.groupby("season")[["pm10", "pm25"]].mean().reset_index()

        fig_air_season = px.bar(
            df_season_air,
            x="season",
            y=["pm10", "pm25"],
            barmode="group",
            labels={"value": "농도 (㎛/㎥)", "season": "계절", "variable": "구분"},
            color_discrete_map={"pm10": "#FFAA00", "pm25": "#FF4444"},
            title="🌫️ 계절별 미세먼지 및 초미세먼지 평균 변화",
            text_auto=".1f",
        )
        fig_air_season.update_layout(
            height=420,
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_air_season, use_container_width=True)

    with row2_col2:
        df_season_rain = (
            df.groupby("season")
            .agg({"rainfall": "sum", "solar_radiation": "mean"})
            .reset_index()
        )

        fig_rain_season = make_subplots(specs=[[{"secondary_y": True}]])
        fig_rain_season.add_trace(
            go.Bar(
                x=df_season_rain["season"],
                y=df_season_rain["rainfall"],
                name="총 강수량 (mm)",
                marker_color="#29B6F6",
                text=df_season_rain["rainfall"].round(1),
                textposition="auto",
            ),
            secondary_y=False,
        )
        fig_rain_season.add_trace(
            go.Scatter(
                x=df_season_rain["season"],
                y=df_season_rain["solar_radiation"],
                name="평균 일사량 (MJ/㎡)",
                line=dict(color="#FFA726", width=3),
                mode="lines+markers+text",
                text=df_season_rain["solar_radiation"].round(2),
                textposition="top center",
            ),
            secondary_y=True,
        )
        fig_rain_season.update_layout(
            title="🌧️☀️ 계절별 기상 통계 (총 강수량 vs 평균 일사량)",
            xaxis_title="계절",
            height=420,
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        fig_rain_season.update_yaxes(title_text="강수량 (mm)", secondary_y=False)
        fig_rain_season.update_yaxes(title_text="일사량 (MJ/㎡)", secondary_y=True)
        st.plotly_chart(fig_rain_season, use_container_width=True)

    # ------------------------------------------------------------
    # 6. 수집 데이터 미리보기
    # ------------------------------------------------------------
    st.markdown("---")
    with st.expander("📋 수집 데이터 원본 상세보기 (최근 100건)"):
        st.dataframe(
            df.sort_values("date", ascending=False).head(100),
            use_container_width=True,
        )
