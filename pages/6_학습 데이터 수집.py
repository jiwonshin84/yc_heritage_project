from datetime import datetime
import os
import time
from zoneinfo import ZoneInfo

from github import Github, GithubException
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
    page_title="10개년 기상·대기오염 데이터 수집 및 분석",
    page_icon="📊",
    layout="wide",
)

current_year = datetime.now().year
end_year = current_year - 1
start_year = end_year - 9

file_name = f"[{start_year}_{end_year}] yeongcheon.csv"

# 공공데이터포털 공용 인증키
PUBLIC_SERVICE_KEY = (
    "feb2bfabd299d5d05e89c7aec49ba7e706112603e76549a92e868bd86ec60323"
)

# 기상청 ASOS 설정
ASOS_URL = "http://apis.data.go.kr/1360000/AsosDalyInfoService/getWthrDataList"
STN_ID = "281"  # 영천 관측소

st.title(
    f"📊 {start_year}~{end_year}년 ({end_year-start_year+1}개년) 기상·대기오염 데이터 수집 및 분석"
)
st.markdown(
    "<p style='color:#4b5563;'>선화여고 - 영천 헤리티지 AI 탐구단 학습용 데이터 파이프라인</p>",
    unsafe_allow_html=True,
)


# ------------------------------------------------------------
# 2. 데이터 수집 및 가공 함수
# ------------------------------------------------------------
def fetch_asos_year(year):
  start_dt = f"{year}0101"
  end_dt = f"{year}1231"
  params = {
      "serviceKey": PUBLIC_SERVICE_KEY,
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
    items = (
        result.get("response", {}).get("body", {}).get("items", {}).get("item", [])
    )
    if items:
      return pd.DataFrame(items)
  except Exception:
    pass
  return pd.DataFrame()


def fetch_airkorea_year(year, weather_df_dates):
  """에어코리아 API의 과거 데이터 제공 한계 및 오류를 방어하기 위한 함수

  - 에어코리아 API는 실시간 위주이므로 10개년 과거 데이터 직접 조회가 불가능합니다.
  - API 호출 실패 시 ASOS 날짜 인덱스에 맞춰 대기오염 기본 구조(NaN 또는 추정치)를 생성하여
    데이터 병합 오류 및 앱 크래시를 방지합니다.
  """
  air_url = (
      "http://apis.data.go.kr/B552584/ArpltnInforInqireSvc/getMsrstnAcctoRltmMesureDnsty"
  )
  params = {
      "serviceKey": PUBLIC_SERVICE_KEY,
      "returnType": "JSON",
      "numOfRows": "1",
      "pageNo": "1",
      "stationName": "영천",
      "dataTerm": "DAILY",
      "ver": "1.0",
  }

  success = False
  try:
    response = requests.get(air_url, params=params, timeout=5)
    if response.status_code == 200:
      res_json = response.json()
      items = res_json.get("response", {}).get("body", {}).get("items", [])
      if items:
        success = True
  except Exception:
    pass

  # 만약 API로 과거 연도 데이터를 가져오지 못할 경우 (에어코리아 구조상 과거 데이터 미지원)
  # 해당 연도의 날짜 포맷에 맞추어 빈 대기오염 프레임을 생성하되 경고를 줄이고 파이프라인 유지
  df_air = pd.DataFrame({"date": weather_df_dates})
  df_air["pm15"] = None
  for col in ["pm10", "pm25", "o3", "no2", "co", "so2"]:
    df_air[col] = None

  return df_air


def get_season(month):
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
  all_weather_years = []

  for i, year in enumerate(range(start_year, end_year + 1)):
    status_container.update(
        label=f"📡 [{i+1}/{total_years}] {year}년 기상(ASOS) 데이터 수집 중...",
        state="running",
    )

    # 1. 기상청 ASOS 기상 데이터 수집
    df_year = fetch_asos_year(year)
    if not df_year.empty:
      all_weather_years.append(df_year)

    progress_bar.progress((i + 1) / (total_years + 1))
    time.sleep(0.05)

  status_container.update(
      label="🔄 수집 데이터 통합 및 정제(전처리) 중...", state="running"
  )

  if all_weather_years:
    weather_raw = pd.concat(all_weather_years, ignore_index=True)
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
        weather.dropna(subset=["date"])
        .sort_values("date")
        .reset_index(drop=True)
    )
  else:
    weather = pd.DataFrame(
        columns=[
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
    )

  # 대기오염 데이터 프레임 구조 맞추기 (에어코리아 API는 10개년 과거 데이터 미지원 하므로 날짜 인덱스 동기화)
  air_frames = []
  for year in range(start_year, end_year + 1):
    year_dates = weather[weather["date"].dt.year == year]["date"]
    if not year_dates.empty:
      df_air_y = fetch_airkorea_year(year, year_dates)
      air_frames.append(df_air_y)

  if air_frames:
    air = pd.concat(air_frames, ignore_index=True)
  else:
    air = pd.DataFrame(
        columns=["date", "pm10", "pm25", "o3", "no2", "co", "so2"]
    )

  df = pd.merge(weather, air, on="date", how="left")

  df["month"] = df["date"].dt.month
  df["year"] = df["date"].dt.year
  df["season"] = df["month"].apply(get_season)

  # GitHub 업로드 로직
  status_container.update(
      label="☁️ GitHub 저장소로 자동 업로드 중...", state="running"
  )
  try:
    token = st.secrets["GITHUB_TOKEN"]
    repo_name = st.secrets["GITHUB_REPO"]

    g = Github(token)
    repo = g.get_repo(repo_name)

    git_file_path = f"data/processed/{file_name}"
    file_content = df.to_csv(index=False, encoding="utf-8-sig")
    commit_message = (
        f"chore: 웹앱을 통한 {file_name} 기상·대기오염 API 자동 업데이트"
    )

    try:
      contents = repo.get_contents(git_file_path)
      repo.update_file(
          path=git_file_path,
          message=commit_message,
          content=file_content,
          sha=contents.sha,
          branch="main",
      )
      st.toast(
          f"☁️ GitHub [{git_file_path}] 파일이 성공적으로 업데이트되었습니다!",
          icon="🚀",
      )
    except Exception:
      repo.create_file(
          path=git_file_path,
          message=commit_message,
          content=file_content,
          branch="main",
      )
      st.toast(
          f"☁️ GitHub [{git_file_path}] 파일이 새로 생성(업로드)되었습니다!",
          icon="🚀",
      )
  except Exception as e:
    st.warning(f"⚠️ GitHub API 업로드 생략 또는 오류: {e}")

  progress_bar.progress(1.0)
  status_container.update(
      label="✅ 데이터 수집 및 정제 완료!", state="complete", expanded=False
  )

  return df


# ------------------------------------------------------------
# 3. 데이터 로드 및 UI 실행
# ------------------------------------------------------------
if "df_data" not in st.session_state:
  st.session_state.df_data = None

col_ui1, col_ui2 = st.columns([1, 3], vertical_alignment="center")

with col_ui1:
  collect_clicked = st.button("🚀 데이터 수집 시작", use_container_width=True)

with col_ui2:
  if st.session_state.df_data is None:
    st.info("💡 버튼을 누르면 기상청 ASOS 데이터 및 분석 파이프라인을 실행합니다.")
  else:
    st.success("✅ 학습용 데이터셋이 성공적으로 준비되었습니다!")

if collect_clicked:
  status_box = st.status("데이터 수집 준비 중...", expanded=True)
  prog_bar = st.progress(0)
  st.session_state.df_data = collect_and_process_data(status_box, prog_bar)
  st.rerun()

df = st.session_state.df_data

if df is not None and not df.empty:
  csv_bytes = df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")

  st.download_button(
      label=f"📥 {file_name} 파일 다운로드",
      data=csv_bytes,
      file_name=file_name,
      mime="text/csv",
      type="primary",
  )

  st.markdown("---")
  with st.expander("🔍 데이터 전처리 및 품질 리포트 확인하기", expanded=False):
    col_r1, col_r2, col_r3, col_r4 = st.columns(4)
    col_r1.metric("총 수집 행(Row) 수", f"{len(df):,} 개")
    col_r2.metric("날짜 파싱 오류", f"{df['date'].isna().sum()} 건")
    col_r3.metric("강수량 결측치 보정", "0.0 처리 완료")
    col_r4.metric("대기오염 API 상태", "과거 연도 미지원 (구조적 한계)")

  st.markdown("---")
  st.subheader("📌 수집 데이터 주요 요약 지표")

  kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
  kpi1.metric("총 관측 일수", f"{len(df):,} 일")
  kpi2.metric("평균 기온", f"{df['temp_avg'].mean():.1f} °C")
  kpi3.metric("평균 습도", f"{df['humidity'].mean():.1f} %")
  kpi4.metric("평균 PM10", "데이터 없음")
  kpi5.metric("평균 오존($O_3$)", "데이터 없음")

  # 차트 시각화 영역
  st.markdown("---")
  st.subheader("📈 계절별 기상 및 대기오염 성분 종합 분석")

  row1_col1, row1_col2 = st.columns(2)
  with row1_col1:
    df_season_avg = (
        df.groupby("season")
        .agg({"temp_avg": "mean", "humidity": "mean"})
        .reset_index()
    )
    fig_season = make_subplots(specs=[[{"secondary_y": True}]])
    fig_season.add_trace(
        go.Bar(
            x=df_season_avg["season"],
            y=df_season_avg["temp_avg"],
            name="평균 기온 (°C)",
            marker_color="#FF6B6B",
        ),
        secondary_y=False,
    )
    fig_season.add_trace(
        go.Scatter(
            x=df_season_avg["season"],
            y=df_season_avg["humidity"],
            name="평균 습도 (%)",
            line=dict(color="#1C7ED6", width=3, dash="dash"),
        ),
        secondary_y=True,
    )
    fig_season.update_layout(
        title="🌸☀️🍁❄️ 계절별 평균 기온 및 습도 분포", height=420
    )
    st.plotly_chart(fig_season, use_container_width=True)

  with row1_col2:
    df_yearly_season = (
        df.groupby(["year", "season"])["temp_avg"].mean().reset_index()
    )
    fig_season_trend = px.line(
        df_yearly_season,
        x="year",
        y="temp_avg",
        color="season",
        markers=True,
        title="📅 연도별 계절 평균 기온 추이",
    )
    fig_season_trend.update_layout(
        height=420, xaxis=dict(type="category")
    )
    st.plotly_chart(fig_season_trend, use_container_width=True)
