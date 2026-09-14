from __future__ import annotations

from datetime import datetime
from pathlib import Path
import time

from github import Github
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import streamlit as st

from utils.feature_engineering import create_environment_features


# ============================================================
# 1. 페이지 설정
# ============================================================

st.set_page_config(
    page_title="2019~2025 영천 학습 데이터 수집 및 분석",
    page_icon="🚀",
    layout="wide",
)

# ------------------------------------------------------------
# 수정중(5) 모델의 학습/테스트 구조와 맞추기 위해
# 학습 데이터 수집기간은 2019~2025로 고정
#
# 2019~2024 : 모델 학습 및 Expanding-Window 검증
# 2025      : Final Test Set
# ------------------------------------------------------------

START_YEAR = 2019
END_YEAR = 2025

RAW_FILE_NAME = f"[{START_YEAR}_{END_YEAR}] yeongcheon.csv"
FEATURE_FILE_NAME = f"[{START_YEAR}_{END_YEAR}] yeongcheon_features.csv"

DATA_DIR = Path("data/processed")
DATA_DIR.mkdir(parents=True, exist_ok=True)

RAW_SAVE_PATH = DATA_DIR / RAW_FILE_NAME
FEATURE_SAVE_PATH = DATA_DIR / FEATURE_FILE_NAME


# ============================================================
# 2. API 설정
# ============================================================

ASOS_URL = (
    "http://apis.data.go.kr/"
    "1360000/AsosDalyInfoService/getWthrDataList"
)

STN_ID = "281"  # 영천 관측소

# ------------------------------------------------------------
# API KEY는 코드에 직접 넣지 않고 Streamlit Secrets 사용
# .streamlit/secrets.toml
#
# ASOS_SERVICE_KEY = "..."
# GITHUB_TOKEN = "..."
# GITHUB_REPO = "사용자명/저장소명"
# ------------------------------------------------------------

ASOS_SERVICE_KEY = st.secrets.get("ASOS_SERVICE_KEY", "")

# 학습용 대기환경 데이터
AIR_TRAIN_URL = (
    "https://docs.google.com/spreadsheets/d/"
    "1fBEnheVOP-23Hmv_5ZJZVy6m9VmNkpVd2XutOdmlYc8/"
    "export?format=csv&gid=700055413"
)


# ============================================================
# 3. 화면 제목
# ============================================================

st.title(
    f"📊 {START_YEAR}~{END_YEAR}년 "
    f"영천 학습 데이터 수집 및 시각화"
)

st.caption(
    "기상청 ASOS + 대기환경 자료를 결합하고, "
    "모델 학습과 실시간 예측에 공통으로 사용하는 "
    "파생변수를 생성합니다."
)

st.info(
    "📌 학습 구조: 2019~2024년은 모델 학습·검증, "
    "2025년은 최종 Test Set으로 사용합니다."
)


# ============================================================
# 4. 기상청 ASOS 연도별 수집 함수
# ============================================================

def fetch_asos_year(year: int) -> pd.DataFrame:
    """
    영천 ASOS 일별 자료를 1년 단위로 수집한다.
    """

    if not ASOS_SERVICE_KEY:
        raise ValueError(
            "Streamlit Secrets에 ASOS_SERVICE_KEY가 없습니다."
        )

    params = {
        "serviceKey": ASOS_SERVICE_KEY,
        "numOfRows": "400",
        "pageNo": "1",
        "dataType": "JSON",
        "dataCd": "ASOS",
        "dateCd": "DAY",
        "startDt": f"{year}0101",
        "endDt": f"{year}1231",
        "stnIds": STN_ID,
    }

    try:
        response = requests.get(
            ASOS_URL,
            params=params,
            timeout=40,
        )
        response.raise_for_status()

        result = response.json()

        items = (
            result.get("response", {})
            .get("body", {})
            .get("items", {})
            .get("item", [])
        )

        if not items:
            raise RuntimeError(
                f"{year}년 ASOS 데이터가 없습니다."
            )

        return pd.DataFrame(items)

    except Exception as e:
        raise RuntimeError(
            f"{year}년 ASOS 데이터 수집 실패: {e}"
        ) from e


# ============================================================
# 5. 계절 변환 함수
# ============================================================

def get_display_season(month: int) -> str:
    """
    화면 시각화용 계절명.
    모델용 season은 create_environment_features()에서
    별도로 동일한 규칙으로 생성한다.
    """

    if month in [3, 4, 5]:
        return "1. 봄 (3~5월)"
    elif month in [6, 7, 8]:
        return "2. 여름 (6~8월)"
    elif month in [9, 10, 11]:
        return "3. 가을 (9~11월)"
    else:
        return "4. 겨울 (12~2월)"


# ============================================================
# 6. GitHub 업로드 함수
# ============================================================

def upload_to_github(
    dataframe: pd.DataFrame,
    git_file_path: str,
    commit_message: str,
) -> None:
    """
    Streamlit Secrets의 GitHub 정보로
    CSV를 GitHub 저장소에 생성 또는 업데이트한다.
    """

    try:
        token = st.secrets.get("GITHUB_TOKEN", "")
        repo_name = st.secrets.get("GITHUB_REPO", "")

        if not token or not repo_name:
            st.warning(
                "⚠️ GITHUB_TOKEN 또는 GITHUB_REPO가 없어 "
                "GitHub 자동 업로드를 건너뜁니다."
            )
            return

        g = Github(token)
        repo = g.get_repo(repo_name)

        file_content = dataframe.to_csv(
            index=False,
            encoding="utf-8-sig",
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
                f"☁️ GitHub [{git_file_path}] 업데이트 완료",
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
                f"☁️ GitHub [{git_file_path}] 새 파일 생성 완료",
                icon="🚀",
            )

    except Exception as e:
        st.warning(
            f"⚠️ GitHub 자동 업로드 중 오류: {e}"
        )


# ============================================================
# 7. 학습 데이터 수집 및 전처리
# ============================================================

def collect_and_process_data(
    status_container,
    progress_bar,
):
    """
    1) 2019~2025 ASOS 수집
    2) 대기환경 병합
    3) 이상치/결측값 정리
    4) 날짜 연속성 검사
    5) 공통 파생변수 생성
    6) 로컬 + GitHub 저장

    Returns
    -------
    cleaned_df:
        기상+대기환경 전처리 완료 데이터

    train_df:
        create_environment_features() 적용 완료 데이터
    """

    total_years = END_YEAR - START_YEAR + 1
    all_years = []

    # --------------------------------------------------------
    # 7-1. ASOS 수집
    # --------------------------------------------------------

    for i, year in enumerate(
        range(START_YEAR, END_YEAR + 1),
        start=1,
    ):
        status_container.update(
            label=(
                f"📡 [{i}/{total_years}] "
                f"{year}년 기상 공공 API 데이터 수집 중..."
            ),
            state="running",
        )

        df_year = fetch_asos_year(year)
        all_years.append(df_year)

        progress_bar.progress(
            i / (total_years + 4)
        )

        time.sleep(0.05)

    if not all_years:
        raise RuntimeError(
            "기상 데이터를 한 건도 수집하지 못했습니다."
        )

    # --------------------------------------------------------
    # 7-2. ASOS 데이터 정리
    # --------------------------------------------------------

    status_container.update(
        label="🔄 기상 데이터 통합 및 정제 중...",
        state="running",
    )

    weather_raw = pd.concat(
        all_years,
        ignore_index=True,
    )

    required_weather_cols = [
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

    missing_weather_cols = [
        col
        for col in required_weather_cols
        if col not in weather_raw.columns
    ]

    if missing_weather_cols:
        raise ValueError(
            "기상 API 응답에 필요한 컬럼이 없습니다: "
            f"{missing_weather_cols}"
        )

    weather = weather_raw[
        required_weather_cols
    ].copy()

    # 중요:
    # sumSsHr는 일사량이 아니라 일조시간이므로
    # solar_radiation이 아니라 sunshine_hours로 사용
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

    weather_numeric_cols = [
        "temp_avg",
        "temp_max",
        "temp_min",
        "humidity",
        "rainfall",
        "wind_speed",
        "sunshine_hours",
        "ground_temp",
    ]

    for col in weather_numeric_cols:
        weather[col] = pd.to_numeric(
            weather[col],
            errors="coerce",
        )

    # 강수량 결측은 0 mm
    weather["rainfall"] = (
        weather["rainfall"]
        .fillna(0)
    )

    weather = (
        weather
        .dropna(subset=["date"])
        .sort_values("date")
        .reset_index(drop=True)
    )

    progress_bar.progress(
        (total_years + 1) / (total_years + 4)
    )

    # --------------------------------------------------------
    # 7-3. 대기환경 데이터
    # --------------------------------------------------------

    status_container.update(
        label="🌫️ 대기환경 학습 데이터 불러오는 중...",
        state="running",
    )

    air = pd.read_csv(
        AIR_TRAIN_URL
    )

    required_air_cols = [
        "date",
        "pm10",
        "pm25",
        "o3",
        "no2",
        "co",
        "so2",
    ]

    missing_air_cols = [
        col
        for col in required_air_cols
        if col not in air.columns
    ]

    if missing_air_cols:
        raise ValueError(
            "대기환경 데이터에 필요한 컬럼이 없습니다: "
            f"{missing_air_cols}"
        )

    air = air[
        required_air_cols
    ].copy()

    air["date"] = pd.to_datetime(
        air["date"],
        errors="coerce",
    ).dt.floor("D")

    air_numeric_cols = [
        "pm10",
        "pm25",
        "o3",
        "no2",
        "co",
        "so2",
    ]

    for col in air_numeric_cols:
        air[col] = pd.to_numeric(
            air[col],
            errors="coerce",
        )

    air = (
        air
        .dropna(subset=["date"])
        .sort_values("date")
        .reset_index(drop=True)
    )

    # 같은 날짜가 여러 건이면 학습자료가 1일 1행이 되도록 평균
    if air["date"].duplicated().any():
        air = (
            air
            .groupby("date", as_index=False)
            .mean(numeric_only=True)
            .sort_values("date")
            .reset_index(drop=True)
        )

    progress_bar.progress(
        (total_years + 2) / (total_years + 4)
    )

    # --------------------------------------------------------
    # 7-4. 기상 + 대기환경 Left Join
    # --------------------------------------------------------

    status_container.update(
        label="🔗 기상 + 대기환경 자료 병합 중...",
        state="running",
    )

    df = pd.merge(
        weather,
        air,
        on="date",
        how="left",
    )

    df = (
        df
        .sort_values("date")
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # 7-5. 이상치 처리
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
        if col in df.columns:
            df.loc[
                df[col] < 0,
                col,
            ] = np.nan

    # 습도 범위
    df.loc[
        (df["humidity"] < 0)
        | (df["humidity"] > 100),
        "humidity",
    ] = np.nan

    # PM 극단값
    df.loc[
        df["pm10"] > 1000,
        "pm10",
    ] = np.nan

    df.loc[
        df["pm25"] > 500,
        "pm25",
    ] = np.nan

    # 강수량은 결측 = 무강수로 처리
    df["rainfall"] = (
        df["rainfall"]
        .fillna(0)
    )

    # --------------------------------------------------------
    # 7-6. 시계열 결측값 처리
    # 미래값을 사용하지 않고 ffill만 사용
    # --------------------------------------------------------

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
        if col in df.columns
    ]

    df[cols_to_fill] = (
        df[cols_to_fill]
        .ffill()
    )

    before_drop = len(df)

    # 데이터 시작구간에서 과거값이 없어
    # ffill할 수 없는 행은 제거
    df = (
        df
        .dropna(subset=cols_to_fill)
        .reset_index(drop=True)
    )

    removed_rows = (
        before_drop - len(df)
    )

    # --------------------------------------------------------
    # 7-7. 날짜 연속성 검사
    # rolling(7), rolling(28)이 실제 일수 기준인지 확인
    # --------------------------------------------------------

    expected_dates = pd.date_range(
        start=df["date"].min(),
        end=df["date"].max(),
        freq="D",
    )

    missing_dates = (
        expected_dates
        .difference(df["date"])
    )

    if len(missing_dates) > 0:
        raise ValueError(
            "학습 데이터 날짜가 연속적이지 않습니다. "
            f"누락 날짜 {len(missing_dates)}일: "
            f"{list(missing_dates[:10])}"
        )

    # --------------------------------------------------------
    # 7-8. 공통 파생변수 함수 적용
    # 학습/실시간 예측에서 동일 함수 사용
    # --------------------------------------------------------

    status_container.update(
        label="🧮 학습용 환경 파생변수 생성 중...",
        state="running",
    )

    train_df = (
        create_environment_features(
            df.copy(),
            fill_remaining_numeric=True,
        )
    )

    # 화면 시각화 편의를 위한 컬럼
    # 모델용 season 컬럼은 create_environment_features가 생성함
    train_df["year"] = (
        train_df["date"].dt.year
    )

    train_df["display_season"] = (
        train_df["date"]
        .dt.month
        .apply(get_display_season)
    )

    progress_bar.progress(
        (total_years + 3) / (total_years + 4)
    )

    # --------------------------------------------------------
    # 7-9. 학습 데이터 기간 안전성 검사
    # --------------------------------------------------------

    actual_years = set(
        train_df["date"]
        .dt.year
        .unique()
    )

    required_years = set(
        range(
            START_YEAR,
            END_YEAR + 1,
        )
    )

    missing_years = (
        required_years
        - actual_years
    )

    if missing_years:
        raise ValueError(
            "학습 데이터에 누락 연도가 있습니다: "
            f"{sorted(missing_years)}"
        )

    # --------------------------------------------------------
    # 7-10. 로컬 CSV 저장
    # --------------------------------------------------------

    status_container.update(
        label="💾 학습 데이터 파일 저장 중...",
        state="running",
    )

    # 파생변수 생성 전 전처리 데이터
    df.to_csv(
        RAW_SAVE_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    # 최종 학습 Feature 데이터
    train_df.to_csv(
        FEATURE_SAVE_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # 7-11. GitHub 자동 업로드
    # --------------------------------------------------------

    status_container.update(
        label="☁️ GitHub 저장소 자동 업로드 중...",
        state="running",
    )

    upload_to_github(
        df,
        f"data/processed/{RAW_FILE_NAME}",
        (
            f"chore: {START_YEAR}~{END_YEAR} "
            "영천 전처리 데이터 업데이트"
        ),
    )

    upload_to_github(
        train_df,
        f"data/processed/{FEATURE_FILE_NAME}",
        (
            f"chore: {START_YEAR}~{END_YEAR} "
            "영천 학습 Feature 데이터 업데이트"
        ),
    )

    progress_bar.progress(1.0)

    status_container.update(
        label="✅ 학습 데이터 구축 및 파생변수 생성 완료!",
        state="complete",
        expanded=False,
    )

    return (
        df,
        train_df,
        removed_rows,
        len(missing_dates),
    )


# ============================================================
# 8. Session State
# ============================================================

if "df_cleaned" not in st.session_state:
    st.session_state.df_cleaned = None

if "df_train_features" not in st.session_state:
    st.session_state.df_train_features = None

if "data_quality_info" not in st.session_state:
    st.session_state.data_quality_info = None


# ============================================================
# 9. 데이터 수집 버튼
# ============================================================

col_ui1, col_ui2 = st.columns(
    [1, 3],
    vertical_alignment="center",
)

with col_ui1:
    collect_clicked = st.button(
        "🚀 데이터 수집 시작",
        type="primary",
        use_container_width=True,
    )

with col_ui2:
    if st.session_state.df_train_features is None:
        st.info(
            "💡 버튼을 누르면 2019~2025년 기상·대기환경 자료를 "
            "수집하고 학습용 파생변수를 생성합니다."
        )
    else:
        st.success(
            "✅ 학습용 데이터셋과 파생변수 데이터가 준비되었습니다."
        )


if collect_clicked:
    try:
        status_box = st.status(
            "데이터 수집 준비 중...",
            expanded=True,
        )

        prog_bar = st.progress(0)

        (
            cleaned_df,
            train_df,
            removed_rows,
            missing_date_count,
        ) = collect_and_process_data(
            status_box,
            prog_bar,
        )

        st.session_state.df_cleaned = (
            cleaned_df
        )

        st.session_state.df_train_features = (
            train_df
        )

        st.session_state.data_quality_info = {
            "removed_rows": removed_rows,
            "missing_date_count": missing_date_count,
        }

        st.rerun()

    except Exception as e:
        st.error(
            f"❌ 학습 데이터 구축 실패: {e}"
        )


# ============================================================
# 10. 결과 화면
# ============================================================

df = st.session_state.df_train_features
cleaned_df = st.session_state.df_cleaned
quality_info = st.session_state.data_quality_info


if df is not None:

    # ========================================================
    # 10-1. 다운로드
    # ========================================================

    col_d1, col_d2 = st.columns(2)

    with col_d1:
        raw_csv_bytes = (
            cleaned_df
            .to_csv(index=False)
            .encode("utf-8-sig")
        )

        st.download_button(
            label=f"📥 {RAW_FILE_NAME} 다운로드",
            data=raw_csv_bytes,
            file_name=RAW_FILE_NAME,
            mime="text/csv",
            use_container_width=True,
        )

    with col_d2:
        feature_csv_bytes = (
            df
            .to_csv(index=False)
            .encode("utf-8-sig")
        )

        st.download_button(
            label=f"📥 {FEATURE_FILE_NAME} 다운로드",
            data=feature_csv_bytes,
            file_name=FEATURE_FILE_NAME,
            mime="text/csv",
            type="primary",
            use_container_width=True,
        )

    # ========================================================
    # 10-2. 데이터 품질 리포트
    # ========================================================

    st.markdown("---")

    with st.expander(
        "🔍 데이터 전처리 및 품질 리포트",
        expanded=False,
    ):
        col_r1, col_r2, col_r3, col_r4, col_r5 = st.columns(5)

        col_r1.metric(
            "최종 관측 일수",
            f"{len(df):,}일",
        )

        col_r2.metric(
            "시작일",
            df["date"].min().strftime("%Y-%m-%d"),
        )

        col_r3.metric(
            "종료일",
            df["date"].max().strftime("%Y-%m-%d"),
        )

        col_r4.metric(
            "초기 결측 제거",
            f"{quality_info['removed_rows']}일",
        )

        col_r5.metric(
            "날짜 누락",
            f"{quality_info['missing_date_count']}일",
        )

        air_merge_rate = (
            cleaned_df["pm10"]
            .notna()
            .mean()
            * 100
        )

        st.write(
            f"**PM10 병합·보완 후 유효 비율:** "
            f"{air_merge_rate:.1f}%"
        )

        st.info(
            "전처리 원칙: 미래 시점 값을 과거에 사용하는 bfill이나 "
            "양방향 보간은 사용하지 않고 ffill만 사용합니다. "
            "데이터 시작구간에서 과거 관측값이 없어 채울 수 없는 행은 제거합니다."
        )

    # ========================================================
    # 10-3. 연도별 데이터 확인
    # ========================================================

    st.markdown("---")
    st.subheader("📅 학습 데이터 연도별 구성")

    year_counts = (
        df["date"]
        .dt.year
        .value_counts()
        .sort_index()
        .rename("건수")
        .reset_index()
    )

    year_counts.columns = [
        "연도",
        "건수",
    ]

    st.dataframe(
        year_counts,
        use_container_width=True,
        hide_index=True,
    )

    # ========================================================
    # 10-4. KPI
    # ========================================================

    st.markdown("---")
    st.subheader("📌 수집 데이터 주요 요약 지표")

    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)

    kpi1.metric(
        "총 관측 일수",
        f"{len(df):,}일",
    )

    kpi2.metric(
        "평균 기온",
        f"{df['temp_avg'].mean():.1f} °C",
    )

    kpi3.metric(
        "평균 습도",
        f"{df['humidity'].mean():.1f} %",
    )

    kpi4.metric(
        "평균 PM10",
        f"{df['pm10'].mean():.1f} ㎍/㎥",
    )

    kpi5.metric(
        "평균 PM2.5",
        f"{df['pm25'].mean():.1f} ㎍/㎥",
    )

    # ========================================================
    # 10-5. 파생변수 확인
    # ========================================================

    st.markdown("---")
    st.subheader("🧮 모델 학습용 주요 파생변수")

    important_features = [
        "temp_range",
        "temp_change",
        "humidity_change",
        "humidity_std3",
        "rainfall_7d",
        "rh60_days_28",
        "rh75_days_28",
        "rh75_consecutive_days",
        "wood_mold_days_28",
        "rh70_days_28",
        "rh70_consecutive_days",
        "metal_so2_humidity",
        "pm_total",
        "pm_load_7d",
        "so2_ma7",
        "no2_ma7",
        "o3_ma7",
    ]

    feature_check = pd.DataFrame({
        "파생변수": important_features,
        "포함 여부": [
            "✅"
            if col in df.columns
            else "❌"
            for col in important_features
        ],
    })

    st.dataframe(
        feature_check,
        use_container_width=True,
        hide_index=True,
    )

    # ========================================================
    # 10-6. 계절 분석
    # ========================================================

    st.markdown("---")
    st.subheader("📈 계절별 기상 및 대기환경 종합 분석")

    row1_col1, row1_col2 = st.columns(2)

    with row1_col1:

        df_season_avg = (
            df
            .groupby("display_season")
            .agg({
                "temp_avg": "mean",
                "humidity": "mean",
            })
            .reset_index()
        )

        fig_season = make_subplots(
            specs=[[
                {
                    "secondary_y": True
                }
            ]]
        )

        fig_season.add_trace(
            go.Bar(
                x=df_season_avg["display_season"],
                y=df_season_avg["temp_avg"],
                name="평균 기온 (°C)",
                text=df_season_avg["temp_avg"].round(1),
                textposition="auto",
            ),
            secondary_y=False,
        )

        fig_season.add_trace(
            go.Scatter(
                x=df_season_avg["display_season"],
                y=df_season_avg["humidity"],
                name="평균 습도 (%)",
                mode="lines+markers+text",
                text=(
                    df_season_avg["humidity"]
                    .round(1)
                    .astype(str)
                    + "%"
                ),
                textposition="top center",
            ),
            secondary_y=True,
        )

        fig_season.update_layout(
            title="🌸☀️🍁❄️ 계절별 평균 기온 및 습도",
            xaxis_title="계절",
            height=420,
            hovermode="x unified",
        )

        fig_season.update_yaxes(
            title_text="기온 (°C)",
            secondary_y=False,
        )

        fig_season.update_yaxes(
            title_text="습도 (%)",
            secondary_y=True,
            range=[0, 100],
        )

        st.plotly_chart(
            fig_season,
            use_container_width=True,
        )

    with row1_col2:

        df_yearly_season = (
            df
            .groupby(
                [
                    "year",
                    "display_season",
                ]
            )["temp_avg"]
            .mean()
            .reset_index()
        )

        fig_season_trend = px.line(
            df_yearly_season,
            x="year",
            y="temp_avg",
            color="display_season",
            markers=True,
            title="📅 연도별 계절 평균 기온 추이",
            labels={
                "temp_avg": "평균 기온 (°C)",
                "year": "연도",
                "display_season": "계절",
            },
        )

        fig_season_trend.update_layout(
            height=420,
            hovermode="x unified",
            xaxis=dict(type="category"),
        )

        st.plotly_chart(
            fig_season_trend,
            use_container_width=True,
        )

    row2_col1, row2_col2 = st.columns(2)

    with row2_col1:

        df_season_air = (
            df
            .groupby("display_season")[
                [
                    "pm10",
                    "pm25",
                ]
            ]
            .mean()
            .reset_index()
        )

        fig_air_season = px.bar(
            df_season_air,
            x="display_season",
            y=[
                "pm10",
                "pm25",
            ],
            barmode="group",
            title="🌫️ 계절별 PM10·PM2.5 평균",
            labels={
                "value": "농도 (㎍/㎥)",
                "display_season": "계절",
                "variable": "구분",
            },
            text_auto=".1f",
        )

        fig_air_season.update_layout(
            height=420,
            hovermode="x unified",
        )

        st.plotly_chart(
            fig_air_season,
            use_container_width=True,
        )

    with row2_col2:

        # 기존 코드의 sumSsHr는 일사량이 아니라 일조시간임
        df_season_rain = (
            df
            .groupby("display_season")
            .agg({
                "rainfall": "sum",
                "sunshine_hours": "mean",
            })
            .reset_index()
        )

        fig_rain_season = (
            make_subplots(
                specs=[[
                    {
                        "secondary_y": True
                    }
                ]]
            )
        )

        fig_rain_season.add_trace(
            go.Bar(
                x=df_season_rain["display_season"],
                y=df_season_rain["rainfall"],
                name="총 강수량 (mm)",
                text=df_season_rain["rainfall"].round(1),
                textposition="auto",
            ),
            secondary_y=False,
        )

        fig_rain_season.add_trace(
            go.Scatter(
                x=df_season_rain["display_season"],
                y=df_season_rain["sunshine_hours"],
                name="평균 일조시간 (hr)",
                mode="lines+markers+text",
                text=(
                    df_season_rain["sunshine_hours"]
                    .round(2)
                ),
                textposition="top center",
            ),
            secondary_y=True,
        )

        fig_rain_season.update_layout(
            title="🌧️☀️ 계절별 총 강수량과 평균 일조시간",
            xaxis_title="계절",
            height=420,
            hovermode="x unified",
        )

        fig_rain_season.update_yaxes(
            title_text="강수량 (mm)",
            secondary_y=False,
        )

        fig_rain_season.update_yaxes(
            title_text="일조시간 (hr)",
            secondary_y=True,
        )

        st.plotly_chart(
            fig_rain_season,
            use_container_width=True,
        )

    # ========================================================
    # 10-7. 원본 미리보기
    # ========================================================

    st.markdown("---")

    with st.expander(
        "📋 최종 학습 Feature 데이터 최근 100건",
        expanded=False,
    ):

        st.dataframe(
            df.sort_values(
                "date",
                ascending=False,
            ).head(100),
            use_container_width=True,
        )
