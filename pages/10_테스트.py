from datetime import datetime, timedelta
import itertools  # 👈 itertools 임포트 추가 완료
import time
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from streamlit_autorefresh import st_autorefresh
import streamlit as st

# ------------------------------------------------------------
# 1. 페이지 설정 및 실시간 리프레시 (2초 간격)
# ------------------------------------------------------------
st.set_page_config(
    page_title="문화재 실시간 환경 모니터링 & 위험 진단",
    page_icon="🏛️",
    layout="wide",
)

st_autorefresh(interval=2 * 1000, key="sensor_refresh")

FIREBASE_SENSOR_URL = "https://heritage-project-4a361-default-rtdb.asia-southeast1.firebasedatabase.app/sensor.json"
FIREBASE_HISTORY_URL = "https://heritage-project-4a361-default-rtdb.asia-southeast1.firebasedatabase.app/sensor/history.json"
DATA_PATH = "data/processed/[2016_2025] yeongcheon.csv"

FEATURE_NAME_KO = {
    "corrosion_risk": "부식 위험도",
    "humidity": "평균 습도",
    "high_humidity_risk": "고습도 위험 지속도",
    "oxidation_risk": "산화 위험도",
    "acid_risk": "산성 위험도",
    "pm25": "초미세먼지(PM2.5)",
    "weathering_risk": "풍화 위험도",
    "no2": "이산화질소(NO2)",
    "pm_load": "미세먼지 누적부하",
    "pm10": "미세먼지(PM10)",
    "temp_range": "일교차",
    "humidity_std3": "3일 습도 변동성",
    "rainfall_7d": "7일 누적 강수량",
    "mold_risk": "곰팡이 발생 위험도",
    "temp_avg": "평균 기온",
    "temp_max": "최고 기온",
    "temp_min": "최저 기온",
    "rainfall": "일 강수량",
    "wind_speed": "평균 풍속",
    "solar_radiation": "일사량",
    "ground_temp": "지면 온도",
    "o3": "오존(O3)",
    "co": "일산화탄소(CO)",
    "so2": "아황산가스(SO2)",
}


# ------------------------------------------------------------
# 2. 유틸리티 함수
# ------------------------------------------------------------
def to_float(value):
    try:
        return float(value)
    except:
        return 0.0


def parse_time(timestamp):
    try:
        return datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
    except:
        return None


def get_device_status(timestamp):
    dt = parse_time(timestamp)
    if dt is None:
        return "⚠️ 시간 오류"
    now = datetime.now()
    diff = now - dt
    if diff > timedelta(minutes=10):
        return "🔴 수신 지연"
    elif diff > timedelta(minutes=2):
        return "🟠 확인 필요"
    else:
        return "🟢 정상 수신"


def metric_value(value, unit, zero_check=False):
    if zero_check and value == 0:
        return "센서 확인"
    return f"{value:.1f} {unit}"


# ------------------------------------------------------------
# 3. 데이터 및 모델 캐싱 로드
# ------------------------------------------------------------
@st.cache_resource
def load_ml_pipeline():
    try:
        df = pd.read_csv(DATA_PATH)
    except:
        return None, None, None

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    if "rainfall" in df.columns:
        df["rainfall"] = df["rainfall"].fillna(0)

    # 파생변수 생성
    df["temp_range"] = df["temp_max"] - df["temp_min"]
    df["humidity_std3"] = df["humidity"].rolling(3, min_periods=1).std()
    df["rainfall_7d"] = df["rainfall"].rolling(7, min_periods=1).sum()
    df["high_humidity_risk"] = (
        (df["humidity"] >= 75).rolling(3, min_periods=1).sum()
    )
    df["weathering_risk"] = (
        df["temp_range"] * 0.4 + df["humidity_std3"] * 0.3 + df["wind_speed"] * 0.3
    )
    df["mold_risk"] = (
        (df["humidity"] >= 75) & (df["ground_temp"] >= 15)
    ).astype(int)
    df["pm_load"] = (df["pm10"] + df["pm25"]).rolling(3, min_periods=1).sum()
    df["acid_risk"] = df["so2"] * 0.6 + df["no2"] * 0.4
    df["oxidation_risk"] = df["o3"] * 0.7 + df["pm25"] * 0.3
    df["corrosion_risk"] = df["humidity"] * 0.5 + df["so2"] * 0.5
    df = df.fillna(0)

    materials = ["석조", "목조", "금속", "회화", "기타"]
    exposures = ["실외", "반실외", "실내"]
    comb = pd.DataFrame(
        list(itertools.product(materials, exposures)),
        columns=["material", "exposure"],
    )

    df["key"] = 1
    comb["key"] = 1
    dataset = pd.merge(df, comb, on="key").drop("key", axis=1)

    risk_cols = [
        "weathering_risk",
        "acid_risk",
        "rainfall_7d",
        "temp_range",
        "pm_load",
        "corrosion_risk",
        "mold_risk",
        "humidity_std3",
        "oxidation_risk",
        "high_humidity_risk",
    ]
    for col in risk_cols:
        dataset[col + "_norm"] = (
            (dataset[col] - dataset[col].min())
            / (dataset[col].max() - dataset[col].min() + 1e-6)
        ) * 100

    mat = dataset["material"].values
    exp = dataset["exposure"].values

    calc_석조 = (
        dataset["weathering_risk_norm"] * 0.25
        + dataset["acid_risk_norm"] * 0.20
        + dataset["rainfall_7d_norm"] * 0.18
        + dataset["temp_range_norm"] * 0.15
        + dataset["pm_load_norm"] * 0.12
        + dataset["corrosion_risk_norm"] * 0.10
    )
    calc_목조 = (
        dataset["mold_risk_norm"] * 0.25
        + dataset["humidity_std3_norm"] * 0.20
        + dataset["high_humidity_risk_norm"] * 0.18
        + dataset["rainfall_7d_norm"] * 0.15
        + dataset["oxidation_risk_norm"] * 0.12
        + dataset["pm_load_norm"] * 0.10
    )
    calc_금속 = (
        dataset["corrosion_risk_norm"] * 0.30
        + dataset["acid_risk_norm"] * 0.22
        + dataset["high_humidity_risk_norm"] * 0.18
        + dataset["humidity_std3_norm"] * 0.12
        + dataset["pm_load_norm"] * 0.10
        + dataset["weathering_risk_norm"] * 0.08
    )
    calc_회화 = (
        dataset["oxidation_risk_norm"] * 0.28
        + dataset["pm_load_norm"] * 0.20
        + dataset["humidity_std3_norm"] * 0.18
        + dataset["high_humidity_risk_norm"] * 0.14
        + dataset["temp_range_norm"] * 0.10
        + dataset["weathering_risk_norm"] * 0.10
    )
    calc_기타 = (
        dataset["weathering_risk_norm"] * 0.2
        + dataset["acid_risk_norm"] * 0.2
        + dataset["oxidation_risk_norm"] * 0.2
        + dataset["corrosion_risk_norm"] * 0.2
        + dataset["pm_load_norm"] * 0.2
    )

    r_base = np.select(
        [mat == "석조", mat == "목조", mat == "금속", mat == "회화"],
        [calc_석조, calc_목조, calc_금속, calc_회화],
        default=calc_기타,
    )
    exp_mult = np.select(
        [exp == "실외", exp == "반실외"], [1.3, 1.1], default=0.85
    )
    dataset["material_risk"] = np.clip(r_base * exp_mult, 0, 100)

    conditions = [dataset["material_risk"] >= 80, dataset["material_risk"] >= 40]
    dataset["target"] = np.select(conditions, ["위험", "주의"], default="안전")

    X = dataset[[
        "temp_avg",
        "temp_max",
        "temp_min",
        "humidity",
        "rainfall",
        "wind_speed",
        "solar_radiation",
        "ground_temp",
        "pm10",
        "pm25",
        "o3",
        "no2",
        "co",
        "so2",
        "temp_range",
        "humidity_std3",
        "rainfall_7d",
        "high_humidity_risk",
        "weathering_risk",
        "mold_risk",
        "pm_load",
        "acid_risk",
        "oxidation_risk",
        "corrosion_risk",
        "material",
        "exposure",
    ]]
    y = dataset["target"]
    X_encoded = pd.get_dummies(X, columns=["material", "exposure"])

    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.2, random_state=42, stratify=y
    )
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)

    return dataset, rf_model, X_encoded


@st.cache_data(ttl=2)
def load_realtime_devices():
    try:
        response = requests.get(
            FIREBASE_SENSOR_URL, params={"t": time.time()}, timeout=10
        )
        if response.status_code != 200:
            return {}
        sensor_data = response.json()
        if sensor_data is None:
            return {}
        return {
            k: v for k, v in sensor_data.items() if k.startswith("realtime_device_")
        }
    except:
        return {}


@st.cache_data(ttl=20)
def load_history_data():
    try:
        response = requests.get(
            FIREBASE_HISTORY_URL, params={"t": time.time()}, timeout=10
        )
        if response.status_code != 200:
            return pd.DataFrame()
        history_data = response.json()
        if history_data is None:
            return pd.DataFrame()

        df = pd.DataFrame(history_data).T
        df.reset_index(drop=True, inplace=True)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

        numeric_cols = ["temperature", "humidity", "pressure", "light", "pm1", "pm25", "pm10"]
        for col in numeric_cols:
            if col not in df.columns:
                df[col] = 0
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

        return df.dropna(subset=["timestamp"]).sort_values("timestamp")
    except:
        return pd.DataFrame()


# ------------------------------------------------------------
# 4. 메인 대시보드 레이아웃
# ------------------------------------------------------------
st.title("🏛️ 문화재 실시간 환경 모니터링 & 스마트 위험 진단")
st.subheader("BME280 / BH1750 / PMS7003 센서 데이터 스트림 및 머신러닝 위험도 분석")

realtime_devices = load_realtime_devices()
dataset, rf_model, X_encoded = load_ml_pipeline()

if not realtime_devices:
    st.warning("⚠️ Firebase에 저장된 실시간 장치 데이터가 없습니다.")
else:
    if "last_timestamps" not in st.session_state:
        st.session_state.last_timestamps = {}

    new_devices = []
    for device_key, data in realtime_devices.items():
        timestamp = data.get("timestamp", "-")
        device_name = data.get("device", device_key)
        old_timestamp = st.session_state.last_timestamps.get(device_key)
        if old_timestamp is not None and old_timestamp != timestamp:
            new_devices.append(device_name)
        st.session_state.last_timestamps[device_key] = timestamp

    if len(new_devices) > 0:
        st.success(f"🆕 [{', '.join(new_devices)}] 실시간 센서 데이터가 업데이트되었습니다.")

    st.divider()

    # 실시간 센서 카드 UI
    for device_key, data in sorted(realtime_devices.items()):
        temp = to_float(data.get("temperature", 0))
        hum = to_float(data.get("humidity", 0))
        pressure = to_float(data.get("pressure", 0))
        light = to_float(data.get("light", 0))
        pm1 = to_float(data.get("pm1", 0))
        pm25 = to_float(data.get("pm25", 0))
        pm10 = to_float(data.get("pm10", 0))
        timestamp = data.get("timestamp", "-")
        device = data.get("device", device_key)
        status = get_device_status(timestamp)

        t_col1, t_col2 = st.columns([3, 1])
        with t_col1:
            st.subheader(f"📡 {device}")
        with t_col2:
            st.markdown(f"### {status}")
        st.caption(f"마지막 측정 시간 : {timestamp}")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("🌡️ 기온", metric_value(temp, "℃", zero_check=True))
        c2.metric("💧 습도", metric_value(hum, "%", zero_check=True))
        c3.metric("🌬️ 기압", metric_value(pressure, "hPa", zero_check=True))
        c4.metric("☀️ 조도", metric_value(light, "lux", zero_check=True))

        c5, c6, c7, c8 = st.columns(4)
        c5.metric("🌫️ PM1.0", f"{pm1:.1f} ㎍/㎥")
        c6.metric("🌫️ PM2.5", f"{pm25:.1f} ㎍/㎥")
        c7.metric("🌫️ PM10", f"{pm10:.1f} ㎍/㎥")
        c8.empty()

        # ------------------------------------------------------------
        # 5. 실시간 센서 연계 스마트 문화재 위험 진단 추가 영역
        # ------------------------------------------------------------
        st.markdown("#### 🛡️ 실시간 센서 기반 문화재 맞춤 위험 진단")
        d_col1, d_col2 = st.columns(2)
        with d_col1:
            eval_material = st.selectbox(
                "관심 문화재 재질 선택", ["목조", "석조", "금속", "회화", "기타"], key=f"mat_{device_key}"
            )
        with d_col2:
            eval_exposure = st.selectbox(
                "노출 환경 선택", ["실외", "반실외", "실내"], key=f"exp_{device_key}"
            )

        # 실시간 룰 기반 경고 진단 (목조 곰팡이, 금속 고습 등)
        mold_danger = (hum >= 75) and (temp >= 15)
        corrosion_danger = (hum >= 80)

        if eval_material == "목조" and mold_danger:
            st.error(
                f"🚨 **[위험 경보] {device}** : 현재 습도({hum}%)와 기온({temp}℃)이 목조 문화재 곰팡이 번식 최적 조건입니다! 즉각적인 환기 및 제습이 필요합니다."
            )
        elif eval_material == "금속" and corrosion_danger:
            st.warning(
                f"⚠️ **[주의 경보] {device}** : 고습 상태 지속으로 금속 표면 산화 및 부식 위험이 높습니다."
            )
        else:
            st.success(
                f"✅ **[안전] {device}** : 현재 [{eval_material} / {eval_exposure}] 환경 조건은 비교적 안정적입니다."
            )

        st.divider()

# ------------------------------------------------------------
# 6. 이력 통계 및 트렌드 시각화 섹션
# ------------------------------------------------------------
st.subheader("📊 센서 데이터 이력 통계 및 트렌드 분석")
history_df = load_history_data()

if history_df.empty:
    st.info("아직 누적된 센서 이력 데이터가 없습니다.")
else:
    history_df["date"] = history_df["timestamp"].dt.date
    min_date, max_date = history_df["date"].min(), history_df["date"].max()

    if "device" in history_df.columns:
        device_list = sorted(history_df["device"].dropna().unique())
        selected_devices = st.multiselect("조회할 장치 선택", device_list, default=device_list)
        history_df = history_df[history_df["device"].isin(selected_devices)]

    st.markdown("#### 📅 조회 기간 선택")
    d_col1, d_col2 = st.columns(2)
    with d_col1:
        start_date = st.date_input("시작 날짜", value=min_date, min_value=min_date, max_value=max_date)
    with d_col2:
        end_date = st.date_input("종료 날짜", value=max_date, min_value=min_date, max_value=max_date)

    if start_date > end_date:
        st.error("시작 날짜가 종료 날짜보다 늦을 수 없습니다.")
        st.stop()

    filtered_df = history_df[(history_df["date"] >= start_date) & (history_df["date"] <= end_date)]

    if filtered_df.empty:
        st.warning("선택한 조건에 해당하는 데이터가 없습니다.")
        st.stop()

    st.markdown("#### 📈 선택 기간 센서 변화 추이")
    chart_cols = ["temperature", "humidity", "pressure", "light", "pm1", "pm25", "pm10"]
    selected_cols = st.multiselect("그래프 표시 항목", chart_cols, default=["temperature", "humidity", "pm25"])

    if selected_cols:
        st.line_chart(filtered_df, x="timestamp", y=selected_cols)

    # 데이터 다운로드 버튼
    csv = filtered_df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        label="📥 선택 기간 센서 이력 CSV 다운로드",
        data=csv,
        file_name="firebase_sensor_history_filtered.csv",
        mime="text/csv",
    )
