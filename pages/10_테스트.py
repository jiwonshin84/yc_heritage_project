from datetime import datetime, timedelta
import itertools
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
    page_title="문화재 실시간 통합 모니터링 & 스마트 위험 진단",
    page_icon="🏛️",
    layout="wide",
)

st_autorefresh(interval=2 * 1000, key="sensor_refresh")

FIREBASE_SENSOR_URL = "https://heritage-project-4a361-default-rtdb.asia-southeast1.firebasedatabase.app/sensor.json"
FIREBASE_HISTORY_URL = "https://heritage-project-4a361-default-rtdb.asia-southeast1.firebasedatabase.app/sensor/history.json"
DATA_PATH = "data/processed/[2016_2025] yeongcheon.csv"


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
st.title("🏛️ 문화재 실시간 통합 모니터링 & 스마트 위험 진단")
st.subheader("복수 센서(Pico) 평균 연동 및 재질·환경별 실시간 위험도 카드 분석")

realtime_devices = load_realtime_devices()
dataset, rf_model, X_encoded = load_ml_pipeline()

if not realtime_devices:
    st.warning("⚠️ Firebase에 저장된 실시간 장치 데이터가 없습니다.")
else:
    # ------------------------------------------------------------
    # 4-1. 개별 센서 데이터 출력 및 평균치(Integrated Average) 산출
    # ------------------------------------------------------------
    st.markdown("### 📡 개별 센서 실시간 현황")
    device_items = sorted(realtime_devices.items())
    
    temps, hums, pressures, lights = [], [], [], []
    pm1s, pm25s, pm10s = [], [], []

    cols = st.columns(len(device_items) if len(device_items) > 0 else 1)
    for idx, (device_key, data) in enumerate(device_items):
        temp = to_float(data.get("temperature", 0))
        hum = to_float(data.get("humidity", 0))
        pressure = to_float(data.get("pressure", 0))
        light = to_float(data.get("light", 0))
        pm1 = to_float(data.get("pm1", 0))
        pm25 = to_float(data.get("pm25", 0))
        pm10 = to_float(data.get("pm10", 0))
        timestamp = data.get("timestamp", "-")
        device_name = data.get("device", device_key)
        status = get_device_status(timestamp)

        temps.append(temp)
        hums.append(hum)
        pressures.append(pressure)
        lights.append(light)
        pm1s.append(pm1)
        pm25s.append(pm25)
        pm10s.append(pm10)

        with cols[idx]:
            with st.container(border=True):
                st.markdown(f"**{device_name}** ({status})")
                st.caption(f"측정: {timestamp}")
                st.metric("🌡️ 기온", f"{temp:.1f} ℃")
                st.metric("💧 습도", f"{hum:.1f} %")
                st.metric("🌫️ 미세먼지(PM2.5)", f"{pm25:.1f} ㎍/㎥")

    # 평균치 계산
    avg_temp = np.mean(temps) if temps else 0
    avg_hum = np.mean(hums) if hums else 0
    avg_pressure = np.mean(pressures) if pressures else 0
    avg_light = np.mean(lights) if lights else 0
    avg_pm1 = np.mean(pm1s) if pm1s else 0
    avg_pm25 = np.mean(pm25s) if pm25s else 0
    avg_pm10 = np.mean(pm10s) if pm10s else 0

    st.markdown("---")
    
    # ------------------------------------------------------------
    # 4-2. 통합 대표 센서 값 (두 센서 평균) 요약 섹션
    # ------------------------------------------------------------
    st.markdown("### 📊 센서 통합 대표 평균 지표 (복수 센서 융합)")
    
    avg_c1, avg_c2, avg_c3, avg_c4 = st.columns(4)
    avg_c1.metric("🌡️ 통합 평균 기온", f"{avg_temp:.1f} ℃")
    avg_c2.metric("💧 통합 평균 습도", f"{avg_hum:.1f} %")
    avg_c3.metric("🌬️ 통합 평균 기압", f"{avg_pressure:.1f} hPa")
    avg_c4.metric("☀️ 통합 평균 조도", f"{avg_light:.1f} lux")

    avg_c5, avg_c6, avg_c7, avg_c8 = st.columns(4)
    avg_c5.metric("🌫️ 통합 PM1.0", f"{avg_pm1:.1f} ㎍/㎥")
    avg_c6.metric("🌫️ 통합 PM2.5", f"{avg_pm25:.1f} ㎍/㎥")
    avg_c7.metric("🌫️ 통합 PM10", f"{avg_pm10:.1f} ㎍/㎥")
    avg_c8.metric("📡 연동 센서 대수", f"{len(device_items)} 대")

    st.markdown("---")

    # ------------------------------------------------------------
    # 5. 재질별/노출환경별 위험 진단 카드 UI 
    # ------------------------------------------------------------
    st.markdown("### 🛡️ 문화재 재질 및 노출 환경별 실시간 맞춤 위험 진단")
    
    card_col1, card_col2 = st.columns(2)
    with card_col1:
        eval_material = st.selectbox(
            "진단 대상 문화재 재질 선택", ["목조", "석조", "금속", "회화", "기타"]
        )
    with card_col2:
        eval_exposure = st.selectbox(
            "배치 노출 환경 선택", ["실외", "반실외", "실내"]
        )

    # 룰 기반 정밀 진단 로직 (통합 평균값 기준)
    mold_danger = (avg_hum >= 75) and (avg_temp >= 15)
    corrosion_danger = (avg_hum >= 80)
    high_pm_danger = (avg_pm25 >= 50)

    # 위험 등급 및 메시지 결정
    if (eval_material == "목조" and mold_danger) or (eval_material == "금속" and corrosion_danger):
        risk_level = "위험 (Danger)"
        card_color = "red"
        advice_text = "현재 온습도 및 환경 조건이 문화재 손상 유발 기준치를 초과했습니다. 즉각적인 보호 조치 및 환기/제습이 필요합니다."
    elif high_pm_danger or avg_hum >= 70:
        risk_level = "주의 (Caution)"
        card_color = "orange"
        advice_text = "미세먼지 농도 혹은 습도가 다소 높은 상태입니다. 지속적인 모니터링과 국부 환경 조절을 검토하세요."
    else:
        risk_level = "안전 (Safe)"
        card_color = "green"
        advice_text = "현재 수집된 통합 센서 환경 조건은 선택하신 문화재 재질과 환경에 비교적 안정적인 상태입니다."

    # 카드 형식으로 시각화 (Container 안에서 마크다운 카드로 구성)
    with st.container(border=True):
        st.markdown(f"#### 📌 진단 리포트 : [{eval_material} / {eval_exposure}]")
        
        c_status, c_score = st.columns([2, 2])
        with c_status:
            if risk_level.startswith("위험"):
                st.error(f"### 🚨 판정 등급 : {risk_level}")
            elif risk_level.startswith("주의"):
                st.warning(f"### ⚠️ 판정 등급 : {risk_level}")
            else:
                st.success(f"### ✅ 판정 등급 : {risk_level}")
        with c_score:
            st.metric("진단 기준 통합 습도/기온", f"{avg_hum:.1f}% / {avg_temp:.1f}℃")

        st.markdown(f"**💡 맞춤 관리 가이드** : {advice_text}")
        
        # 세부 항목별 체크리스트 배지 표시
        b1, b2, b3 = st.columns(3)
        b1.markdown(f"- 곰팡이 위험(목조): {'🔴 주의' if mold_danger else '🟢 보통'}")
        b2.markdown(f"- 부식 위험(금속): {'🔴 높음' if corrosion_danger else '🟢 안정'}")
        b3.markdown(f"- 미세먼지 부하: {'🟠 경계' if high_pm_danger else '🟢 양호'}")

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
