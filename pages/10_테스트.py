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


# Plotly 게이지 차트 생성 함수
def create_gauge_chart(value, title, unit, max_val=100, color="#4F46E5"):
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            title={"text": title, "font": {"size": 14}},
            number={"suffix": f" {unit}", "font": {"size": 18}},
            gauge={
                "axis": {"range": [None, max_val], "tickwidth": 1, "tickcolor": "darkblue"},
                "bar": {"color": color},
                "bgcolor": "white",
                "borderwidth": 2,
                "bordercolor": "gray",
            },
        )
    )
    fig.update_layout(height=160, margin=dict(l=20, r=20, t=30, b=10))
    return fig


# ------------------------------------------------------------
# 3. 데이터 캐싱 로드
# ------------------------------------------------------------
@st.cache_resource
def load_ml_pipeline():
    try:
        df = pd.read_csv(DATA_PATH)
    except:
        return None

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    if "rainfall" in df.columns:
        df["rainfall"] = df["rainfall"].fillna(0)

    df["temp_range"] = df["temp_max"] - df["temp_min"]
    df["humidity_std3"] = df["humidity"].rolling(3, min_periods=1).std()
    df["rainfall_7d"] = df["rainfall"].rolling(7, min_periods=1).sum()
    df["high_humidity_risk"] = ((df["humidity"] >= 75).rolling(3, min_periods=1).sum())
    df["weathering_risk"] = (
        df["temp_range"] * 0.4 + df["humidity_std3"] * 0.3 + df["wind_speed"] * 0.3
    )
    df["mold_risk"] = ((df["humidity"] >= 75) & (df["ground_temp"] >= 15)).astype(int)
    df["pm_load"] = (df["pm10"] + df["pm25"]).rolling(3, min_periods=1).sum()
    df["acid_risk"] = df["so2"] * 0.6 + df["no2"] * 0.4
    df["oxidation_risk"] = df["o3"] * 0.7 + df["pm25"] * 0.3
    df["corrosion_risk"] = df["humidity"] * 0.5 + df["so2"] * 0.5
    df = df.fillna(0)
    return df


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
        return {k: v for k, v in sensor_data.items() if k.startswith("realtime_device_")}
    except:
        return {}


# ------------------------------------------------------------
# 4. 메인 대시보드 레이아웃
# ------------------------------------------------------------
st.title("🏛️ 문화재 실시간 통합 모니터링 & 스마트 위험 진단")
st.markdown("복수 센서(Pico)의 실시간 데이터를 융합·비교하고, 문화재 재질 및 노출 환경에 따른 위험도를 정밀 진단합니다.")

realtime_devices = load_realtime_devices()
dataset = load_ml_pipeline()

if not realtime_devices:
    st.warning("⚠️ Firebase에 저장된 실시간 장치 데이터가 없습니다.")
else:
    # ------------------------------------------------------------
    # 4-1. 개별 센서 데이터 시각화 (컬럼별 카드 및 게이지)
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
                st.markdown(f"#### 🏷️ {device_name}")
                st.caption(f"상태: {status} | 측정 시각: {timestamp}")
                
                # 미니 게이지 차트로 시각화 극대화
                st.plotly_chart(create_gauge_chart(temp, "기온", "℃", max_val=50, color="#EF4444"), use_container_width=True, key=f"t_{device_key}")
                st.plotly_chart(create_gauge_chart(hum, "습도", "%", max_val=100, color="#3B82F6"), use_container_width=True, key=f"h_{device_key}")
                st.plotly_chart(create_gauge_chart(pm25, "초미세먼지", "㎍/㎥", max_val=150, color="#10B981"), use_container_width=True, key=f"pm_{device_key}")

    # 통합 평균치 계산
    avg_temp = np.mean(temps) if temps else 0
    avg_hum = np.mean(hums) if hums else 0
    avg_pressure = np.mean(pressures) if pressures else 0
    avg_light = np.mean(lights) if lights else 0
    avg_pm1 = np.mean(pm1s) if pm1s else 0
    avg_pm25 = np.mean(pm25s) if pm25s else 0
    avg_pm10 = np.mean(pm10s) if pm10s else 0

    st.markdown("---")
    
    # ------------------------------------------------------------
    # 4-2. 센서 통합 대표 평균 지표 (시각화된 메트릭 대시보드)
    # ------------------------------------------------------------
    st.markdown("### 📊 센서 통합 대표 평균 지표 (복수 센서 융합)")
    
    ic1, ic2, ic3, ic4 = st.columns(4)
    with ic1:
        st.metric("🌡️ 통합 평균 기온", f"{avg_temp:.1f} ℃", delta=f"{avg_temp - 20:.1f}℃ (기준 대비)")
    with ic2:
        st.metric("💧 통합 평균 습도", f"{avg_hum:.1f} %", delta=f"{avg_hum - 50:.1f}% (적정 대비)")
    with ic3:
        st.metric("🌬️ 통합 평균 기압", f"{avg_pressure:.1f} hPa")
    with ic4:
        st.metric("☀️ 통합 평균 조도", f"{avg_light:.1f} lux")

    ic5, ic6, ic7, ic8 = st.columns(4)
    with ic5:
        st.metric("🌫️ 통합 PM1.0", f"{avg_pm1:.1f} ㎍/㎥")
    with ic6:
        st.metric("🌫️ 통합 PM2.5", f"{avg_pm25:.1f} ㎍/㎥")
    with ic7:
        st.metric("🌫️ 통합 PM10", f"{avg_pm10:.1f} ㎍/㎥")
    with ic8:
        st.metric("📡 연동된 센서 총 대수", f"{len(device_items)} 대 정상 작동")

    st.markdown("---")

    # ------------------------------------------------------------
    # 4-3. 재질별/노출환경별 위험 진단 카드 UI (인터랙티브 디자인)
    # ------------------------------------------------------------
    st.markdown("### 🛡️ 문화재 재질 및 노출 환경별 스마트 맞춤 위험 진단")
    
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

    # 환경 가중치 부여 종합 위험 점수 산출 (0~100 스케일)
    base_score = (avg_hum * 0.4) + (avg_pm25 * 0.3) + (abs(avg_temp - 20) * 3 * 0.3)
    exposure_multiplier = 1.3 if eval_exposure == "실외" else (1.1 if eval_exposure == "반실외" else 0.85)
    material_multiplier = 1.2 if eval_material in ["목조", "금속"] else 1.0
    total_risk_score = min(max(base_score * exposure_multiplier * material_multiplier * 0.6, 5), 98)

    # 위험 등급 분류
    if total_risk_score >= 75 or (eval_material == "목조" and mold_danger) or (eval_material == "금속" and corrosion_danger):
        risk_level = "위험 (Danger)"
        card_border_color = "red"
        advice_text = "현재 온습도 및 환경 조건이 문화재 손상 유발 임계치를 초과했습니다. 즉각적인 환기, 제습 및 보존 처리 조치가 필요합니다."
    elif total_risk_score >= 45 or high_pm_danger or avg_hum >= 70:
        risk_level = "주의 (Caution)"
        card_border_color = "orange"
        advice_text = "미세먼지 농도 혹은 습도가 다소 높은 상태입니다. 변형 및 오염 방지를 위해 지속적인 모니터링과 국부 환경 조절을 검토하세요."
    else:
        risk_level = "안전 (Safe)"
        card_border_color = "green"
        advice_text = "현재 수집된 통합 센서 환경 조건은 선택하신 문화재 재질과 배치 환경에 매우 안정적이고 적합한 상태입니다."

    # 고급 시각화 카드 출력
    with st.container(border=True):
        st.markdown(f"### 📋 맞춤형 진단 결과 리포트")
        st.markdown(f"**진단 대상:** `재질 - {eval_material}` / `환경 - {eval_exposure}`")
        
        col_res1, col_res2, col_res3 = st.columns([1, 1, 2])
        
        with col_res1:
            if "위험" in risk_level:
                st.error(f"**판정 등급**\n\n### 🚨 {risk_level}")
            elif "주의" in risk_level:
                st.warning(f"**판정 등급**\n\n### ⚠️ {risk_level}")
            else:
                st.success(f"**판정 등급**\n\n### ✅ {risk_level}")
                
        with col_res2:
            st.metric("산출된 위험 지수", f"{total_risk_score:.1f} 점", delta=f"{total_risk_score - 50:.1f} (위험기준선 대비)")
            
        with col_res3:
            st.markdown(f"**💡 핵심 관리 가이드**")
            st.info(advice_text)
        
        st.markdown("#### 🔍 세부 환경 취약 요소 분석")
        b1, b2, b3 = st.columns(3)
        with b1:
            st.markdown(f"• **곰팡이 활성 위험 (목조):** {'🔴 높음 (주의)' if mold_danger else '🟢 안정'}")
        with b2:
            st.markdown(f"• **산화/부식 위험 (금속):** {'🔴 높음 (주의)' if corrosion_danger else '🟢 안정'}")
        with b3:
            st.markdown(f"• **입자상 물질 오염 (미세먼지):** {'🟠 경계' if high_pm_danger else '🟢 양호'}")

    st.divider()
