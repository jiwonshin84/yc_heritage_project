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
st.markdown("활성 상태인 센서 데이터를 연동·필터링하여 문화재 재질 및 노출 환경별 위험도를 정밀 진단합니다.")

realtime_devices = load_realtime_devices()
dataset = load_ml_pipeline()

if not realtime_devices:
    st.warning("⚠️ Firebase에 저장된 실시간 장치 데이터가 없습니다.")
else:
    # ------------------------------------------------------------
    # 4-1. 유효 센서 필터링 (기온이 0인 비정상 데이터 제외)
    # ------------------------------------------------------------
    valid_temps, valid_hums, valid_pressures, valid_lights = [], [], [], []
    valid_pm1s, valid_pm25s, valid_pm10s = [], [], []
    active_device_count = 0

    for device_key, data in sorted(realtime_devices.items()):
        temp = to_float(data.get("temperature", 0))
        hum = to_float(data.get("humidity", 0))
        
        # 기온이 0인 경우 데이터 미수신/오류 상태로 간주하여 연산에서 제외
        if temp == 0.0:
            continue

        active_device_count += 1
        valid_temps.append(temp)
        valid_hums.append(hum)
        valid_pressures.append(to_float(data.get("pressure", 0)))
        valid_lights.append(to_float(data.get("light", 0)))
        valid_pm1s.append(to_float(data.get("pm1", 0)))
        valid_pm25s.append(to_float(data.get("pm25", 0)))
        valid_pm10s.append(to_float(data.get("pm10", 0)))

    if active_device_count == 0:
        st.error("🚨 현재 유효한 센서 데이터(기온 데이터 정상 유입)가 존재하지 않습니다. 센서 연결 상태를 확인해주세요.")
        st.stop()

    # 통합(또는 단독 유효 센서) 대표 값 산출
    avg_temp = np.mean(valid_temps)
    avg_hum = np.mean(valid_hums)
    avg_pressure = np.mean(valid_pressures)
    avg_light = np.mean(valid_lights)
    avg_pm1 = np.mean(valid_pm1s)
    avg_pm25 = np.mean(valid_pm25s)
    avg_pm10 = np.mean(valid_pm10s)

    # ------------------------------------------------------------
    # 4-2. 센서 통합 대표 평균 지표 (유효 센서 기반)
    # ------------------------------------------------------------
    st.markdown(f"### 📊 센서 대표 실시간 모니터링 지표 (유효 센서 {active_device_count}대 반영)")
    
    ic1, ic2, ic3, ic4 = st.columns(4)
    with ic1:
        st.metric("🌡️ 대표 기온", f"{avg_temp:.1f} ℃", delta=f"{avg_temp - 20:.1f}℃ (기준 대비)")
    with ic2:
        st.metric("💧 대표 습도", f"{avg_hum:.1f} %", delta=f"{avg_hum - 50:.1f}% (적정 대비)")
    with ic3:
        st.metric("🌬️ 대표 기압", f"{avg_pressure:.1f} hPa")
    with ic4:
        st.metric("☀️ 대표 조도", f"{avg_light:.1f} lux")

    ic5, ic6, ic7, ic8 = st.columns(4)
    with ic5:
        st.metric("🌫️ 대표 PM1.0", f"{avg_pm1:.1f} ㎍/㎥")
    with ic6:
        st.metric("🌫️ 대표 PM2.5", f"{avg_pm25:.1f} ㎍/㎥")
    with ic7:
        st.metric("🌫️ 대표 PM10", f"{avg_pm10:.1f} ㎍/㎥")
    with ic8:
        st.metric("📡 정상 연동 센서", f"{active_device_count} 대")

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

    # 룰 기반 정밀 진단 로직 (유효 대표값 기준)
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
        advice_text = "현재 온습도 및 환경 조건이 문화재 손상 유발 임계치를 초과했습니다. 즉각적인 환기, 제습 및 보존 처리 조치가 필요합니다."
    elif total_risk_score >= 45 or high_pm_danger or avg_hum >= 70:
        risk_level = "주의 (Caution)"
        advice_text = "미세먼지 농도 혹은 습도가 다소 높은 상태입니다. 변형 및 오염 방지를 위해 지속적인 모니터링과 국부 환경 조절을 검토하세요."
    else:
        risk_level = "안전 (Safe)"
        advice_text = "현재 수집된 센서 환경 조건은 선택하신 문화재 재질과 배치 환경에 매우 안정적이고 적합한 상태입니다."

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
