from datetime import datetime, timedelta
import time

import numpy as np
import pandas as pd
import requests
import streamlit as st
from streamlit_autorefresh import st_autorefresh


# ------------------------------------------------------------
# 1. 페이지 설정 및 실시간 리프레시
#    Pico W 측정 주기(약 20초)에 맞춰 20초마다 새로고침
# ------------------------------------------------------------
st.set_page_config(
    page_title="문화유산 실시간 환경 모니터링 & 규칙 기반 위험 진단",
    page_icon="🏛️",
    layout="wide",
)

REFRESH_SECONDS = 20
st_autorefresh(interval=REFRESH_SECONDS * 1000, key="sensor_refresh")

FIREBASE_SENSOR_URL = (
    "https://heritage-project-4a361-default-rtdb.asia-southeast1."
    "firebasedatabase.app/sensor.json"
)


# ------------------------------------------------------------
# 2. 유틸리티 함수
# ------------------------------------------------------------
def to_float(value):
    """센서값을 실수로 변환한다. 변환 실패/비정상 값은 0이 아니라 NaN으로 처리한다."""
    try:
        if value is None:
            return np.nan
        number = float(value)
        return number if np.isfinite(number) else np.nan
    except (TypeError, ValueError):
        return np.nan


def valid_range(value, min_value=None, max_value=None, zero_is_missing=False):
    """센서별 물리적 범위를 벗어난 값을 NaN으로 처리한다."""
    value = to_float(value)
    if pd.isna(value):
        return np.nan
    if zero_is_missing and value == 0:
        return np.nan
    if min_value is not None and value < min_value:
        return np.nan
    if max_value is not None and value > max_value:
        return np.nan
    return value


def parse_time(timestamp):
    if not timestamp:
        return None

    text = str(timestamp).strip()
    formats = (
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%S.%f",
    )

    for fmt in formats:
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            pass

    return None


def get_device_status(timestamp):
    dt = parse_time(timestamp)
    if dt is None:
        return "⚠️ 시간 오류"

    diff = datetime.now() - dt
    if diff > timedelta(minutes=10):
        return "🔴 수신 지연"
    if diff > timedelta(minutes=2):
        return "🟠 확인 필요"
    return "🟢 정상 수신"


def safe_mean(values):
    """NaN을 제외한 평균. 유효값이 없으면 NaN을 반환한다."""
    valid = [v for v in values if not pd.isna(v)]
    return float(np.mean(valid)) if valid else np.nan


def format_metric(value, unit="", digits=1):
    if pd.isna(value):
        return "측정값 없음"
    return f"{value:.{digits}f} {unit}".strip()


# ------------------------------------------------------------
# 3. Firebase 실시간 데이터 로드
# ------------------------------------------------------------
@st.cache_data(ttl=REFRESH_SECONDS)
def load_realtime_devices():
    try:
        response = requests.get(
            FIREBASE_SENSOR_URL,
            params={"t": time.time()},
            timeout=10,
        )
        response.raise_for_status()
        sensor_data = response.json()

        if not isinstance(sensor_data, dict):
            return {}

        return {
            key: value
            for key, value in sensor_data.items()
            if key.startswith("realtime_device_") and isinstance(value, dict)
        }
    except (requests.RequestException, ValueError):
        return {}


# ------------------------------------------------------------
# 4. 메인 대시보드
# ------------------------------------------------------------
st.title("🏛️ 문화유산 실시간 환경 모니터링 & 규칙 기반 위험 진단")
st.caption(
    "Pico W 센서의 실시간 환경값을 통합하여 재질과 노출환경에 따른 상태를 규칙 기반으로 진단합니다. "
    "이 페이지의 위험지수는 머신러닝 예측값이 아니라 실시간 모니터링을 위한 별도의 규칙 기반 지표입니다."
)
st.info(
    "📌 머신러닝 기반 문화유산 환경 취약도 예측은 '실측 데이터 기반 취약도 예측' 페이지에서 수행합니다. "
    "현재 페이지는 센서값의 즉시 확인과 현장 환경 상태 점검을 위한 보조 진단 화면입니다."
)

realtime_devices = load_realtime_devices()

if not realtime_devices:
    st.warning("⚠️ Firebase에 저장된 실시간 장치 데이터가 없습니다.")
    st.stop()


# ------------------------------------------------------------
# 5. 센서별 독립적인 유효값 처리
# ------------------------------------------------------------
valid_temps = []
valid_hums = []
valid_pressures = []
valid_lights = []
valid_pm1s = []
valid_pm25s = []
valid_pm10s = []

device_rows = []

for device_key, data in sorted(realtime_devices.items()):
    # 기온 0℃는 실제 환경에서 가능하므로 0 자체를 오류로 보지 않는다.
    temp = valid_range(data.get("temperature"), -40, 85)
    hum = valid_range(data.get("humidity"), 0, 100)

    # BME280 기압은 일반적인 관측 범위 밖이면 결측 처리
    pressure = valid_range(data.get("pressure"), 300, 1100, zero_is_missing=True)

    # 조도/미세먼지는 실제 0이 가능하므로 음수만 결측 처리
    light = valid_range(data.get("light"), 0, None)
    pm1 = valid_range(data.get("pm1"), 0, 1000)
    pm25 = valid_range(data.get("pm25"), 0, 500)
    pm10 = valid_range(data.get("pm10"), 0, 1000)

    values = [temp, hum, pressure, light, pm1, pm25, pm10]
    has_any_valid_value = any(not pd.isna(v) for v in values)

    if not has_any_valid_value:
        continue

    if not pd.isna(temp):
        valid_temps.append(temp)
    if not pd.isna(hum):
        valid_hums.append(hum)
    if not pd.isna(pressure):
        valid_pressures.append(pressure)
    if not pd.isna(light):
        valid_lights.append(light)
    if not pd.isna(pm1):
        valid_pm1s.append(pm1)
    if not pd.isna(pm25):
        valid_pm25s.append(pm25)
    if not pd.isna(pm10):
        valid_pm10s.append(pm10)

    device_rows.append(
        {
            "장치": device_key.replace("realtime_", ""),
            "수신 상태": get_device_status(data.get("timestamp")),
            "측정 시각": data.get("timestamp", "-"),
            "기온(℃)": temp,
            "습도(%)": hum,
            "기압(hPa)": pressure,
            "조도(lux)": light,
            "PM1.0(㎍/㎥)": pm1,
            "PM2.5(㎍/㎥)": pm25,
            "PM10(㎍/㎥)": pm10,
        }
    )

active_device_count = len(device_rows)

if active_device_count == 0:
    st.error("🚨 현재 유효한 센서값이 없습니다. Pico W 및 센서 연결 상태를 확인해주세요.")
    st.stop()

avg_temp = safe_mean(valid_temps)
avg_hum = safe_mean(valid_hums)
avg_pressure = safe_mean(valid_pressures)
avg_light = safe_mean(valid_lights)
avg_pm1 = safe_mean(valid_pm1s)
avg_pm25 = safe_mean(valid_pm25s)
avg_pm10 = safe_mean(valid_pm10s)


# ------------------------------------------------------------
# 6. 센서 대표 실시간 모니터링 지표
# ------------------------------------------------------------
st.markdown(
    f"### 📊 센서 대표 실시간 모니터링 지표 "
    f"(유효 데이터 장치 {active_device_count}대 반영)"
)

ic1, ic2, ic3, ic4 = st.columns(4)
with ic1:
    delta = None if pd.isna(avg_temp) else f"{avg_temp - 20:.1f}℃ (20℃ 대비)"
    st.metric("🌡️ 대표 기온", format_metric(avg_temp, "℃"), delta=delta)
with ic2:
    delta = None if pd.isna(avg_hum) else f"{avg_hum - 50:.1f}% (50% 대비)"
    st.metric("💧 대표 습도", format_metric(avg_hum, "%"), delta=delta)
with ic3:
    st.metric("🌬️ 대표 기압", format_metric(avg_pressure, "hPa"))
with ic4:
    st.metric("☀️ 대표 조도", format_metric(avg_light, "lux"))

ic5, ic6, ic7, ic8 = st.columns(4)
with ic5:
    st.metric("🌫️ 대표 PM1.0", format_metric(avg_pm1, "㎍/㎥"))
with ic6:
    st.metric("🌫️ 대표 PM2.5", format_metric(avg_pm25, "㎍/㎥"))
with ic7:
    st.metric("🌫️ 대표 PM10", format_metric(avg_pm10, "㎍/㎥"))
with ic8:
    st.metric("📡 유효 데이터 장치", f"{active_device_count} 대")

with st.expander("🔎 장치별 실시간 측정값 확인", expanded=False):
    device_df = pd.DataFrame(device_rows)
    st.dataframe(device_df, use_container_width=True, hide_index=True)
    st.caption("※ 센서별 결측값은 0으로 대체하지 않고 표시에서 비워 두며 대표 평균에서도 제외합니다.")

st.markdown("---")


# ------------------------------------------------------------
# 7. 재질별/노출환경별 규칙 기반 위험 진단
# ------------------------------------------------------------
st.markdown("### 🛡️ 문화유산 재질 및 노출환경별 규칙 기반 환경 진단")

card_col1, card_col2 = st.columns(2)
with card_col1:
    eval_material = st.selectbox(
        "진단 대상 문화유산 재질 선택",
        ["목조", "석조", "금속", "회화", "기타"],
    )
with card_col2:
    eval_exposure = st.selectbox(
        "배치 노출 환경 선택",
        ["실외", "반실외", "실내"],
    )

# 기존 규칙 기반 산식을 유지하되, 계산에 필요한 핵심 센서값이 없으면
# 0으로 임의 대체하지 않고 진단을 중단한다.
missing_core = []
if pd.isna(avg_temp):
    missing_core.append("기온")
if pd.isna(avg_hum):
    missing_core.append("습도")
if pd.isna(avg_pm25):
    missing_core.append("PM2.5")

if missing_core:
    st.warning(
        "⚠️ 규칙 기반 진단에 필요한 핵심 센서값이 부족합니다: "
        + ", ".join(missing_core)
        + ". 결측값을 0으로 간주하지 않으며, 센서값이 정상 수신되면 자동으로 다시 진단합니다."
    )
    st.stop()

# 기존 실시간 규칙
mold_danger = (avg_hum >= 75) and (avg_temp >= 15)
corrosion_danger = avg_hum >= 80
high_pm_danger = avg_pm25 >= 50

# 기존 환경 가중치 기반 보조 위험지수(0~100 범위)
base_score = (
    (avg_hum * 0.4)
    + (avg_pm25 * 0.3)
    + (abs(avg_temp - 20) * 3 * 0.3)
)

exposure_multiplier = {
    "실외": 1.3,
    "반실외": 1.1,
    "실내": 0.85,
}[eval_exposure]

material_multiplier = 1.2 if eval_material in ["목조", "금속"] else 1.0

total_risk_score = np.clip(
    base_score * exposure_multiplier * material_multiplier * 0.6,
    5,
    98,
)

# 규칙 기반 등급 분류
if (
    total_risk_score >= 75
    or (eval_material == "목조" and mold_danger)
    or (eval_material == "금속" and corrosion_danger)
):
    risk_level = "위험 (Danger)"
    advice_text = (
        "현재 센서 환경조건이 프로젝트의 실시간 진단 기준에서 높은 수준으로 나타났습니다. "
        "현장 상태를 확인하고 환기·제습 등 예방적 관리 필요성을 검토하세요."
    )
elif total_risk_score >= 45 or high_pm_danger or avg_hum >= 70:
    risk_level = "주의 (Caution)"
    advice_text = (
        "현재 미세먼지 또는 습도 등의 환경조건이 주의 수준에 해당합니다. "
        "센서값의 지속 여부를 확인하고 환경 변화를 계속 모니터링하세요."
    )
else:
    risk_level = "안전 (Safe)"
    advice_text = (
        "현재 수집된 센서값은 프로젝트의 실시간 규칙 기준에서 안전 수준으로 분류됩니다. "
        "다만 이는 실제 문화유산의 훼손 여부를 의미하지 않습니다."
    )


# ------------------------------------------------------------
# 8. 진단 결과 카드
# ------------------------------------------------------------
with st.container(border=True):
    st.markdown("### 📋 규칙 기반 실시간 진단 결과")
    st.markdown(
        f"**진단 대상:** `재질 - {eval_material}` / `환경 - {eval_exposure}`"
    )

    col_res1, col_res2, col_res3 = st.columns([1, 1, 2])

    with col_res1:
        if "위험" in risk_level:
            st.error(f"**판정 등급**\n\n### 🚨 {risk_level}")
        elif "주의" in risk_level:
            st.warning(f"**판정 등급**\n\n### ⚠️ {risk_level}")
        else:
            st.success(f"**판정 등급**\n\n### ✅ {risk_level}")

    with col_res2:
        st.metric(
            "규칙 기반 환경 위험지수",
            f"{total_risk_score:.1f} 점",
            delta=f"{total_risk_score - 50:.1f} (50점 대비)",
        )

    with col_res3:
        st.markdown("**💡 핵심 관리 가이드**")
        st.info(advice_text)

    st.markdown("#### 🔍 세부 환경 취약 요소")
    b1, b2, b3 = st.columns(3)
    with b1:
        st.markdown(
            f"• **곰팡이 활성 조건(목조 참고):** "
            f"{'🔴 해당' if mold_danger else '🟢 비해당'}"
        )
    with b2:
        st.markdown(
            f"• **고습 부식 조건(금속 참고):** "
            f"{'🔴 해당' if corrosion_danger else '🟢 비해당'}"
        )
    with b3:
        st.markdown(
            f"• **PM2.5 고농도 조건:** "
            f"{'🟠 해당' if high_pm_danger else '🟢 비해당'}"
        )

st.caption(
    "※ 이 페이지의 등급과 위험지수는 실시간 센서값에 적용한 프로젝트 내부 규칙 기반 보조지표입니다. "
    "실제 문화유산의 훼손 발생 여부 또는 공식 보존등급을 의미하지 않습니다."
)
