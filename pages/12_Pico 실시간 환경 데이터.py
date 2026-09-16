import streamlit as st
import requests
import pandas as pd
from streamlit_autorefresh import st_autorefresh
from datetime import datetime, timedelta
import time

st.set_page_config(
    page_title="문화재 환경 모니터링",
    page_icon="🏛️",
    layout="wide"
)

st_autorefresh(
    interval=20 * 1000,
    key="sensor_refresh"
)

FIREBASE_SENSOR_URL = "https://heritage-project-4a361-default-rtdb.asia-southeast1.firebasedatabase.app/sensor.json"
FIREBASE_HISTORY_URL = "https://heritage-project-4a361-default-rtdb.asia-southeast1.firebasedatabase.app/sensor/history.json"


def to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def parse_time(timestamp):
    try:
        return datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
    except (TypeError, ValueError):
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


def metric_value(value, unit):
    if pd.isna(value):
        return "센서 확인"
    return f"{value:.1f} {unit}"


def stat_value(series, unit, mode="mean"):
    """
    선택 기간 통계 표시용.
    유효한 값이 하나도 없으면 '센서 확인'을 표시한다.
    """
    valid = pd.to_numeric(
        series,
        errors="coerce"
    ).dropna()

    if valid.empty:
        return "센서 확인"

    if mode == "mean":
        value = valid.mean()
    elif mode == "max":
        value = valid.max()
    elif mode == "min":
        value = valid.min()
    else:
        return "센서 확인"

    return f"{value:.1f} {unit}"


@st.cache_data(ttl=2)
def load_realtime_devices():
    try:
        response = requests.get(
            FIREBASE_SENSOR_URL,
            params={"t": time.time()},
            timeout=10
        )

        if response.status_code != 200:
            return {}

        sensor_data = response.json()

        if sensor_data is None:
            return {}

        devices = {}

        for key, value in sensor_data.items():
            if key.startswith("realtime_device_"):
                devices[key] = value

        return devices

    except Exception as e:
        st.warning(
            f"Firebase 실시간 데이터 조회 실패: {e}"
        )
        return {}


@st.cache_data(ttl=20)
def load_history_data():
    try:
        response = requests.get(
            FIREBASE_HISTORY_URL,
            params={"t": time.time()},
            timeout=10
        )

        if response.status_code != 200:
            return pd.DataFrame()

        history_data = response.json()

        if history_data is None:
            return pd.DataFrame()

        df = pd.DataFrame(history_data).T
        df.reset_index(drop=True, inplace=True)

        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(
                df["timestamp"],
                errors="coerce"
            )

        numeric_cols = [
            "temperature",
            "humidity",
            "pressure",
            "light",
            "pm1",
            "pm25",
            "pm10"
        ]

        for col in numeric_cols:
            if col not in df.columns:
                df[col] = pd.NA

            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )

        df = df.dropna(subset=["timestamp"])
        df = df.sort_values("timestamp")

        return df

    except Exception as e:
        st.warning(
            f"Firebase 이력 데이터 조회 실패: {e}"
        )
        return pd.DataFrame()


st.title("🏛️ 문화재 실시간 환경 모니터링 (20초마다 센서 측정)")
st.subheader("온습도기압 센서 BME280 / 조도 센서 BH1750 / 미세먼지 센서 PMS7003")

realtime_devices = load_realtime_devices()

if not realtime_devices:
    st.warning("Firebase에 저장된 실시간 장치 데이터가 없습니다.")

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
        st.success(f"🆕 {', '.join(new_devices)} 데이터 업데이트")

    st.divider()

    for device_key, data in sorted(realtime_devices.items()):

        temp = to_float(data.get("temperature"))
        hum = to_float(data.get("humidity"))
        pressure = to_float(data.get("pressure"))
        light = to_float(data.get("light"))

        pm1 = to_float(data.get("pm1"))
        pm25 = to_float(data.get("pm25"))
        pm10 = to_float(data.get("pm10"))

        timestamp = data.get("timestamp", "-")
        device = data.get("device", device_key)
        status = get_device_status(timestamp)

        title_col1, title_col2 = st.columns([3, 1])

        with title_col1:
            st.subheader(f"📡 {device}")

        with title_col2:
            st.markdown(f"### {status}")

        st.caption(f"마지막 측정 : {timestamp}")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "🌡️ 기온",
                metric_value(temp, "℃")
            )
            
            #st.metric("🌡️ 기온", f"{temp:.1f} ℃")

        with col2:
            st.metric(
                "💧 습도",
                metric_value(hum, "%")
            )
            #st.metric("💧 습도", f"{hum:.1f} %")

        with col3:
            st.metric(
                "🌬️ 기압",
                metric_value(pressure, "hPa")
            )

        with col4:
            st.metric(
                "☀️ 조도",
                metric_value(light, "lux")
            )

        col5, col6, col7, col8 = st.columns(4)

        with col5:
            st.metric("🌫️ PM1.0", metric_value(pm1, "㎍/㎥"))

        with col6:
            st.metric("🌫️ PM2.5", metric_value(pm25, "㎍/㎥"))

        with col7:
            st.metric("🌫️ PM10", metric_value(pm10, "㎍/㎥"))

        with col8:
            st.empty()

        st.divider()


st.subheader("📊 센서 데이터 이력 통계")

history_df = load_history_data()

if history_df.empty:
    st.info("아직 sensor/history에 누적된 데이터가 없습니다.")

else:
    st.caption(f"전체 누적 데이터 수 : {len(history_df)}개")

    history_df["date"] = history_df["timestamp"].dt.date

    min_date = history_df["date"].min()
    max_date = history_df["date"].max()

    if "device" in history_df.columns:
        device_list = sorted(history_df["device"].dropna().unique())

        selected_devices = st.multiselect(
            "조회할 장치 선택",
            device_list,
            default=device_list
        )

        history_df = history_df[
            history_df["device"].isin(selected_devices)
        ]

    st.markdown("#### 📅 조회 기간 선택")

    date_col1, date_col2 = st.columns(2)

    with date_col1:
        start_date = st.date_input(
            "시작 날짜",
            value=min_date,
            min_value=min_date,
            max_value=max_date
        )

    with date_col2:
        end_date = st.date_input(
            "종료 날짜",
            value=max_date,
            min_value=min_date,
            max_value=max_date
        )

    if start_date > end_date:
        st.error("시작 날짜가 종료 날짜보다 늦을 수 없습니다.")
        st.stop()

    filtered_df = history_df[
        (history_df["date"] >= start_date)
        &
        (history_df["date"] <= end_date)
    ]

    if filtered_df.empty:
        st.warning("선택한 조건에 해당하는 데이터가 없습니다.")
        st.stop()

    st.caption(
        f"선택 데이터 수 : {len(filtered_df)}개 "
        f"({start_date} ~ {end_date})"
    )

    st.markdown("#### 선택 기간 평균값")

    stat_col1, stat_col2, stat_col3, stat_col4, stat_col5 = st.columns(5)

    with stat_col1:
        st.metric("평균 기온", stat_value(filtered_df["temperature"], "℃", "mean"))

    with stat_col2:
        st.metric("평균 습도", stat_value(filtered_df["humidity"], "%", "mean"))

    with stat_col3:
        st.metric("평균 기압", stat_value(filtered_df["pressure"], "hPa", "mean"))

    with stat_col4:
        st.metric("평균 조도", stat_value(filtered_df["light"], "lux", "mean"))

    with stat_col5:
        st.metric("평균 PM2.5", stat_value(filtered_df["pm25"], "㎍/㎥", "mean"))

    st.markdown("#### 선택 기간 최대값")

    max_col1, max_col2, max_col3, max_col4, max_col5 = st.columns(5)

    with max_col1:
        st.metric("최고 기온", stat_value(filtered_df["temperature"], "℃", "max"))

    with max_col2:
        st.metric("최고 습도", stat_value(filtered_df["humidity"], "%", "max"))

    with max_col3:
        st.metric("최고 기압", stat_value(filtered_df["pressure"], "hPa", "max"))

    with max_col4:
        st.metric("최고 조도", stat_value(filtered_df["light"], "lux", "max"))

    with max_col5:
        st.metric("최고 PM2.5", stat_value(filtered_df["pm25"], "㎍/㎥", "max"))

    st.markdown("#### 선택 기간 최소값")

    min_col1, min_col2, min_col3, min_col4, min_col5 = st.columns(5)

    with min_col1:
        st.metric("최저 기온", stat_value(filtered_df["temperature"], "℃", "min"))

    with min_col2:
        st.metric("최저 습도", stat_value(filtered_df["humidity"], "%", "min"))

    with min_col3:
        st.metric("최저 기압", stat_value(filtered_df["pressure"], "hPa", "min"))

    with min_col4:
        st.metric("최저 조도", stat_value(filtered_df["light"], "lux", "min"))

    with min_col5:
        st.metric("최저 PM2.5", stat_value(filtered_df["pm25"], "㎍/㎥", "min"))

    st.markdown("#### 선택 기간 데이터 변화")

    chart_cols = [
        "temperature",
        "humidity",
        "pressure",
        "light",
        "pm1",
        "pm25",
        "pm10"
    ]

    selected_cols = st.multiselect(
        "그래프로 표시할 항목",
        chart_cols,
        default=[
            "temperature",
            "humidity",
            "pressure",
            "pm25"
        ]
    )

    if selected_cols:
        st.line_chart(
            filtered_df,
            x="timestamp",
            y=selected_cols
        )

    if "device" in filtered_df.columns:
        st.markdown("#### 장치별 평균 비교")

        device_mean_df = (
            filtered_df
            .groupby("device")[
                [
                    "temperature",
                    "humidity",
                    "pressure",
                    "light",
                    "pm1",
                    "pm25",
                    "pm10"
                ]
            ]
            .mean()
            .reset_index()
        )

        st.dataframe(
            device_mean_df,
            use_container_width=True,
            hide_index=True
        )

    st.markdown("#### 항목별 기초 통계")

    stats_cols = [
        "temperature",
        "humidity",
        "pressure",
        "light",
        "pm1",
        "pm25",
        "pm10"
    ]

    stats = filtered_df[stats_cols].describe().T

    stats = stats[
        [
            "count",
            "mean",
            "min",
            "max"
        ]
    ]

    stats.columns = [
        "개수",
        "평균",
        "최솟값",
        "최댓값"
    ]

    st.dataframe(
        stats,
        use_container_width=True
    )

    st.markdown("#### 일별 평균 변화")

    daily_df = (
        filtered_df
        .groupby("date")[
            [
                "temperature",
                "humidity",
                "pressure",
                "light",
                "pm1",
                "pm25",
                "pm10"
            ]
        ]
        .mean()
        .reset_index()
    )

    daily_selected_cols = st.multiselect(
        "일별 평균으로 표시할 항목",
        chart_cols,
        default=[
            "temperature",
            "humidity",
            "pressure",
            "pm25"
        ],
        key="daily_chart_select"
    )

    if daily_selected_cols:
        st.line_chart(
            daily_df,
            x="date",
            y=daily_selected_cols
        )

    with st.expander("🧾 선택 기간 원본 데이터 보기"):
        st.dataframe(
            filtered_df.sort_values(
                "timestamp",
                ascending=False
            ),
            use_container_width=True,
            hide_index=True
        )

    csv = filtered_df.to_csv(index=False).encode("utf-8-sig")

    st.download_button(
        label="📥 선택 기간 센서 이력 CSV 다운로드",
        data=csv,
        file_name="firebase_sensor_history_filtered.csv",
        mime="text/csv"
    )
