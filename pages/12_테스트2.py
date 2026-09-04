import streamlit as st
import pandas as pd
import requests
import urllib.parse
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

# ==========================================================
# 0. 설정
# ==========================================================

st.set_page_config(
    page_title="영천 지역 대기환경 모니터링",
    page_icon="🌫",
    layout="wide"
)

SERVICE_KEY = st.secrets.get("SERVICE_KEY", "feb2bfabd299d5d05e89c7aec49ba7e706112603e76549a92e868bd86ec60323")

# ==========================================================
# 1. 대기오염 데이터 수집 (실시간 측정정보 API)
# ==========================================================

air_list = []
try:
    air_url = "http://apis.data.go.kr/B552584/ArpltnInforInqireSvc/getMsrstnAcctoRltmDnsty"
    safe_service_key = urllib.parse.unquote(SERVICE_KEY)

    air_params = {
        "serviceKey": safe_service_key,
        "returnType": "json",
        "numOfRows": "100",
        "pageNo": "1",
        "stationName": "영천",
        "dataTerm": "DAILY",
        "ver": "1.3"
    }

    air_response = requests.get(air_url, params=air_params, timeout=15)

    if air_response.status_code == 200 and air_response.text.strip().startswith("{"):
        air_data = air_response.json()
        items = air_data.get("response", {}).get("body", {}).get("items", [])

        for item in items:
            raw_time = item.get("dataTime", "")
            if raw_time:
                raw_date = raw_time.split(" ")[0]
                air_list.append({
                    "date": pd.to_datetime(raw_date),
                    "pm10": float(item.get("pm10Value", 0) or 0),
                    "pm25": float(item.get("pm25Value", 0) or 0),
                    "o3": float(item.get("o3Value", 0) or 0),
                    "no2": float(item.get("no2Value", 0) or 0),
                    "co": float(item.get("coValue", 0) or 0),
                    "so2": float(item.get("so2Value", 0) or 0)
                })
except Exception as e:
    st.warning(f"대기오염 실시간 데이터 수집 실패: {e}")

# ==========================================================
# 2. 데이터프레임 정제 및 집계
# ==========================================================

a_df_curr = pd.DataFrame(air_list)

if a_df_curr.empty:
    st.error("대기오염 데이터를 불러오지 못했습니다. API 키나 네트워크 상태를 확인해주세요.")
    st.stop()

# 날짜별 평균 집계
a_df_curr = a_df_curr.sort_values("date").groupby("date", as_index=False).mean(numeric_only=True)
a_df_curr = a_df_curr.drop_duplicates(subset=["date"], keep="last")
a_df_curr = a_df_curr.sort_values("date").reset_index(drop=True)

# 가장 최신 기준일 데이터
target_row = a_df_curr.iloc[-1]
tm = target_row["date"].strftime("%Y-%m-%d")

# ==========================================================
# 3. 화면 UI 구성
# ==========================================================

st.title("🌫 영천 지역 대기오염 모니터링")
st.caption(f"조회 기준일자: {tm} (실시간 집계)")
st.divider()

# 최신 대기오염 지표 요약
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("미세먼지 (PM10)", f"{target_row.get('pm10', 0):.0f} ㎍/㎥")
    st.metric("초미세먼지 (PM2.5)", f"{target_row.get('pm25', 0):.0f} ㎍/㎥")

with col2:
    st.metric("오존 (O₃)", f"{target_row.get('o3', 0):.3f} ppm")
    st.metric("이산화질소 (NO₂)", f"{target_row.get('no2', 0):.3f} ppm")

with col3:
    st.metric("일산화탄소 (CO)", f"{target_row.get('co', 0):.1f} ppm")
    st.metric("아황산가스 (SO₂)", f"{target_row.get('so2', 0):.3f} ppm")

st.divider()
st.subheader("📅 최근 대기환경 데이터 목록")

# 날짜 형식 정리용 복사본
display_df = a_df_curr.tail(7).copy()
display_df["date"] = display_df["date"].dt.strftime("%Y-%m-%d")

st.dataframe(
    display_df[["date", "pm10", "pm25", "o3", "no2", "co", "so2"]],
    use_container_width=True,
    hide_index=True
)
