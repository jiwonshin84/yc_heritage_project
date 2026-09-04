
import streamlit as st
import pandas as pd
import requests
import urllib.parse
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

# ==========================================================
# 0. 기본 설정
# ==========================================================

st.set_page_config(
    page_title="영천시 최근 7일 대기오염 실시간 모니터링",
    page_icon="🌫",
    layout="wide"
)

# API 인증키 설정
SERVICE_KEY = st.secrets.get("SERVICE_KEY", "feb2bfabd299d5d05e89c7aec49ba7e706112603e76549a92e868bd86ec60323")

# 날짜 범위 설정: 어제(end_date)부터 7일 전(start_date)까지
now = datetime.now(ZoneInfo("Asia/Seoul"))
end_date_dt = now - timedelta(days=1)
start_date_dt = now - timedelta(days=7)

st.title("🌫 영천시 최근 7일 대기오염 실시간 데이터")
st.caption(f"조회 기간: {start_date_dt.strftime('%Y-%m-%d')} ~ {end_date_dt.strftime('%Y-%m-%d')} (실시간 측정정보 기반)")
st.divider()

# ==========================================================
# 1. 실시간 대기오염 API 호출 및 기간 필터링
# ==========================================================

air_list = []
try:
    air_url = "http://apis.data.go.kr/B552584/ArpltnInforInqireSvc/getMsrstnAcctoRltmDnsty"
    safe_service_key = urllib.parse.unquote(SERVICE_KEY)

    air_params = {
        "serviceKey": safe_service_key,
        "returnType": "json",
        "numOfRows": "200",   # 7일 치 시간별 데이터(약 168시간)를 확보하기 넉넉하게 설정
        "pageNo": "1",
        "stationName": "영천", # 영천 측정소
        "dataTerm": "MONTH",   # 기간 넉넉히 조회 후 파이썬에서 필터링
        "ver": "1.3"
    }

    response = requests.get(air_url, params=air_params, timeout=15)

    if response.status_code == 200 and response.text.strip().startswith("{"):
        data = response.json()
        items = data.get("response", {}).get("body", {}).get("items", [])

        for item in items:
            raw_time = item.get("dataTime", "") # 예: "2026-09-03 14:00"
            if raw_time:
                raw_date_str = raw_time.split(" ")[0]
                item_date = pd.to_datetime(raw_date_str)

                # 정확히 [어제 ~ 7일 전] 범위에 포함되는 데이터만 수집
                if start_date_dt.date() <= item_date.date() <= end_date_dt.date():
                    air_list.append({
                        "date": item_date,
                        "pm10": float(item.get("pm10Value", 0) or 0),
                        "pm25": float(item.get("pm25Value", 0) or 0),
                        "o3": float(item.get("o3Value", 0) or 0),
                        "no2": float(item.get("no2Value", 0) or 0),
                        "co": float(item.get("coValue", 0) or 0),
                        "so2": float(item.get("so2Value", 0) or 0)
                    })
except Exception as e:
    st.error(f"대기오염 데이터 수집 중 오류 발생: {e}")

# ==========================================================
# 2. 데이터프레임 정제 및 날짜별 평균 집계
# ==========================================================

if air_list:
    df = pd.DataFrame(air_list)

    # 시간 단위 데이터를 날짜별 평균(Daily Mean)으로 그룹화
    df_daily = df.groupby("date", as_index=False).mean(numeric_only=True)
    df_daily = df_daily.sort_values("date").reset_index(drop=True)

    # 화면 출력을 위해 날짜 형식을 문자열로 변환
    df_daily["date"] = df_daily["date"].dt.strftime("%Y-%m-%d")

    st.success(f"성공적으로 {len(df_daily)}일 치 데이터를 불러왔습니다.")
    
    # 데이터프레임 출력
    st.dataframe(
        df_daily[["date", "pm10", "pm25", "o3", "no2", "co", "so2"]],
        use_container_width=True,
        hide_index=True
    )
else:
    st.warning("조건에 해당하는 대기오염 데이터가 없습니다. 네트워크 상태나 API 키를 확인해주세요.")
