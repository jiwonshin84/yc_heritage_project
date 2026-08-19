import os
import time
from datetime import datetime
import pandas as pd
import requests

# ------------------------------------------------------------
# 1. 동적 연도 계산 및 저장 경로 설정
# ------------------------------------------------------------
current_year = datetime.now().year
end_year = current_year - 1
start_year = end_year - 9  # 최근 10년치 (예: 2026년 기준 2016~2025)

# data/processed/ 디렉터리가 없으면 자동 생성
output_dir = os.path.join("data", "processed")
os.makedirs(output_dir, exist_ok=True)

# 저장 파일명 정의 (예: data/processed/[2016_2025] yeongcheon_data.csv)
file_name = f"[{start_year}_{end_year}] yeongcheon_data.csv"
save_path = os.path.join(output_dir, file_name)

# ------------------------------------------------------------
# 2. API 설정 및 수집 함수
# ------------------------------------------------------------
ASOS_SERVICE_KEY = (
    "feb2bfabd299d5d05e89c7aec49ba7e706112603e76549a92e868bd86ec60323"
)
ASOS_URL = "http://apis.data.go.kr/1360000/AsosDalyInfoService/getWthrDataList"
STN_ID = "281"  # 영천 관측소


def fetch_asos_year(year):
    start_dt = f"{year}0101"
    end_dt = f"{year}1231"
    params = {
        "serviceKey": ASOS_SERVICE_KEY,
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
        items = result["response"]["body"]["items"]["item"]
        return pd.DataFrame(items)
    except Exception as e:
        print(f"❌ {year}년 수집 실패: {e}")
        return pd.DataFrame()


# ------------------------------------------------------------
# 3. 데이터 수집 및 전처리
# ------------------------------------------------------------
print(
    f"🚀 {start_year}년부터 {end_year}년까지 총 10년치 기상데이터 수집 시작..."
)

all_years = []
for year in range(start_year, end_year + 1):
    df_year = fetch_asos_year(year)
    all_years.append(df_year)
    time.sleep(0.2)

weather_raw = pd.concat(all_years, ignore_index=True)

# 컬럼 선택 및 이름 변경
weather = weather_raw[
    [
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
].copy()
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

# 타입 변환 및 결측치 처리
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
    weather.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
)

# 미세먼지 데이터 수집 및 병합
print("🌫️ 미세먼지 데이터 수집 중...")
air_url = "https://docs.google.com/spreadsheets/d/1fBEnheVOP-23Hmv_5ZJZVy6m9VmNkpVd2XutOdmlYc8/export?format=csv&gid=700055413"
air = pd.read_csv(air_url)
air["date"] = pd.to_datetime(air["date"], errors="coerce")

df = pd.merge(weather, air, on="date", how="left")

# ------------------------------------------------------------
# 4. 파일 저장
# ------------------------------------------------------------
df.to_csv(save_path, index=False, encoding="utf-8-sig")
print(f"✅ 수집 및 저장 완료: {save_path}")
