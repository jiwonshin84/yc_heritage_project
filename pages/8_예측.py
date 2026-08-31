import streamlit as st
import pandas as pd
import numpy as np
import requests
import datetime
import itertools
import joblib

st.set_page_config(page_title="최근 7일 위험도 예측", layout="wide")

st.title("🔮 최근 7일 데이터 기반 위험도 예측")

# ------------------------------------------------------------
# 1. 저장된 모델 및 특성 목록 로드
# ------------------------------------------------------------
@st.cache_resource
def load_model():
    try:
        model = joblib.load("best_rf_model.pkl")
        features = joblib.load("model_features.pkl")
        return model, features
    except FileNotFoundError:
        return None, None

model, feature_cols = load_model()

if model is None:
    st.warning("⚠️ 위험 예측 분류 모델 학습(`best_rf_model.pkl`)을 먼저 완료해 주세요!")
    st.stop()

# ------------------------------------------------------------
# 2. 최근 7일 기상/미세먼지 데이터 실시간 수집 함수
# ------------------------------------------------------------
ASOS_SERVICE_KEY = "feb2bfabd299d5d05e89c7aec49ba7e706112603e76549a92e868bd86ec60323"
ASOS_URL = "http://apis.data.go.kr/1360000/AsosDalyInfoService/getWthrDataList"
STN_ID = "281" # 영천 관측소

def fetch_recent_7days_data():
    today = datetime.date.today()
    start_date = today - datetime.timedelta(days=10) # 롤링 파생변수 계산을 위해 여유 있게 10일 전부터 수집
    
    start_str = start_date.strftime("%Y%m%d")
    end_str = today.strftime("%Y%m%d")
    
    params = {
        "serviceKey": ASOS_SERVICE_KEY,
        "numOfRows": "50",
        "pageNo": "1",
        "dataType": "JSON",
        "dataCd": "ASOS",
        "dateCd": "DAY",
        "startDt": start_str,
        "endDt": end_str,
        "stnIds": STN_ID
    }
    
    try:
        # 1) 기상청 API 수집
        response = requests.get(ASOS_URL, params=params, timeout=30)
        items = response.json()["response"]["body"]["items"]["item"]
        weather = pd.DataFrame(items)
        
        weather = weather[["tm", "avgTa", "maxTa", "minTa", "avgRhm", "sumRn", "avgWs", "sumSsHr", "avgTs"]].copy()
        weather.columns = ["date", "temp_avg", "temp_max", "temp_min", "humidity", "rainfall", "wind_speed", "solar_radiation", "ground_temp"]
        weather["date"] = pd.to_datetime(weather["date"], errors="coerce")
        
        numeric_cols = ["temp_avg", "temp_max", "temp_min", "humidity", "rainfall", "wind_speed", "solar_radiation", "ground_temp"]
        for col in numeric_cols:
            weather[col] = pd.to_numeric(weather[col], errors="coerce")
        weather["rainfall"] = weather["rainfall"].fillna(0)

        # 2) 미세먼지 데이터 수집
        air_url = "https://docs.google.com/spreadsheets/d/1fBEnheVOP-23Hmv_5ZJZVy6m9VmNkpVd2XutOdmlYc8/export?format=csv&gid=700055413"
        air = pd.read_csv(air_url)
        air["date"] = pd.to_datetime(air["date"], errors="coerce")

        # 3) 병합
        df = pd.merge(weather, air, on="date", how="left").sort_values("date").reset_index(drop=True)

        # 4) 파생변수 생성 (학습할 때와 동일한 공식 적용)
        df["temp_range"] = df["temp_max"] - df["temp_min"]
        df["humidity_std3"] = df["humidity"].rolling(3, min_periods=1).std()
        df["rainfall_7d"] = df["rainfall"].rolling(7, min_periods=1).sum()
        df["high_humidity_risk"] = (df["humidity"] >= 75).rolling(3, min_periods=1).sum()
        df["weathering_risk"] = df["temp_range"] * 0.4 + df["humidity_std3"] * 0.3 + df["wind_speed"] * 0.3
        df["mold_risk"] = ((df["humidity"] >= 75) & (df["ground_temp"] >= 15)).astype(int)
        df["pm_load"] = (df["pm10"] + df["pm25"]).rolling(3, min_periods=1).sum()
        df["acid_risk"] = df["so2"] * 0.6 + df["no2"] * 0.4
        df["oxidation_risk"] = df["o3"] * 0.7 + df["pm25"] * 0.3
        df["corrosion_risk"] = df["humidity"] * 0.5 + df["so2"] * 0.5
        df = df.fillna(0)

        return df
    except Exception as e:
        st.error(f"최근 데이터 수집 실패: {e}")
        return None

# ------------------------------------------------------------
# 3. 화면 UI 및 예측 실행
# ------------------------------------------------------------
col1, col2 = st.columns(2)
with col1:
    selected_material = st.selectbox("재질 선택", ["석조", "목조", "금속", "회화", "기타"])
with col2:
    selected_exposure = st.selectbox("노출 정도 선택", ["실외", "반실외", "실내"])

if st.button("🔄 실시간 최근 7일 위험도 예측 실행"):
    with st.spinner("최근 기상/미세먼지 데이터를 불러와 예측 중입니다..."):
        df_recent = fetch_recent_7days_data()

        if df_recent is not None and not df_recent.empty:
            # 선택한 재질 및 노출 조건 결합
            df_recent["material"] = selected_material
            df_recent["exposure"] = selected_exposure

            # 최근 7일 데이터만 추출
            df_7days = df_recent.tail(7).copy()

            # 특성 벡터 구성
            X_input = df_7days[["temp_avg","temp_max","temp_min","humidity","rainfall","wind_speed","solar_radiation","ground_temp","pm10","pm25","o3","no2","co","so2","temp_range","humidity_std3","rainfall_7d","high_humidity_risk","weathering_risk","mold_risk","pm_load","acid_risk","oxidation_risk","corrosion_risk","material","exposure"]]
            X_input_encoded = pd.get_dummies(X_input, columns=["material","exposure"])

            # 학습 때 생성된 컬럼 구조 보장
            for col in feature_cols:
                if col not in X_input_encoded.columns:
                    X_input_encoded[col] = 0
            X_input_encoded = X_input_encoded[feature_cols]

            # 2025년까지 학습된 모델로 예측 수행
            df_7days["예측_위험등급"] = model.predict(X_input_encoded)

            st.subheader(f"📅 최근 7일 예측 결과 ({df_7days['date'].min().strftime('%Y-%m-%d')} ~ {df_7days['date'].max().strftime('%Y-%m-%d')})")
            
            # 결과 표 출력
            st.dataframe(
                df_7days[["date", "material", "exposure", "temp_avg", "humidity", "pm10", "pm25", "예측_위험등급"]],
                use_container_width=True
            )

            # 등급 분포 시각화
            st.subheader("📊 최근 7일간 예측 등급 변화")
            st.bar_chart(df_7days.set_index("date")["예측_위험등급"].value_counts())
