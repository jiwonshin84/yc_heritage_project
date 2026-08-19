import streamlit as st
import requests
import time
import pandas as pd
import numpy as np
import itertools
import matplotlib.pyplot as plt
import koreanize_matplotlib
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Matplotlib 기본 한글 및 폰트 설정 (Streamlit Cloud 리눅스 환경)
plt.rcParams['font.family'] = 'DejaVu Sans' # 영문 표기 시 에러 방지
plt.rcParams['axes.unicode_minus'] = False


st.set_page_config(page_title="데이터 수집 및 중요도 분석", layout="wide")

st.title("📊 1페이지: 기상·미세먼지 데이터 수집 및 재질별 중요도 분석")

# ------------------------------------------------------------
# 1. API 및 기본 설정
# ------------------------------------------------------------
ASOS_SERVICE_KEY = "feb2bfabd299d5d05e89c7aec49ba7e706112603e76549a92e868bd86ec60323"
ASOS_URL = "http://apis.data.go.kr/1360000/AsosDalyInfoService/getWthrDataList"
STN_ID = "281" # 영천 관측소

def fetch_asos_year(year):
    start_dt = f"{year}0101"
    end_dt   = f"{year}1231"
    params = {
        "serviceKey": ASOS_SERVICE_KEY,
        "numOfRows": "400",
        "pageNo": "1",
        "dataType": "JSON",
        "dataCd": "ASOS",
        "dateCd": "DAY",
        "startDt": start_dt,
        "endDt": end_dt,
        "stnIds": STN_ID
    }
    try:
        response = requests.get(ASOS_URL, params=params, timeout=30)
        result = response.json()
        items = result["response"]["body"]["items"]["item"]
        return pd.DataFrame(items)
    except Exception as e:
        st.error(f"{year}년 수집 실패: {e}")
        return pd.DataFrame()

# ------------------------------------------------------------
# 2. 데이터 수집 및 전처리 실행 버튼
# ------------------------------------------------------------
if st.button("🚀 데이터 수집 및 분석 시작"):
    with st.spinner("기상청 API 및 미세먼지 데이터를 수집 중입니다..."):
        all_years = []
        for year in range(2016, 2026):
            df_year = fetch_asos_year(year)
            all_years.append(df_year)
            time.sleep(0.2)
        
        weather_raw = pd.concat(all_years, ignore_index=True)
        
        # 컬럼 선택 및 이름 변경
        weather = weather_raw[["tm", "avgTa", "maxTa", "minTa", "avgRhm", "sumRn", "avgWs", "sumSsHr", "avgTs"]].copy()
        weather.columns = ["date", "temp_avg", "temp_max", "temp_min", "humidity", "rainfall", "wind_speed", "solar_radiation", "ground_temp"]
        
        # 타입 변환
        weather["date"] = pd.to_datetime(weather["date"], errors="coerce")
        numeric_cols = ["temp_avg", "temp_max", "temp_min", "humidity", "rainfall", "wind_speed", "solar_radiation", "ground_temp"]
        for col in numeric_cols:
            weather[col] = pd.to_numeric(weather[col], errors="coerce")
            
        weather["rainfall"] = weather["rainfall"].fillna(0)
        weather = weather.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

        # 미세먼지 데이터 수집
        air_url = "https://docs.google.com/spreadsheets/d/1fBEnheVOP-23Hmv_5ZJZVy6m9VmNkpVd2XutOdmlYc8/export?format=csv&gid=700055413"
        air = pd.read_csv(air_url)
        air["date"] = pd.to_datetime(air["date"], errors="coerce")

        # 데이터 병합
        df = pd.merge(weather, air, on="date", how="left")

        # ------------------------------------------------------------
        # 3. 파생변수 생성
        # ------------------------------------------------------------
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

        # ------------------------------------------------------------
        # 4. 재질 x 노출 조합 및 정규화, 위험도 라벨링
        # ------------------------------------------------------------
        materials = ["석조", "목조", "금속", "회화", "기타"]
        exposures = ["실외", "반실외", "실내"]
        comb = pd.DataFrame(list(itertools.product(materials, exposures)), columns=["material", "exposure"])

        df["key"] = 1
        comb["key"] = 1
        dataset = pd.merge(df, comb, on="key").drop("key", axis=1)

        risk_cols = ["weathering_risk","acid_risk","rainfall_7d","temp_range","pm_load","corrosion_risk","mold_risk","humidity_std3","oxidation_risk","high_humidity_risk"]
        for col in risk_cols:
            dataset[col+"_norm"] = ((dataset[col] - dataset[col].min()) / (dataset[col].max() - dataset[col].min() + 1e-6)) * 100

        def calc_risk(row):
            m, e = row["material"], row["exposure"]
            if m == "석조":
                r = (row["weathering_risk_norm"]*0.25 + row["acid_risk_norm"]*0.20 + row["rainfall_7d_norm"]*0.18 + row["temp_range_norm"]*0.15 + row["pm_load_norm"]*0.12 + row["corrosion_risk_norm"]*0.10)
            elif m == "목조":
                r = (row["mold_risk_norm"]*0.25 + row["humidity_std3_norm"]*0.20 + row["high_humidity_risk_norm"]*0.18 + row["rainfall_7d_norm"]*0.15 + row["oxidation_risk_norm"]*0.12 + row["pm_load_norm"]*0.10)
            elif m == "금속":
                r = (row["corrosion_risk_norm"]*0.30 + row["acid_risk_norm"]*0.22 + row["high_humidity_risk_norm"]*0.18 + row["humidity_std3_norm"]*0.12 + row["pm_load_norm"]*0.10 + row["weathering_risk_norm"]*0.08)
            elif m == "회화":
                r = (row["oxidation_risk_norm"]*0.28 + row["pm_load_norm"]*0.20 + row["humidity_std3_norm"]*0.18 + row["high_humidity_risk_norm"]*0.14 + row["temp_range_norm"]*0.10 + row["weathering_risk_norm"]*0.10)
            else:
                r = (row["weathering_risk_norm"]*0.2 + row["acid_risk_norm"]*0.2 + row["oxidation_risk_norm"]*0.2 + row["corrosion_risk_norm"]*0.2 + row["pm_load_norm"]*0.2)
            
            if e == "실외": r *= 1.3
            elif e == "반실외": r *= 1.1
            else: r *= 0.85
            return min(r, 100)

        dataset["material_risk"] = dataset.apply(calc_risk, axis=1)

        def label(x):
            if x >= 80: return "위험"
            elif x >= 40: return "주의"
            else: return "안전"

        dataset["target"] = dataset["material_risk"].apply(label)

        # CSV 저장 (다음 페이지 연동용)
        dataset.to_csv("processed_dataset.csv", index=False, encoding="utf-8-sig")

        # ------------------------------------------------------------
        # 5. RandomForest 모델 학습 및 저장
        # ------------------------------------------------------------
        X = dataset[["temp_avg","temp_max","temp_min","humidity","rainfall","wind_speed","solar_radiation","ground_temp","pm10","pm25","o3","no2","co","so2","temp_range","humidity_std3","rainfall_7d","high_humidity_risk","weathering_risk","mold_risk","pm_load","acid_risk","oxidation_risk","corrosion_risk","material","exposure"]]
        y = dataset["target"]
        X_encoded = pd.get_dummies(X, columns=["material","exposure"])

        X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42, stratify=y)

        # 성능이 가장 우수했던 RandomForest 모델 사용
        rf_model = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
        rf_model.fit(X_train, y_train)

        # 모델 및 특성 목록 저장
        joblib.dump(rf_model, "best_rf_model.pkl")
        joblib.dump(X_encoded.columns.tolist(), "model_features.pkl")

    st.success("데이터 수집, 전처리 및 모델 학습이 완료되었습니다!")

    # ------------------------------------------------------------
    # 6. 중요도 분석 시각화
    # ------------------------------------------------------------
    st.subheader("📌 환경 요인 중요도 분석 TOP 10")
    
    feature_cols = [c for c in X_encoded.columns if not c.startswith("material_") and not c.startswith("exposure_")]
    importance_df = pd.DataFrame({
        "Feature": X_encoded.columns,
        "Importance": rf_model.feature_importances_
    })
    importance_df = importance_df[importance_df["Feature"].isin(feature_cols)].sort_values("Importance", ascending=False)
    
    top10 = importance_df.head(10)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(top10["Feature"], top10["Importance"], color="skyblue")
    ax.invert_yaxis()
    ax.set_xlabel("Importance")
    ax.set_title("Environmental Feature Importance")
    st.pyplot(fig)
    
    st.dataframe(top10)
