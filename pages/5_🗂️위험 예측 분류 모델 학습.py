import streamlit as st
import requests
import time
import pandas as pd
import numpy as np
import itertools
import joblib

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="데이터 수집 및 중요도 분석", layout="wide")

st.title("📊 위험 예측 분류 모델 / 재질별 중요도 분석")

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

# 영문 변수명 -> 한글 변수명 매핑 사전
FEATURE_NAME_KO = {
    "corrosion_risk": "부식 위험도",
    "humidity": "평균 습도",
    "high_humidity_risk": "고습도 위험 지속도",
    "oxidation_risk": "산화 위험도",
    "acid_risk": "산성 위험도",
    "pm25": "초미세먼지(PM2.5)",
    "weathering_risk": "풍화 위험도",
    "no2": "이산화질소(NO2)",
    "pm_load": "미세먼지 누적부하",
    "pm10": "미세먼지(PM10)",
    "temp_range": "일교차",
    "humidity_std3": "3일 습도 변동성",
    "rainfall_7d": "7일 누적 강수량",
    "mold_risk": "곰팡이 발생 위험도",
    "temp_avg": "평균 기온",
    "temp_max": "최고 기온",
    "temp_min": "최저 기온",
    "rainfall": "일 강수량",
    "wind_speed": "평균 풍속",
    "solar_radiation": "일사량",
    "ground_temp": "지면 온도",
    "o3": "오존(O3)",
    "co": "일산화탄소(CO)",
    "so2": "아황산가스(SO2)"
}

# ------------------------------------------------------------
# 2. 데이터 수집 및 전처리 실행 버튼
# ------------------------------------------------------------
if st.button("🚀 데이터 수집, 모델 학습 및 분석 시작"):
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
        dataset.to_csv("processed_dataset.csv", index=False, encoding="utf-8-sig")

        # ------------------------------------------------------------
        # 5. 머신러닝 3개 모델 비교 학습 및 최적 모델 저장
        # ------------------------------------------------------------
        X = dataset[["temp_avg","temp_max","temp_min","humidity","rainfall","wind_speed","solar_radiation","ground_temp","pm10","pm25","o3","no2","co","so2","temp_range","humidity_std3","rainfall_7d","high_humidity_risk","weathering_risk","mold_risk","pm_load","acid_risk","oxidation_risk","corrosion_risk","material","exposure"]]
        y = dataset["target"]
        X_encoded = pd.get_dummies(X, columns=["material","exposure"])

        X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42, stratify=y)

        # 3개 모델 정의 및 학습
        rf_model = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
        gb_model = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, random_state=42)
        lr_model = LogisticRegression(max_iter=2000, solver="lbfgs")

        rf_model.fit(X_train, y_train)
        gb_model.fit(X_train, y_train)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        lr_model.fit(X_train_scaled, y_train)

        # 모델별 정확도 평가
        acc_rf = accuracy_score(y_test, rf_model.predict(X_test))
        acc_gb = accuracy_score(y_test, gb_model.predict(X_test))
        acc_lr = accuracy_score(y_test, lr_model.predict(X_test_scaled))

        model_results = {
            "RandomForest": acc_rf,
            "GradientBoosting": acc_gb,
            "LogisticRegression": acc_lr
        }

        # 최적 모델 저장 (RandomForest)
        joblib.dump(rf_model, "best_rf_model.pkl")
        joblib.dump(X_encoded.columns.tolist(), "model_features.pkl")

    st.success("데이터 수집, 전처리 및 3개 분류 모델 학습이 완료되었습니다!")

    # ------------------------------------------------------------
    # 6. 시각화 (Plotly 변환: 한글 깨짐 완전 해결)
    # ------------------------------------------------------------
    st.markdown("---")
    col1, col2 = st.columns(2)

    # [시각화 1] 3개 분류 모델 성능 비교 (Plotly)
    with col1:
        st.subheader("📈 분류 모델 정확도(Accuracy) 비교")
        df_models = pd.DataFrame({
            "Model": list(model_results.keys()),
            "Accuracy": [v * 100 for v in model_results.values()]
        })

        fig1 = px.bar(
            df_models, 
            x="Model", 
            y="Accuracy",
            color="Model",
            color_discrete_sequence=["#4C72B0", "#55A868", "#C44E52"],
            title="머신러닝 알고리즘별 예측 정확도 비교"
        )
        fig1.update_traces(
            texttemplate="%{y:.2f}%", 
            textposition="outside"
        )
        fig1.update_layout(
            yaxis_range=[70, 100],
            yaxis_title="정확도 (%)",
            xaxis_title="",
            showlegend=False,
            height=400,
            margin=dict(t=50, b=20, l=10, r=10)
        )
        st.plotly_chart(fig1, use_container_width=True)

    # [시각화 2] 최고 성능 모델(RandomForest) 기준 환경 요인 중요도 TOP 10 (Plotly)
    with col2:
        st.subheader("📌 환경 요인 중요도 TOP 10 (RandomForest)")
        feature_cols = [c for c in X_encoded.columns if not c.startswith("material_") and not c.startswith("exposure_")]
        
        importance_df = pd.DataFrame({
            "Feature": X_encoded.columns,
            "Importance": rf_model.feature_importances_
        })
        importance_df = importance_df[importance_df["Feature"].isin(feature_cols)].sort_values("Importance", ascending=False)
        top10 = importance_df.head(10).copy()
        
        # 한글 변수명 적용
        top10["Feature_KO"] = top10["Feature"].map(lambda x: FEATURE_NAME_KO.get(x, x))
        
        # Plotly 가로 막대 그래프는 아래에서 위로 그려지므로 오름차순 정렬
        top10_sorted = top10.sort_values("Importance", ascending=True)

        fig2 = px.bar(
            top10_sorted,
            x="Importance",
            y="Feature_KO",
            orientation="h",
            title="문화유산 위험도 측정 시 주요 환경 요인 TOP 10",
            color_discrete_sequence=["#3498db"]
        )
        fig2.update_layout(
            xaxis_title="특성 중요도 (Feature Importance)",
            yaxis_title="",
            height=400,
            margin=dict(t=50, b=20, l=10, r=10)
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    
    # [시각화 3] 재질별 파생변수 가중치(중요도) 비교 (Plotly Subplots)
    st.subheader("🏛️ 문화유산 재질별 주요 환경 파생변수 가중치 구조")
    
    material_weights = {
        "석조": {"풍화 위험도": 25, "산성 위험도": 20, "7일 누적강수량": 18, "일교차": 15, "미세먼지 누적부하": 12, "부식 위험도": 10},
        "목조": {"곰팡이 위험도": 25, "3일 습도변동성": 20, "고습도 위험지속도": 18, "7일 누적강수량": 15, "산화 위험도": 12, "미세먼지 누적부하": 10},
        "금속": {"부식 위험도": 30, "산성 위험도": 22, "고습도 위험지속도": 18, "3일 습도변동성": 12, "미세먼지 누적부하": 10, "풍화 위험도": 8},
        "회화": {"산화 위험도": 28, "미세먼지 누적부하": 20, "3일 습도변동성": 18, "고습도 위험지속도": 14, "일교차": 10, "풍화 위험도": 10},
        "기타": {"풍화 위험도": 20, "산성 위험도": 20, "산화 위험도": 20, "부식 위험도": 20, "미세먼지 누적부하": 20}
    }

    fig3 = make_subplots(
        rows=1, cols=5, 
        subplot_titles=[f"[{mat}] 가중치 (%)" for mat in material_weights.keys()]
    )
    colors = ["#8D6E63", "#D7CCC8", "#78909C", "#EC407A", "#AB47BC"]

    for idx, (mat, weights) in enumerate(material_weights.items()):
        # 오름차순 정렬하여 상위 요소가 위쪽에 배치되도록 설정
        sorted_weights = dict(sorted(weights.items(), key=lambda item: item[1]))
        
        fig3.add_trace(
            go.Bar(
                x=list(sorted_weights.values()),
                y=list(sorted_weights.keys()),
                orientation="h",
                marker_color=colors[idx],
                showlegend=False
            ),
            row=1, col=idx+1
        )

    fig3.update_layout(
        height=450,
        margin=dict(t=50, b=30, l=10, r=10)
    )
    st.plotly_chart(fig3, use_container_width=True)

    # 데이터 프레임 출력
    st.subheader("📋 분류 모델 성능 및 상위 중요도 요인 요약")
    top10_display = top10[["Feature_KO", "Importance"]].rename(columns={"Feature_KO": "환경 요인 명칭", "Importance": "중요도 비율"})
    st.dataframe(top10_display.reset_index(drop=True), use_container_width=True)
