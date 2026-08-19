import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="7일전 데이터 기반 예측", layout="wide")

st.title("🔮 2페이지: 최적 모델 기반 전일~7일 전 데이터 예측")

@st.cache_data
def load_data():
    try:
        df = pd.read_csv("processed_dataset.csv")
        df["date"] = pd.to_datetime(df["date"])
        return df
    except FileNotFoundError:
        return None

@st.cache_resource
def load_model():
    try:
        model = joblib.load("best_rf_model.pkl")
        features = joblib.load("model_features.pkl")
        return model, features
    except FileNotFoundError:
        return None, None

dataset = load_data()
model, feature_cols = load_model()

if dataset is None or model is None:
    st.warning("⚠️ 1페이지에서 '데이터 수집 및 분석 시작' 버튼을 먼저 실행해 주세요!")
else:
    st.subheader("📅 최근 데이터 기준 전일 ~ 7일 전 위험도 조회")
    
    # 1. 재질 및 노출 선택
    col1, col2 = st.columns(2)
    with col1:
        selected_material = st.selectbox("재질 선택", ["석조", "목조", "금속", "회화", "기타"])
    with col2:
        selected_exposure = st.selectbox("노출 정도 선택", ["실외", "반실외", "실내"])

    # 2. 데이터 추출 (가장 최근 날짜 기준 7일전~전일)
    max_date = dataset["date"].max()
    start_date = max_date - pd.Timedelta(days=7)
    end_date = max_date - pd.Timedelta(days=1)

    filtered_df = dataset[
        (dataset["date"] >= start_date) & 
        (dataset["date"] <= end_date) & 
        (dataset["material"] == selected_material) & 
        (dataset["exposure"] == selected_exposure)
    ].sort_values("date", ascending=False)

    st.write(f"**조회 기간:** {start_date.strftime('%Y-%m-%d')} ~ {end_date.strftime('%Y-%m-%d')}")

    # 3. 모델을 활용한 예측 진행
    X_input = filtered_df[["temp_avg","temp_max","temp_min","humidity","rainfall","wind_speed","solar_radiation","ground_temp","pm10","pm25","o3","no2","co","so2","temp_range","humidity_std3","rainfall_7d","high_humidity_risk","weathering_risk","mold_risk","pm_load","acid_risk","oxidation_risk","corrosion_risk","material","exposure"]]
    X_input_encoded = pd.get_dummies(X_input, columns=["material","exposure"])

    # 원-핫 인코딩 시 누락된 컬럼 보장
    for col in feature_cols:
        if col not in X_input_encoded.columns:
            X_input_encoded[col] = 0
    X_input_encoded = X_input_encoded[feature_cols]

    # 예측 실행
    predictions = model.predict(X_input_encoded)
    filtered_df["예측_위험등급"] = predictions

    # 결과 출력
    st.dataframe(
        filtered_df[["date", "material", "exposure", "temp_avg", "humidity", "pm10", "pm25", "material_risk", "target", "예측_위험등급"]],
        use_container_width=True
    )

    # 4. 시각화 (날짜별 실제 위험점수 추이)
    st.subheader("📈 최근 7일간 위험도 점수 변화")
    st.line_chart(filtered_df.set_index("date")["material_risk"])
