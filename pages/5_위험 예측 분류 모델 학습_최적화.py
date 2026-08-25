import itertools
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import streamlit as st

st.set_page_config(
    page_title="문화유산 위험 예측 현황 분석", layout="wide"
)
st.title("📊 문화유산 위험 예측 및 재질별 현황 분석")

# ------------------------------------------------------------
# 1. 기본 설정 및 한글 변수명 매핑 사전
# ------------------------------------------------------------
DATA_PATH = "data/processed/[2016_2025] yeongcheon.csv"

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
    "so2": "아황산가스(SO2)",
}


# ------------------------------------------------------------
# 2. 데이터 전처리 및 최적화된 백엔드 파이프라인
# ------------------------------------------------------------
@st.cache_resource
def run_model_pipeline():
    try:
        df = pd.read_csv(DATA_PATH)
    except Exception as e:
        st.error(f"데이터 파일 로드 실패: {e}")
        st.stop()

    # 날짜 타입 변환 및 정렬
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    if "rainfall" in df.columns:
        df["rainfall"] = df["rainfall"].fillna(0)

    # 파생변수 생성 (벡터화)
    df["temp_range"] = df["temp_max"] - df["temp_min"]
    df["humidity_std3"] = df["humidity"].rolling(3, min_periods=1).std()
    df["rainfall_7d"] = df["rainfall"].rolling(7, min_periods=1).sum()
    df["high_humidity_risk"] = (
        (df["humidity"] >= 75).rolling(3, min_periods=1).sum()
    )
    df["weathering_risk"] = (
        df["temp_range"] * 0.4 + df["humidity_std3"] * 0.3 + df["wind_speed"] * 0.3
    )
    df["mold_risk"] = (
        (df["humidity"] >= 75) & (df["ground_temp"] >= 15)
    ).astype(int)
    df["pm_load"] = (df["pm10"] + df["pm25"]).rolling(3, min_periods=1).sum()
    df["acid_risk"] = df["so2"] * 0.6 + df["no2"] * 0.4
    df["oxidation_risk"] = df["o3"] * 0.7 + df["pm25"] * 0.3
    df["corrosion_risk"] = df["humidity"] * 0.5 + df["so2"] * 0.5
    df = df.fillna(0)

    # 재질 x 노출 조합 생성
    materials = ["석조", "목조", "금속", "회화", "기타"]
    exposures = ["실외", "반실외", "실내"]
    comb = pd.DataFrame(
        list(itertools.product(materials, exposures)),
        columns=["material", "exposure"],
    )

    df["key"] = 1
    comb["key"] = 1
    dataset = pd.merge(df, comb, on="key").drop("key", axis=1)

    # MinMax 정규화
    risk_cols = [
        "weathering_risk",
        "acid_risk",
        "rainfall_7d",
        "temp_range",
        "pm_load",
        "corrosion_risk",
        "mold_risk",
        "humidity_std3",
        "oxidation_risk",
        "high_humidity_risk",
    ]
    for col in risk_cols:
        dataset[col + "_norm"] = (
            (dataset[col] - dataset[col].min())
            / (dataset[col].max() - dataset[col].min() + 1e-6)
        ) * 100

    mat = dataset["material"].values
    exp = dataset["exposure"].values

    calc_석조 = (
        dataset["weathering_risk_norm"] * 0.25
        + dataset["acid_risk_norm"] * 0.20
        + dataset["rainfall_7d_norm"] * 0.18
        + dataset["temp_range_norm"] * 0.15
        + dataset["pm_load_norm"] * 0.12
        + dataset["corrosion_risk_norm"] * 0.10
    )
    calc_목조 = (
        dataset["mold_risk_norm"] * 0.25
        + dataset["humidity_std3_norm"] * 0.20
        + dataset["high_humidity_risk_norm"] * 0.18
        + dataset["rainfall_7d_norm"] * 0.15
        + dataset["oxidation_risk_norm"] * 0.12
        + dataset["pm_load_norm"] * 0.10
    )
    calc_금속 = (
        dataset["corrosion_risk_norm"] * 0.30
        + dataset["acid_risk_norm"] * 0.22
        + dataset["high_humidity_risk_norm"] * 0.18
        + dataset["humidity_std3_norm"] * 0.12
        + dataset["pm_load_norm"] * 0.10
        + dataset["weathering_risk_norm"] * 0.08
    )
    calc_회화 = (
        dataset["oxidation_risk_norm"] * 0.28
        + dataset["pm_load_norm"] * 0.20
        + dataset["humidity_std3_norm"] * 0.18
        + dataset["high_humidity_risk_norm"] * 0.14
        + dataset["temp_range_norm"] * 0.10
        + dataset["weathering_risk_norm"] * 0.10
    )
    calc_기타 = (
        dataset["weathering_risk_norm"] * 0.2
        + dataset["acid_risk_norm"] * 0.2
        + dataset["oxidation_risk_norm"] * 0.2
        + dataset["corrosion_risk_norm"] * 0.2
        + dataset["pm_load_norm"] * 0.2
    )

    r_base = np.select(
        [mat == "석조", mat == "목조", mat == "금속", mat == "회화"],
        [calc_석조, calc_목조, calc_금속, calc_회화],
        default=calc_기타,
    )

    exp_mult = np.select(
        [exp == "실외", exp == "반실외"], [1.3, 1.1], default=0.85
    )
    dataset["material_risk"] = np.clip(r_base * exp_mult, 0, 100)

    # Target 라벨링
    conditions = [dataset["material_risk"] >= 80, dataset["material_risk"] >= 40]
    dataset["target"] = np.select(conditions, ["위험", "주의"], default="안전")

    # 머신러닝 모델 학습 (백엔드 고정 하이퍼파라미터 적용)
    X = dataset[[
        "temp_avg",
        "temp_max",
        "temp_min",
        "humidity",
        "rainfall",
        "wind_speed",
        "solar_radiation",
        "ground_temp",
        "pm10",
        "pm25",
        "o3",
        "no2",
        "co",
        "so2",
        "temp_range",
        "humidity_std3",
        "rainfall_7d",
        "high_humidity_risk",
        "weathering_risk",
        "mold_risk",
        "pm_load",
        "acid_risk",
        "oxidation_risk",
        "corrosion_risk",
        "material",
        "exposure",
    ]]
    y = dataset["target"]
    X_encoded = pd.get_dummies(X, columns=["material", "exposure"])

    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.2, random_state=42, stratify=y
    )

    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    gb_model = HistGradientBoostingClassifier(max_iter=100, random_state=42)
    lr_model = LogisticRegression(max_iter=1000, solver="lbfgs")

    rf_model.fit(X_train, y_train)
    gb_model.fit(X_train, y_train)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    lr_model.fit(X_train_scaled, y_train)

    acc_rf = accuracy_score(y_test, rf_model.predict(X_test))
    acc_gb = accuracy_score(y_test, gb_model.predict(X_test))
    acc_lr = accuracy_score(y_test, lr_model.predict(X_test_scaled))

    model_results = {
        "RandomForest": acc_rf,
        "HistGradientBoosting": acc_gb,
        "LogisticRegression": acc_lr,
    }

    return dataset, model_results, rf_model, X_encoded


# ------------------------------------------------------------
# 3. 데이터 로드
# ------------------------------------------------------------
with st.spinner("🚀 분석 데이터를 로드하고 예측 모델을 준비 중입니다..."):
    dataset, model_results, rf_model, X_encoded = run_model_pipeline()

# ------------------------------------------------------------
# 4. 사이드바: 도메인 중심 필터 구성
# ------------------------------------------------------------
st.sidebar.header("🔍 문화유산 조건 필터")

selected_material = st.sidebar.selectbox(
    "🏛️ 문화유산 재질 선택",
    options=["전체"] + list(dataset["material"].unique()),
    index=0,
)

selected_exposure = st.sidebar.selectbox(
    "🌿 노출 환경 선택",
    options=["전체"] + list(dataset["exposure"].unique()),
    index=0,
)

# 필터링 적용
filtered_df = dataset.copy()
if selected_material != "전체":
    filtered_df = filtered_df[filtered_df["material"] == selected_material]
if selected_exposure != "전체":
    filtered_df = filtered_df[filtered_df["exposure"] == selected_exposure]

# ------------------------------------------------------------
# 5. 핵심 지표 메트릭 카드 (Dashboard KPI)
# ------------------------------------------------------------
st.markdown(f"### 📌 [{selected_material}] / [{selected_exposure}] 위험 예측 현황 Summary")

kpi1, kpi2, kpi3, kpi4 = st.columns(4)

total_count = len(filtered_df)
danger_count = (filtered_df["target"] == "위험").sum()
caution_count = (filtered_df["target"] == "주의").sum()
safe_count = (filtered_df["target"] == "안전").sum()

kpi1.metric("총 분석 데이터 수", f"{total_count:,} 건")
kpi2.metric("🚨 위험 등급 비율", f"{(danger_count / total_count * 100):.1f}%", f"{danger_count:,} 건")
kpi3.metric("⚠️ 주의 등급 비율", f"{(caution_count / total_count * 100):.1f}%", f"{caution_count:,} 건")
kpi4.metric("✅ 안전 등급 비율", f"{(safe_count / total_count * 100):.1f}%", f"{safe_count:,} 건")

st.markdown("---")

# ------------------------------------------------------------
# 6. 메인 분석 시각화
# ------------------------------------------------------------
col1, col2 = st.columns(2)

# [시각화 1] 선택 조건 기준 위험 등급 분포 (Pie Chart)
with col1:
    st.subheader("🎯 위험 예측 등급 분포")
    target_counts = filtered_df["target"].value_counts().reset_index()
    target_counts.columns = ["Target", "Count"]

    fig1 = px.pie(
        target_counts,
        names="Target",
        values="Count",
        color="Target",
        color_discrete_map={"위험": "#E74C3C", "주의": "#F39C12", "안전": "#2ECC71"},
        hole=0.4,
        title=f"선택 조건 내 위험도 분포 비율",
    )
    fig1.update_traces(textinfo="percent+label")
    fig1.update_layout(height=380, margin=dict(t=40, b=20, l=10, r=10))
    st.plotly_chart(fig1, use_container_width=True)

# [시각화 2] 주요 환경 위험 요인 TOP 10 (Feature Importance)
with col2:
    st.subheader("📌 주요 영향 환경 요인 TOP 10")
    feature_cols = [
        c for c in X_encoded.columns
        if not c.startswith("material_") and not c.startswith("exposure_")
    ]

    importance_df = pd.DataFrame({
        "Feature": X_encoded.columns,
        "Importance": rf_model.feature_importances_,
    })
    importance_df = importance_df[importance_df["Feature"].isin(feature_cols)].sort_values(
        "Importance", ascending=False
    )
    top10 = importance_df.head(10).copy()
    top10["Feature_KO"] = top10["Feature"].map(lambda x: FEATURE_NAME_KO.get(x, x))
    top10_sorted = top10.sort_values("Importance", ascending=True)

    fig2 = px.bar(
        top10_sorted,
        x="Importance",
        y="Feature_KO",
        orientation="h",
        color_discrete_sequence=["#3498db"],
        title="위험도 판단 시 기여도가 높은 요인",
    )
    fig2.update_layout(
        xaxis_title="기여도 (Importance)",
        yaxis_title="",
        height=380,
        margin=dict(t=40, b=20, l=10, r=10),
    )
    st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")

# [시각화 3] 재질별 파생변수 가중치 구조
st.subheader("🏛️ 문화유산 재질별 주요 환경 파생변수 가중치 구조")

material_weights = {
    "석조": {"풍화 위험도": 25, "산성 위험도": 20, "7일 누적강수량": 18, "일교차": 15, "미세먼지 누적부하": 12, "부식 위험도": 10},
    "목조": {"곰팡이 위험도": 25, "3일 습도변동성": 20, "고습도 위험지속도": 18, "7일 누적강수량": 15, "산화 위험도": 12, "미세먼지 누적부하": 10},
    "금속": {"부식 위험도": 30, "산성 위험도": 22, "고습도 위험지속도": 18, "3일 습도변동성": 12, "미세먼지 누적부하": 10, "풍화 위험도": 8},
    "회화": {"산화 위험도": 28, "미세먼지 누적부하": 20, "3일 습도변동성": 18, "고습도 위험지속도": 14, "일교차": 10, "풍화 위험도": 10},
    "기타": {"풍화 위험도": 20, "산성 위험도": 20, "산화 위험도": 20, "부식 위험도": 20, "미세먼지 누적부하": 20},
}

fig3 = make_subplots(
    rows=1, cols=5,
    subplot_titles=[f"[{mat}] 가중치 (%)" for mat in material_weights.keys()]
)
colors = ["#8D6E63", "#D7CCC8", "#78909C", "#EC407A", "#AB47BC"]

for idx, (mat, weights) in enumerate(material_weights.items()):
    sorted_weights = dict(sorted(weights.items(), key=lambda item: item[1]))
    fig3.add_trace(
        go.Bar(
            x=list(sorted_weights.values()),
            y=list(sorted_weights.keys()),
            orientation="h",
            marker_color=colors[idx],
            showlegend=False,
        ),
        row=1, col=idx + 1,
    )

fig3.update_layout(height=420, margin=dict(t=50, b=30, l=10, r=10))
st.plotly_chart(fig3, use_container_width=True)

# ------------------------------------------------------------
# 7. 필터링된 데이터 표 출력 및 다운로드
# ------------------------------------------------------------
st.subheader("📋 선택 조건 위험 예측 데이터 상세 목록")

display_cols = ["date", "material", "exposure", "material_risk", "target", "temp_avg", "humidity", "pm10", "pm25"]
cols_present = [c for c in display_cols if c in filtered_df.columns]

output_df = filtered_df[cols_present].copy()
output_df = output_df.rename(columns={
    "date": "날짜",
    "material": "재질",
    "exposure": "노출 환경",
    "material_risk": "위험도 점수",
    "target": "위험 등급",
    "temp_avg": "평균 기온(℃)",
    "humidity": "습도(%)",
    "pm10": "PM10",
    "pm25": "PM2.5",
})

st.dataframe(output_df.reset_index(drop=True), use_container_width=True, height=300)
