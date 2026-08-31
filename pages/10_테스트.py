import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# ==========================================================
# 0. 경로 설정
# ==========================================================
def get_path(filename):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(current_dir, ".."))
    paths = [
        os.path.join(root_dir, "data", "processed", filename),
        os.path.join("data", "processed", filename),
        filename
    ]
    for p in paths:
        if os.path.exists(p): return p
    return None

# ==========================================================
# 1. 데이터 정제 및 파생변수 (컬럼명 통일)
# ==========================================================
def preprocess_data(df):
    df = df.copy()
    
    # 수치형 데이터만 보간 (TypeError 방지)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].interpolate(method='linear').ffill().bfill()
    
    # 비수치형 데이터 채우기
    non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns
    df[non_numeric_cols] = df[non_numeric_cols].ffill().bfill()
    
    # 파생 변수 계산 (이름을 wood_risk, stone_risk로 통일)
    df["dew_point"] = df["temp"] - ((100 - df["humidity"]) / 5)
    df["dew_gap"] = df["temp"] - df["dew_point"]
    df["wood_risk"] = np.where(df["temp"] > 2, (df["temp"] - 2) * (df["humidity"] - 30) / 100, 0)
    df["stone_risk"] = (df.get("so2", 0) * 100) + (df.get("no2", 0) * 50) + (df["rainfall"] * 0.1)
    
    return df.replace([np.inf, -np.inf], np.nan).fillna(0)

def label_logic(row):
    score = 0
    if row["humidity"] >= 80 or row["humidity"] < 30: score += 0.35
    if row["temp"] > 33 or row["temp"] < -5: score += 0.25
    if row["dew_gap"] < 2: score += 0.3
    
    age_weight = 0.25 if row.get("문화재연령", 0) > 500 else (0.1 if row.get("문화재연령", 0) > 100 else 0)
    exp_mult = 1.3 if row.get("노출형태", "실외") == "실외" else (1.0 if row.get("노출형태", "반실외") == "반실외" else 0.6)
    
    total = (score + age_weight) * exp_mult
    return 3 if total >= 1.1 else (2 if total >= 0.7 else (1 if total >= 0.3 else 0))

# ==========================================================
# 2. 모델 학습 (3대 모델 비교)
# ==========================================================
@st.cache_resource
def train_models():
    p_w = get_path("[2016_2025] yeongcheon_weather_daily.csv")
    p_a = get_path("[2019_2025] air_quality.csv")
    p_h = get_path("yc_heritage_feature.csv")
    
    if not all([p_w, p_a, p_h]): return None, None, None, None
    
    w_df = pd.read_csv(p_w).rename(columns={'avg_temperature_c': 'temp', 'daily_precipitation_mm': 'rainfall', 'avg_relative_humidity_pct': 'humidity'})
    a_df = pd.read_csv(p_a)
    h_df = pd.read_csv(p_h)
    
    w_df["date"] = pd.to_datetime(w_df["date"])
    a_df["date"] = pd.to_datetime(a_df["date"])
    
    merged = pd.merge(w_df, a_df, on="date", how="inner")
    merged = preprocess_data(merged)
    
    # 인코더 설정
    le_mat = LabelEncoder().fit(['목조', '석조', '금속', '벽화', '기타', '지석묘', '석탑'])
    le_exp = LabelEncoder().fit(['실외', '실내', '반실외'])
    
    data = []
    # 2019-2025 전수 기간 데이터 샘플링
    sampled_env = merged.sample(n=min(1200, len(merged)), random_state=42)
    
    for _, h in h_df.iterrows():
        for _, e in sampled_env.iterrows():
            row = {**h.to_dict(), **e.to_dict()}
            data.append({
                "age": h["문화재연령"],
                "mat_code": le_mat.transform([h["재질"]])[0] if h["재질"] in le_mat.classes_ else 4,
                "exp_code": le_exp.transform([h["노출형태"]])[0] if h["노출형태"] in le_exp.classes_ else 0,
                "temp": e["temp"], 
                "humidity": e["humidity"], 
                "dew_gap": e["dew_gap"],
                "wood_risk": e["wood_risk"],    # KeyError 해결: 이름 일치
                "stone_risk": e["stone_risk"],  # KeyError 해결: 이름 일치
                "target": label_logic(row)
            })
            
    df_final = pd.DataFrame(data).dropna()
    X = df_final.drop("target", axis=1)
    y = df_final["target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model_pool = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42)
    }
    
    res = {}
    for name, mdl in model_pool.items():
        mdl.fit(X_train, y_train)
        res[name] = {"model": mdl, "acc": accuracy_score(y_test, mdl.predict(X_test))}
    return res, X.columns.tolist(), le_mat, le_exp

# ==========================================================
# 3. 메인 대시보드 시각화
# ==========================================================
st.title("🧪 영천 헤리티지 AI 모델 비교 분석 (2019-2025)")

with st.spinner("빅데이터 분석 모델 학습 중..."):
    results, feat_names, le_mat, le_exp = train_models()

if results:
    # --- 그래프 1: 모델별 정확도 (Line Plot + 수치표시) ---
    st.subheader("📊 1. 알고리즘별 예측 정확도 비교")
    acc_df = pd.DataFrame({"Model": results.keys(), "Accuracy": [v["acc"] for v in results.values()]})
    
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    sns.lineplot(x="Model", y="Accuracy", data=acc_df, marker="o", markersize=12, color="#34495e", ax=ax1, linewidth=3)
    for i, v in enumerate(acc_df["Accuracy"]):
        ax1.text(i, v + 0.005, f"{v*100:.2f}%", ha='center', fontweight='bold', color='#c0392b', fontsize=11)
    ax1.set_ylim(acc_df["Accuracy"].min() - 0.03, acc_df["Accuracy"].max() + 0.05)
    ax1.set_ylabel("Accuracy (정확도)")
    ax1.grid(True, axis='y', linestyle='--', alpha=0.6)
    st.pyplot(fig1)

    # --- 그래프 2: 중요도 분석 (재질별 색상 강조 + 수치표시) ---
    st.subheader("🔎 2. 위험 판단 핵심 변수 및 재질 가중치")
    rf_importance = results["Random Forest"]["model"].feature_importances_
    
    kor_map = {
        "age": "문화재 연령", "mat_code": "재질 유형", "exp_code": "노출 형태",
        "temp": "현재 기온", "humidity": "현재 습도", "dew_gap": "결로 위험(이슬점)",
        "wood_risk": "목조 부후 위험", "stone_risk": "석조 풍화 위험"
    }
    
    fi_df = pd.DataFrame({
        '지표': [kor_map.get(f, f) for f in feat_names], 
        '중요도': rf_importance
    }).sort_values('중요도', ascending=True)

    fig2, ax2 = plt.subplots(figsize=(10, 6))
    # 재질/목조/석조 키워드가 들어간 지표는 주황색으로 강조
    colors = ['#f39c12' if any(k in x for k in ['재질', '목조', '석조']) else '#3498db' for x in fi_df['지표']]
    bars = ax2.barh(fi_df['지표'], fi_df['중요도'], color=colors)
    
    # 막대 끝에 수치값 표시
    for bar in bars:
        ax2.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2, f'{bar.get_width():.3f}', va='center', fontweight='bold')
    
    ax2.set_xlabel("Feature Importance (상대적 기여도)")
    st.pyplot(fig2)
    st.info("💡 **주황색 지표**는 AI가 문화재의 **재질적 특성**을 기후와 결합하여 분석한 결과입니다.")

    # --- 3. 실시간 예측 테이블 ---
    st.divider()
    best_name = max(results, key=lambda k: results[k]['acc'])
    st.subheader(f"🏛️ {best_name} 모델 기반 실시간 위험도 예측 (4단계)")
    
    with st.sidebar:
        st.header("⚙️ 실시간 환경 설정")
        t = st.slider("기온(℃)", -15.0, 40.0, 24.0)
        h = st.slider("습도(%)", 0, 100, 82)
        p = st.slider("미세먼지", 0, 250, 45)

    h_df = pd.read_csv(get_path("yc_heritage_feature.csv"))
    final_view = []
    for _, r in h_df.iterrows():
        dg = t - (t - ((100 - h) / 5))
        inp = pd.DataFrame([{
            "age": r["문화재연령"], 
            "mat_code": le_mat.transform([r["재질"]])[0] if r["재질"] in le_mat.classes_ else 4,
            "exp_code": le_exp.transform([r["노출형태"]])[0] if r["노출형태"] in le_exp.classes_ else 0,
            "temp": t, "humidity": h, "dew_gap": dg,
            "wood_risk": max(0, (t-2)*(h-30)/100), "stone_risk": (p * 0.1)
        }])
        pred = results[best_name]["model"].predict(inp)[0]
        final_view.append({
            "문화재명": r["문화재명(국문)"], 
            "재질": r["재질"], 
            "예측등급": {0:"✅ 안전", 1:"🟡 관심", 2:"⚠️ 주의", 3:"🚨 위험"}[pred]
        })

    st.table(pd.DataFrame(final_view))

else:
    st.error("데이터 파일을 찾을 수 없습니다. 경로와 파일명을 확인해주세요.")
