import streamlit as st
import pandas as pd
import requests
import os
import urllib.parse
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# ==========================================================
# 0. 설정
# ==========================================================

st.set_page_config(
    page_title="영천 지역 문화재 훼손 위험 예측",
    page_icon="🏛",
    layout="wide"
)

SERVICE_KEY = st.secrets.get("SERVICE_KEY", "feb2bfabd299d5d05e89c7aec49ba7e706112603e76549a92e868bd86ec60323")

now = datetime.now(ZoneInfo("Asia/Seoul"))

# 최신 예측용: 최근 7일 범위에서 공공데이터 조회
end_date_dt = now - timedelta(days=1)
start_date_dt = now - timedelta(days=7)

end_date = end_date_dt.strftime("%Y%m%d")
start_date = start_date_dt.strftime("%Y%m%d")

if "danger_count" not in st.session_state:
    st.session_state["danger_count"] = 0

# ==========================================================
# 1. 파생변수 생성
# ==========================================================

def add_derived_features(df):
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    df["temp_change"] = df["temp"].diff().fillna(0)
    df["humidity_change"] = df["humidity"].diff().fillna(0)

    df["dew_point"] = df["temp"] - ((100 - df["humidity"]) / 5)
    df["dew_gap"] = df["temp"] - df["dew_point"]

    df["humidity_ma3"] = df["humidity"].rolling(3).mean().fillna(df["humidity"])
    df["pm10_ma3"] = df["pm10"].rolling(3).mean().fillna(df["pm10"])

    df["temp_std"] = df["temp"].rolling(3).std().fillna(0)
    df["humidity_std"] = df["humidity"].rolling(3).std().fillna(0)

    df["mold_risk"] = (
        (df["humidity"] >= 75)
        .rolling(3)
        .sum()
        .fillna(0) >= 2
    ).astype(int)

    df["pm_load"] = (
        df["pm10"] + df["pm25"]
    ).rolling(3).sum().fillna(0)

    return df

# ==========================================================
# 2. 위험도 규칙 함수
# ==========================================================

def classify_humidity(h):
    if h >= 75 or h < 35:
        return 2
    elif h >= 65 or h < 45:
        return 1
    else:
        return 0


def classify_temp(t):
    if t > 30 or t < 5:
        return 2
    elif t > 25 or t < 15:
        return 1
    else:
        return 0


def classify_dew(d):
    if d < 2:
        return 2
    elif d < 5:
        return 1
    else:
        return 0


def classify_pm10(p):
    if p >= 80:
        return 2
    elif p >= 30:
        return 1
    else:
        return 0


def classify_temp_change(tc):
    if abs(tc) >= 10:
        return 2
    elif abs(tc) >= 5:
        return 1
    else:
        return 0


def classify_humidity_change(hc):
    if abs(hc) >= 30:
        return 2
    elif abs(hc) >= 15:
        return 1
    else:
        return 0


def calc_weighted_risk(row):
    return (
        classify_humidity(row["humidity"]) * 0.30
        + classify_temp(row["temp"]) * 0.20
        + classify_dew(row["dew_gap"]) * 0.20
        + classify_pm10(row["pm10"]) * 0.15
        + classify_temp_change(row["temp_change"]) * 0.10
        + classify_humidity_change(row["humidity_change"]) * 0.05
    )


def final_classify(score):
    if score >= 1.2:
        return 2
    elif score >= 0.5:
        return 1
    else:
        return 0


def exposure_multiplier(exp):
    return {
        "실외": 1.0,
        "반실외": 0.7,
        "실내": 0.3
    }.get(exp, 1.0)


def material_extra_risk(mat, row):
    extra = 0.0

    dew = row.get("dew_gap", 5.0)
    hum = row.get("humidity", 50.0)
    rain = row.get("rainfall", 0.0)

    if mat == "목조":
        if hum > 75 or hum < 35:
            extra += 0.3
        if dew < 2:
            extra += 0.3

    elif mat == "석조":
        if rain > 15:
            extra += 0.3
        if dew < 5:
            extra += 0.2

    elif mat == "금속":
        if hum > 70:
            extra += 0.2
        if dew < 3:
            extra += 0.2

    elif mat == "벽화":
        if hum > 70 or hum < 40:
            extra += 0.4
        if dew < 3:
            extra += 0.4

    return extra

# ==========================================================
# 3. 모델 학습
# ==========================================================

FEATURES = [
    "temp", "humidity", "rainfall", "wind",
    "pm10", "pm25", "so2", "no2", "co", "o3",
    "temp_change", "humidity_change",
    "dew_gap", "humidity_ma3", "pm10_ma3",
    "temp_std", "humidity_std", "pm_load",
    "mat_code", "exp_code"
]


@st.cache_resource
def train_heritage_model():
    weather_path = "data/processed/[2016_2025] yeongcheon_weather_daily.csv"
    air_path = "data/processed/[2019_2025] air_quality.csv"

    if not os.path.exists(weather_path):
        return None, None, None, None, None, f"기상 파일 없음: {weather_path}"

    if not os.path.exists(air_path):
        return None, None, None, None, None, f"대기오염 파일 없음: {air_path}"

    w_df = pd.read_csv(weather_path)
    a_df = pd.read_csv(air_path)

    w_df = w_df.rename(
        columns={
            "avg_temperature_c": "temp",
            "daily_precipitation_mm": "rainfall",
            "avg_wind_speed_ms": "wind",
            "avg_relative_humidity_pct": "humidity"
        }
    )

    w_df["date"] = pd.to_datetime(w_df["date"])
    a_df["date"] = pd.to_datetime(a_df["date"])


    m_df = pd.merge(w_df, a_df, on="date", how="inner")
    m_df = m_df.ffill()
    m_df = add_derived_features(m_df)

    mats = {
        "목조": 0,
        "석조": 1,
        "금속": 2,
        "벽화": 3,
        "기타": 4
    }

    exps = {
        "실외": 0,
        "실내": 1,
        "반실외": 2
    }

    train_rows = []

    for _, r in m_df.tail(1200).iterrows():
        base_score = calc_weighted_risk(r)

        for m_name, m_code in mats.items():
            for e_name, e_code in exps.items():
                adj = exposure_multiplier(e_name)
                extra = material_extra_risk(m_name, r)

                score = min(base_score * adj + extra, 2.0)
                target = final_classify(score)

                train_rows.append(
                    {
                        "temp": r.get("temp", 0),
                        "humidity": r.get("humidity", 0),
                        "rainfall": r.get("rainfall", 0),
                        "wind": r.get("wind", 0),
                        "pm10": r.get("pm10", 0),
                        "pm25": r.get("pm25", 0),
                        "so2": r.get("so2", 0),
                        "no2": r.get("no2", 0),
                        "co": r.get("co", 0),
                        "o3": r.get("o3", 0),
                        "temp_change": r.get("temp_change", 0),
                        "humidity_change": r.get("humidity_change", 0),
                        "dew_gap": r.get("dew_gap", 0),
                        "humidity_ma3": r.get("humidity_ma3", 0),
                        "pm10_ma3": r.get("pm10_ma3", 0),
                        "temp_std": r.get("temp_std", 0),
                        "humidity_std": r.get("humidity_std", 0),
                        "pm_load": r.get("pm_load", 0),
                        "mat_code": m_code,
                        "exp_code": e_code,
                        "target": target
                    }
                )

    tdf = pd.DataFrame(train_rows).fillna(0)

    X = tdf[FEATURES]
    y = tdf["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42
    )

    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, pred)

    report_dict = classification_report(
        y_test,
        pred,
        labels=[0, 1, 2],
        target_names=["안전", "주의", "위험"],
        output_dict=True,
        zero_division=0
    )

    report_df = pd.DataFrame(report_dict).transpose()

    importance_df = pd.DataFrame(
        {
            "변수": FEATURES,
            "중요도": model.feature_importances_
        }
    ).sort_values("중요도", ascending=False)

    return model, FEATURES, report_df, importance_df, accuracy, None


ai_model, feature_names, report_df, importance_df, accuracy, model_error = train_heritage_model()

if model_error:
    st.error(model_error)
    st.stop()

# ==========================================================
# 4. 최신 공공데이터 수집
# ==========================================================

weather_list = []

try:
    asos_params = {
        "serviceKey": SERVICE_KEY,
        "numOfRows": "10",
        "dataType": "JSON",
        "dataCd": "ASOS",
        "dateCd": "DAY",
        "startDt": start_date,
        "endDt": end_date,
        "stnIds": "281"
    }

    res = requests.get(
        "https://apis.data.go.kr/1360000/AsosDalyInfoService/getWthrDataList",
        params=asos_params,
        timeout=10
    ).json()

    items = res["response"]["body"]["items"]["item"]

    for item in items:
        rf = item.get("sumRn", "0.0")

        if rf == "" or rf is None:
            rf = "0.0"

        weather_list.append(
            {
                "date": item["tm"],
                "temp": float(item.get("avgTa", 0)),
                "humidity": float(item.get("avgRhm", 0)),
                "rainfall": float(rf),
                "wind": float(item.get("avgWs", 0))
            }
        )

except Exception as e:
    st.warning(f"기상 데이터 수집 실패: {e}")

air_list = []

try:
    air_url = "http://apis.data.go.kr/B552584/ArpltnStatsSvc/getMsrstnAcctoRDyrg"

    safe_service_key = urllib.parse.unquote(SERVICE_KEY)

    air_params = {
        "serviceKey": safe_service_key,
        "returnType": "json",
        "numOfRows": "10",
        "pageNo": "1",
        "inqBginDt": start_date,
        "inqEndDt": end_date,
        "msrstnName": "영천"
    }

    air_response = requests.get(
        air_url,
        params=air_params,
        timeout=15
    )

    if air_response.status_code == 200 and air_response.text.strip().startswith("{"):
        air_data = air_response.json()
        items = air_data["response"]["body"]["items"]

        for item in items:
            raw_date = item.get("msurDt", "")

            if raw_date:
                air_list.append(
                    {
                        "date": raw_date,
                        "pm10": float(item.get("pm10Value", 0)),
                        "pm25": float(item.get("pm25Value", 0)),
                        "o3": float(item.get("o3Value", 0)),
                        "no2": float(item.get("no2Value", 0)),
                        "co": float(item.get("coValue", 0)),
                        "so2": float(item.get("so2Value", 0))
                    }
                )

except Exception as e:
    st.warning(f"대기오염 데이터 수집 실패: {e}")

# ==========================================================
# 5. 최신 환경 데이터 구성
# ==========================================================

w_df_curr = pd.DataFrame(weather_list)
a_df_curr = pd.DataFrame(air_list)

if w_df_curr.empty or a_df_curr.empty:
    st.error("최신 기상 또는 대기오염 데이터를 불러오지 못했습니다.")
    st.stop()

w_df_curr["date"] = pd.to_datetime(w_df_curr["date"])
a_df_curr["date"] = pd.to_datetime(a_df_curr["date"])

# 날짜별 1행으로 정리
w_df_curr = (
    w_df_curr
    .sort_values("date")
    .groupby("date", as_index=False)
    .mean(numeric_only=True)
)

a_df_curr = (
    a_df_curr
    .sort_values("date")
    .groupby("date", as_index=False)
    .mean(numeric_only=True)
)

with st.expander("🔍 날짜 병합 확인"):
    st.write("기상 API 날짜")
    st.write(w_df_curr["date"].sort_values(ascending=False))

    st.write("대기오염 API 날짜")
    st.write(a_df_curr["date"].sort_values(ascending=False))


st.write("기상 날짜")
st.write(w_df_curr["date"])

st.write("대기 날짜")
st.write(a_df_curr["date"])

merged_curr = pd.merge(
    w_df_curr,
    a_df_curr,
    on="date",
    how="inner"
)

if merged_curr.empty:
    st.error("기상 데이터와 대기오염 데이터의 날짜가 일치하지 않습니다.")
    st.stop()

merged_curr = merged_curr.drop_duplicates(
    subset=["date"],
    keep="last"
)

merged_curr = merged_curr.sort_values("date").reset_index(drop=True)

processed_curr = add_derived_features(merged_curr)

three_days_display = processed_curr.tail(3).copy()

target_row = processed_curr.iloc[-1]

tm = target_row["date"].strftime("%Y-%m-%d")
data_time = tm

curr_raw = {
    k: target_row[k]
    for k in [
        "temp", "humidity", "rainfall", "wind",
        "pm10", "pm25", "so2", "no2", "co", "o3"
    ]
}

curr_env = {
    k: target_row[k]
    for k in FEATURES[:-2]
}

dew_point = curr_raw["temp"] - ((100 - curr_raw["humidity"]) / 5)

curr_weighted_risk = calc_weighted_risk(
    {
        **curr_raw,
        "dew_gap": curr_env.get("dew_gap", 5.0),
        "temp_change": curr_env.get("temp_change", 0.0),
        "humidity_change": curr_env.get("humidity_change", 0.0)
    }
)

curr_risk_grade = final_classify(curr_weighted_risk)

# ==========================================================
# 6. 문화재별 예측
# ==========================================================

heritage_path = "data/processed/yc_heritage_feature.csv"

if os.path.exists(heritage_path):
    heritage_df = pd.read_csv(heritage_path)
else:
    st.error("문화재 특성 데이터 파일을 찾을 수 없습니다.")
    st.stop()

res_df = pd.DataFrame()

mat_map = {
    "목조": 0,
    "석조": 1,
    "금속": 2,
    "벽화": 3
}

exp_map = {
    "실외": 0,
    "실내": 1,
    "반실외": 2
}


def get_class_probability(model, prob_array, class_label):
    if class_label in model.classes_:
        idx = list(model.classes_).index(class_label)
        return prob_array[idx]
    return 0


if ai_model is not None:
    results = []

    for _, row in heritage_df.iterrows():
        mat = str(row["재질"]).strip()
        exp = str(row["노출형태"]).strip()

        m_code = mat_map.get(mat, 4)
        e_code = exp_map.get(exp, 0)

        input_v = pd.DataFrame(
            [
                {
                    **curr_env,
                    "mat_code": m_code,
                    "exp_code": e_code
                }
            ]
        )

        pred = ai_model.predict(input_v[feature_names])[0]
        prob = ai_model.predict_proba(input_v[feature_names])[0]

        adj = exposure_multiplier(exp)
        extra = material_extra_risk(mat, curr_env)

        adj_score = min(curr_weighted_risk * adj + extra, 2.0)

        danger_prob = get_class_probability(ai_model, prob, 2)
        danger_pct = round(min(danger_prob * 100 + extra * 15, 100), 1)

        results.append(
            {
                "문화재명": row["문화재명(국문)"],
                "재질": mat,
                "노출": exp,
                "위험지수": round(adj_score, 3),
                "위험수치": danger_pct,
                "등급": pred
            }
        )

    res_df = pd.DataFrame(results)

cnt_safe = len(res_df[res_df["등급"] == 0])
cnt_warn = len(res_df[res_df["등급"] == 1])
cnt_dang = len(res_df[res_df["등급"] == 2])

st.session_state["danger_count"] = cnt_dang

# ==========================================================
# 7. 화면 구성
# ==========================================================

st.title("🏛 공공 환경 데이터 기반 영천 지역 문화재 훼손 위험 예측")
st.caption(f"기준일자: {tm}")

st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("🌦 기상 환경")
    st.metric("평균기온", f"{curr_raw['temp']:.1f} °C")
    st.metric("평균습도", f"{curr_raw['humidity']:.1f} %")
    st.metric("일강수량", f"{curr_raw['rainfall']:.1f} mm")
    st.metric("평균풍속", f"{curr_raw['wind']:.1f} m/s")

with col2:
    st.subheader("🌫 대기오염")
    st.metric("PM10", f"{curr_raw['pm10']:.0f}")
    st.metric("PM2.5", f"{curr_raw['pm25']:.0f}")
    st.metric("O₃", f"{curr_raw['o3']:.3f}")
    st.metric("NO₂", f"{curr_raw['no2']:.3f}")
    st.metric("CO", f"{curr_raw['co']:.1f}")
    st.metric("SO₂", f"{curr_raw['so2']:.3f}")

with col3:
    grade_kor = {
        0: "안전",
        1: "주의",
        2: "위험"
    }

    curr_label = grade_kor[curr_risk_grade]

    st.subheader("🏛 문화재 현황")
    st.metric("분석 문화재 수", f"{len(heritage_df)}개")
    st.metric("환경 위험지수", f"{curr_weighted_risk:.2f}")
    st.metric("현재 환경 등급", curr_label)
    st.metric("고위험 문화재", f"{cnt_dang}개")

# ==========================================================
# 8. 최근 환경 데이터
# ==========================================================

st.divider()

st.subheader("📅 최근 환경 데이터")

st.dataframe(
    three_days_display[
        [
            "date", "temp", "humidity", "rainfall", "wind",
            "pm10", "pm25", "so2", "no2", "co", "o3"
        ]
    ],
    use_container_width=True,
    hide_index=True
)

# ==========================================================
# 9. 파생변수
# ==========================================================

with st.expander("🔬 파생변수 상세 보기"):
    d1, d2, d3, d4 = st.columns(4)

    d1.metric("이슬점", f"{dew_point:.1f} ℃")
    d1.metric("결로 위험 간격", f"{curr_env.get('dew_gap', 0):.1f} ℃")

    d2.metric("PM 3일 누적 노출", f"{curr_env.get('pm_load', 0):.1f}")
    d2.metric("가중합 위험지수", f"{curr_weighted_risk:.3f}")

    d3.metric("온도 변화량", f"{curr_env.get('temp_change', 0):.1f} ℃")
    d3.metric("습도 변화량", f"{curr_env.get('humidity_change', 0):.1f} %")

    d4.metric("3일 습도 표준편차", f"{curr_env.get('humidity_std', 0):.2f}")
    d4.metric("3일 곰팡이 위험", f"{curr_env.get('mold_risk', 0)}")

# ==========================================================
# 10. AI 모델 성능 평가
# ==========================================================

st.divider()

st.subheader("🤖 AI 모델 성능 평가")

if report_df is not None:
    st.metric("Random Forest 정확도", f"{accuracy * 100:.1f}%")

    st.dataframe(
        report_df.round(3),
        use_container_width=True
    )

    st.caption(
        "precision은 예측한 위험등급 중 실제로 맞은 비율, "
        "recall은 실제 해당 등급 중 모델이 찾아낸 비율, "
        "f1-score는 precision과 recall의 균형 지표입니다."
    )
else:
    st.warning("AI 모델 성능 평가 결과가 없습니다.")

# ==========================================================
# 11. 변수 중요도
# ==========================================================

st.divider()

st.subheader("📊 AI 변수 중요도")

if importance_df is not None:
    st.bar_chart(
        importance_df.head(10).set_index("변수")
    )

    st.dataframe(
        importance_df.round(4),
        use_container_width=True,
        hide_index=True
    )

    top_feature = importance_df.iloc[0]["변수"]

    st.success(f"현재 모델에서 가장 중요한 변수는 '{top_feature}'입니다.")
else:
    st.warning("변수 중요도 결과가 없습니다.")

# ==========================================================
# 12. 문화재별 위험도 예측
# ==========================================================

st.divider()

st.subheader("📊 문화재별 AI 위험도 예측")

if not res_df.empty:
    s1, s2, s3 = st.columns(3)

    s1.metric("✅ 안전", f"{cnt_safe}건")
    s2.metric("⚠️ 주의", f"{cnt_warn}건")
    s3.metric("🚨 위험", f"{cnt_dang}건")

    display_df = (
        res_df
        .assign(
            판정=res_df["등급"].map(
                {
                    0: "안전",
                    1: "주의",
                    2: "위험"
                }
            )
        )
        .sort_values("위험수치", ascending=False)
        [
            [
                "문화재명",
                "재질",
                "노출",
                "위험지수",
                "위험수치",
                "판정"
            ]
        ]
    )

    st.dataframe(
        display_df,
        column_config={
            "위험수치": st.column_config.ProgressColumn(
                "훼손 위험 수치",
                min_value=0,
                max_value=100,
                format="%.1f%%"
            )
        },
        use_container_width=True,
        hide_index=True
    )

    danger_list = res_df[res_df["등급"] == 2]

    if not danger_list.empty:
        st.error(f"🚨 현재 위험 등급 문화재 {len(danger_list)}건")

        for _, r in danger_list.iterrows():
            st.markdown(
                f"- **{r['문화재명']}** | 재질: {r['재질']} | 노출: {r['노출']} "
                f"| 위험지수: `{r['위험지수']:.3f}` | 위험수치: `{r['위험수치']}%`"
            )
else:
    st.error("문화재별 분석 데이터를 불러오지 못했습니다.")

st.caption("제6회 학생 SW·AI 인재양성 프로젝트 | 선화여고 - 영천 헤리티지 AI 탐구단")
