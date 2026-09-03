import datetime
import os
import joblib
import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="영천시 문화재 위험도 예측", layout="wide")

st.title("🏛️ 영천시 주요 문화재 실시간 위험도 예측 시스템")
st.markdown("##### 📌 가장 최근에 기상청 및 대기오염 데이터가 업데이트된 날짜를 기준으로 영천시 주요 문화재의 위험도를 예측합니다.")


@st.cache_resource
def load_model():
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    model_path = os.path.join(root_dir, "best_rf_model.pkl")
    features_path = os.path.join(root_dir, "model_features.pkl")

    try:
        model = joblib.load(model_path)
        features = joblib.load(features_path)
        return model, features
    except FileNotFoundError:
        return None, None


model, feature_cols = load_model()

if model is None:
    st.warning(
        "⚠️ 학습된 모델이 존재하지 않습니다. 먼저 사이드바에서 **[7_위험 예측 분류 모델 학습 최적화]** 페이지로 이동해 모델 학습을 완료해 주세요!"
    )
    st.stop()

ASOS_SERVICE_KEY = (
    "feb2bfabd299d5d05e89c7aec49ba7e706112603e76549a92e868bd86ec60323"
)
ASOS_URL = (
    "http://apis.data.go.kr/1360000/AsosDalyInfoService/getWthrDataList"
)
STN_ID = "281"  # 영천 관측소

# 영천시 주요 문화재 명단 및 고유 재질/노출 환경 매핑
YEONGCHEON_HERITAGES = [
    {"문화재명": "영천 은해사 대웅전", "종류": "보물", "material": "목조", "exposure": "실외"},
    {"문화재명": "영천 거동사 대웅전", "종류": "국보", "material": "목조", "exposure": "실외"},
    {"문화재명": "영천 백의리 삼층석탑", "종류": "보물", "material": "석조", "exposure": "실외"},
    {"문화재명": "영천 자천리 목조한옥", "종류": "국보", "material": "목조", "exposure": "실외"},
    {"문화재명": "영천 임고서원 표충사", "종류": "지정문화재", "material": "목조", "exposure": "반실외"},
    {"문화재명": "영천 오계동 금속유물 소장품", "종류": "지정문화재", "material": "금속", "exposure": "실내"},
    {"문화재명": "영천 최남주 고택 회화작품", "종류": "지정문화재", "material": "회화", "exposure": "실내"},
    {"문화재명": "영천 자양면 사찰 종각", "종류": "일반문화재", "material": "금속", "exposure": "실외"},
]


def fetch_latest_prediction_data():
    # 데이터 집계 지연을 고려하여 2일 전부터 최근 12일간의 데이터를 조회
    today = datetime.date.today() - datetime.timedelta(days=2)
    start_date = today - datetime.timedelta(days=12)
    start_str = start_date.strftime("%Y%m%d")
    end_str = today.strftime("%Y%m%d")

    params = {
        "serviceKey": ASOS_SERVICE_KEY,
        "numOfRows": "20",
        "pageNo": "1",
        "dataType": "JSON",
        "dataCd": "ASOS",
        "dateCd": "DAY",
        "startDt": start_str,
        "endDt": end_str,
        "stnIds": STN_ID,
    }
    try:
        response = requests.get(ASOS_URL, params=params, timeout=30)
        res_json = response.json()

        if (
            "response" not in res_json
            or "body" not in res_json["response"]
            or "items" not in res_json["response"]["body"]
        ):
            st.error("기상청 API 응답 구조를 불러오지 못했습니다.")
            return None

        items_data = res_json["response"]["body"]["items"]
        if not items_data or "item" not in items_data:
            st.error("조회된 기상 데이터 항목이 없습니다.")
            return None

        items = items_data["item"]
        weather = pd.DataFrame(items)

        weather = weather[[
            "tm",
            "avgTa",
            "maxTa",
            "minTa",
            "avgRhm",
            "sumRn",
            "avgWs",
            "sumSsHr",
            "avgTs",
        ]].copy()
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

        air_url = "https://docs.google.com/spreadsheets/d/1fBEnheVOP-23Hmv_5ZJZVy6m9VmNkpVd2XutOdmlYc8/export?format=csv&gid=700055413"
        air = pd.read_csv(air_url)
        air["date"] = pd.to_datetime(air["date"], errors="coerce")

        df = (
            pd.merge(weather, air, on="date", how="left")
            .sort_values("date")
            .reset_index(drop=True)
        )

        df["temp_range"] = df["temp_max"] - df["temp_min"]
        df["humidity_std3"] = df["humidity"].rolling(3, min_periods=1).std()
        df["rainfall_7d"] = df["rainfall"].rolling(7, min_periods=1).sum()
        df["high_humidity_risk"] = (df["humidity"] >= 75).rolling(
            3, min_periods=1
        ).sum()
        df["weathering_risk"] = (
            df["temp_range"] * 0.4
            + df["humidity_std3"] * 0.3
            + df["wind_speed"] * 0.3
        )
        df["mold_risk"] = (
            (df["humidity"] >= 75) & (df["ground_temp"] >= 15)
        ).astype(int)
        df["pm_load"] = (df["pm10"] + df["pm25"]).rolling(3, min_periods=1).sum()
        df["acid_risk"] = df["so2"] * 0.6 + df["no2"] * 0.4
        df["oxidation_risk"] = df["o3"] * 0.7 + df["pm25"] * 0.3
        df["corrosion_risk"] = df["humidity"] * 0.5 + df["so2"] * 0.5
        df = df.fillna(0)

        return df
    except Exception as e:
        st.error(f"데이터 수집 중 오류 발생: {e}")
        return None


# 실행 버튼
if st.button("🚀 영천시 문화재 최신 위험도 분석 실행"):
    with st.spinner("최근 기상 및 대기오염 데이터를 동기화하여 영천시 문화재별 위험도를 예측 중입니다..."):
        df_recent = fetch_latest_prediction_data()

        if df_recent is not None and not df_recent.empty:
            latest_row = df_recent.iloc[-1].copy()
            target_date = latest_row["date"].strftime("%Y-%m-%d")

            st.success(f"✅ 최신 데이터 기준일: **{target_date}** (영천 관측소 및 대기 데이터 연동 완료)")

            results = []
            for heritage in YEONGCHEON_HERITAGES:
                row_data = latest_row.to_dict()
                row_data["material"] = heritage["material"]
                row_data["exposure"] = heritage["exposure"]

                single_df = pd.DataFrame([row_data])
                X_input = single_df[[
                    "temp_avg", "temp_max", "temp_min", "humidity", "rainfall",
                    "wind_speed", "solar_radiation", "ground_temp", "pm10", "pm25",
                    "o3", "no2", "co", "so2", "temp_range", "humidity_std3",
                    "rainfall_7d", "high_humidity_risk", "weathering_risk",
                    "mold_risk", "pm_load", "acid_risk", "oxidation_risk",
                    "corrosion_risk", "material", "exposure"
                ]]
                
                X_input_encoded = pd.get_dummies(X_input, columns=["material", "exposure"])
                for col in feature_cols:
                    if col not in X_input_encoded.columns:
                        X_input_encoded[col] = 0
                X_input_encoded = X_input_encoded[feature_cols]

                pred = model.predict(X_input_encoded)[0]

                results.append({
                    "문화재명": heritage["문화재명"],
                    "구분": heritage["종류"],
                    "재질": heritage["material"],
                    "노출환경": heritage["exposure"],
                    "예측 위험등급": pred,
                    "평균기온(℃)": latest_row["temp_avg"],
                    "습도(%)": latest_row["humidity"],
                    "미세먼지(PM10)": latest_row["pm10"],
                })

            result_df = pd.DataFrame(results)

            def highlight_risk(val):
                if val == "위험":
                    return "background-color: #ffcccc; color: #990000; font-weight: bold;"
                elif val == "주의":
                    return "background-color: #ffe5cc; color: #994c00; font-weight: bold;"
                else:
                    return "background-color: #ccffcc; color: #006600; font-weight: bold;"

            st.markdown("---")
            st.subheader(f"📊 영천시 주요 문화재별 최신 위험 예측 현황 ({target_date})")
            
            # Pandas 스타일 적용 (.applymap 대신 최신 .map 사용)
            st.dataframe(
                result_df.style.map(highlight_risk, subset=["예측 위험등급"]),
                use_container_width=True,
                height=350
            )

            st.markdown("---")
            m1, m2, m3 = st.columns(3)
            danger_count = (result_df["예측 위험등급"] == "위험").sum()
            caution_count = (result_df["예측 위험등급"] == "주의").sum()
            safe_count = (result_df["예측 위험등급"] == "안전").sum()

            m1.metric("🚨 위험 단계 문화재 수", f"{danger_count} 곳")
            m2.metric("⚠️ 주의 단계 문화재 수", f"{caution_count} 곳")
            m3.metric("✅ 안전 단계 문화재 수", f"{safe_count} 곳")

            st.markdown("---")
            st.subheader("📈 최근 7일간 전체 영천시 기상 기반 위험도 추이")
            
            df_7days = df_recent.tail(7).copy()
            trend_results = []
            for _, r in df_7days.iterrows():
                r_dict = r.to_dict()
                r_dict["material"] = "목조"
                r_dict["exposure"] = "실외"
                s_df = pd.DataFrame([r_dict])
                X_in = s_df[list(X_input.columns)]
                X_in_enc = pd.get_dummies(X_in, columns=["material", "exposure"])
                for col in feature_cols:
                    if col not in X_in_enc.columns:
                        X_in_enc[col] = 0
                X_in_enc = X_in_enc[feature_cols]
                p = model.predict(X_in_enc)[0]
                trend_results.append({"date": r["date"].strftime("%Y-%m-%d"), "위험등급": p})
            
            trend_df = pd.DataFrame(trend_results)
            st.bar_chart(trend_df.set_index("date")["위험등급"].value_counts())
