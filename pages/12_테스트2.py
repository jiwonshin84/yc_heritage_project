import datetime
import os
import joblib
import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="영천시 전체 문화재 위험도 예측", layout="wide")

st.title("🏛️ 영천시 전체 문화재 실시간 위험도 예측 시스템")
st.markdown("##### 📌 최근 7일간의 기상 및 대기오염 현황 데이터를 조회합니다.")

# 1. API 키 및 설정 변수를 함수보다 위쪽(상단)으로 이동
ASOS_SERVICE_KEY = "feb2bfabd299d5d05e89c7aec49ba7e706112603e76549a92e868bd86ec60323"
ASOS_URL = "http://apis.data.go.kr/1360000/AsosDalyInfoService/getWthrDataList"
STN_ID = "281"  # 영천 관측소


@st.cache_resource
def load_model_and_heritage():
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    model_path = os.path.join(root_dir, "best_rf_model.pkl")
    features_path = os.path.join(root_dir, "model_features.pkl")
    heritage_csv_path = os.path.join(root_dir, "data", "processed", "yc_heritage_feature.csv")

    try:
        model = joblib.load(model_path)
        features = joblib.load(features_path)
    except FileNotFoundError:
        return None, None, None

    try:
        heritage_df = pd.read_csv(heritage_csv_path)
    except Exception as e:
        heritage_df = None
        st.error(f"문화재 기본 정보 파일(`yc_heritage_feature.csv`) 로드 실패: {e}")

    return model, features, heritage_df


model, feature_cols, heritage_df = load_model_and_heritage()

if model is None:
    st.warning("⚠️ 학습된 모델이 존재하지 않습니다.")
    st.stop()

if heritage_df is None:
    st.stop()


def fetch_recent_7days_data():
    """오늘 날짜 기준으로 최근 7일간의 기상 및 대기오염 데이터를 수집합니다."""
    today = datetime.date.today()
    start_date = today - datetime.timedelta(days=7)
    
    start_str = start_date.strftime("%Y%m%d")
    end_str = today.strftime("%Y%m%d")

    params = {
        "serviceKey": ASOS_SERVICE_KEY,
        "numOfRows": "10",
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

        items_data = res_json.get("response", {}).get("body", {}).get("items", {})
        if not items_data or "item" not in items_data:
            st.error("조회된 기상 데이터 항목이 없습니다.")
            return None

        weather = pd.DataFrame(items_data["item"])
        
        weather = weather[[
            "tm", "avgTa", "maxTa", "minTa", "avgRhm", 
            "sumRn", "avgWs", "sumSsHr", "avgTs"
        ]].copy()
        
        weather.columns = [
            "date", "temp_avg", "temp_max", "temp_min", 
            "humidity", "rainfall", "wind_speed", "solar_radiation", "ground_temp"
        ]
        weather["date"] = pd.to_datetime(weather["date"], errors="coerce")

        numeric_cols = ["temp_avg", "temp_max", "temp_min", "humidity", "rainfall", "wind_speed", "solar_radiation", "ground_temp"]
        for col in numeric_cols:
            weather[col] = pd.to_numeric(weather[col], errors="coerce")
        weather["rainfall"] = weather["rainfall"].fillna(0)

        air_url = "https://docs.google.com/spreadsheets/d/1fBEnheVOP-23Hmv_5ZJZVy6m9VmNkpVd2XutOdmlYc8/export?format=csv&gid=700055413"
        air = pd.read_csv(air_url)
        air["date"] = pd.to_datetime(air["date"], errors="coerce")

        df = pd.merge(weather, air, on="date", how="left").sort_values("date").reset_index(drop=True)
        return df

    except Exception as e:
        st.error(f"데이터 수집 중 오류 발생: {e}")
        return None


# 실행 버튼
if st.button("📊 오늘 기준 최근 7일간 기상 및 대기오염 현황 조회"):
    with st.spinner("최근 7일간의 데이터를 불러오는 중입니다..."):
        df_7days = fetch_recent_7days_data()
        
        if df_7days is not None and not df_7days.empty:
            st.success("✅ 최근 7일간 기상 및 대기오염 데이터 연동 완료")
            
            display_df = df_7days[[
                "date", "temp_avg", "temp_max", "temp_min", 
                "humidity", "rainfall", "pm10", "pm25", "o3", "so2"
            ]].copy()
            
            display_df["date"] = display_df["date"].dt.strftime("%Y-%m-%d")
            display_df.columns = [
                "날짜", "평균기온(℃)", "최고기온(℃)", "최저기온(℃)", 
                "습도(%)", "강수량(mm)", "미세먼지(PM10)", "초미세먼지(PM2.5)", "오존(O3)", "아황산가스(SO2)"
            ]
            
            st.dataframe(display_df, use_container_width=True)
        else:
            st.warning("조회된 데이터가 없습니다.")
