import streamlit as st
import pandas as pd
import requests
import urllib.parse
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

# ============================================
# API KEY 설정
# ============================================
SERVICE_KEY = "feb2bfabd299d5d05e89c7aec49ba7e706112603e76549a92e868bd86ec60323"

# ============================================
# 1. 기상청 지상(AWS) 관측 데이터 수집
# ============================================
AWS_10MIN_URL = "http://apis.data.go.kr/1360000/SfcMtsInfoService/getAws10Min"

# 영천 주요 관측소 목록 (지점 번호)
STATION_MAP = {
    "영천(종합)": "281",
    "신령": "853",
    "청통": "854",
    "화북": "855"
}

def get_latest_tm_10min():
    """기상청 AWS 전송 지연을 고려해 15분 전 10분 단위 시각 생성"""
    now = datetime.now(ZoneInfo("Asia/Seoul"))
    # 안전하게 15분 전 데이터 요청 (API 업데이트 지연 대비)
    target = now - timedelta(minutes=15)
    minute = (target.minute // 10) * 10
    target = target.replace(minute=minute, second=0, microsecond=0)
    
    api_tm = target.strftime("%Y%m%d%H%M")
    display_tm = target.strftime("%Y-%m-%d %H:%M")
    return api_tm, display_tm

def degree_to_direction(deg):
    """풍향(도) -> 16방위 변환"""
    try:
        deg = float(deg)
        if deg < 0 or deg > 360:
            return "-"
        dirs = [
            "북", "북북동", "북동", "동북동", 
            "동", "남동", "남", "남남서", 
            "남서", "서남서", "서", "서북서", 
            "북서", "북북서", "북"
        ]
        idx = int((deg + 11.25) / 22.5) % 16
        return dirs[idx]
    except (ValueError, TypeError):
        return "-"

def safe_val(val, default="-"):
    """API 응답 값 유효성 체크 함수"""
    if val is None or val == "" or str(val).startswith("-99"):
        return default
    return val

def get_aws_weather_data(stn_id, api_tm):
    """특정 지점의 AWS 10분 단위 데이터 수집 및 디버깅용 원본 반환"""
    aws_data = {
        "temp": "-",
        "humidity": "-",
        "rainfall": "-",
        "wind_speed": "-",
        "wind_dir": "-"
    }
    raw_json = {} # 디버깅용 저장소
    
    try:
        # URL 인코딩 키 문제 방지
        decoded_key = urllib.parse.unquote(SERVICE_KEY)
        
        params = {
            "serviceKey": decoded_key,
            "pageNo": "1",
            "numOfRows": "10",
            "dataType": "JSON",
            "stnId": str(stn_id),
            "tm": api_tm
        }

        response = requests.get(AWS_10MIN_URL, params=params, timeout=10)
        
        if response.status_code == 200:
            raw_json = response.json() # 디버깅을 위해 응답 전체 저장
            items = raw_json.get("response", {}).get("body", {}).get("items", {}).get("item", [])
            
            if items:
                item = items[0] if isinstance(items, list) else items
                
                # 대소문자 키 대응 (API 사양 보완)
                ta = item.get("ta") or item.get("TA")
                hm = item.get("hm") or item.get("HM")
                rn = item.get("rn1hr") or item.get("rn10m") or item.get("RN")
                ws = item.get("ws") or item.get("WS")
                wd = item.get("wd") or item.get("WD")

                aws_data["temp"] = safe_val(ta)
                aws_data["humidity"] = safe_val(hm)
                aws_data["rainfall"] = safe_val(rn, default="0.0")
                aws_data["wind_speed"] = safe_val(ws)
                aws_data["wind_dir"] = degree_to_direction(wd if wd is not None else -1)
        else:
            raw_json = {"error": f"HTTP Status Code {response.status_code}"}
    except Exception as e:
        raw_json = {"error": str(e)}
        
    return aws_data, raw_json

# 최신 10분 관측 시각 및 데이터 수집
api_tm, display_tm = get_latest_tm_10min()

weather_results = {}
debug_aws_responses = {} # 디버깅 데이터 저장

for name, stn_id in STATION_MAP.items():
    w_data, raw_res = get_aws_weather_data(stn_id, api_tm)
    weather_results[name] = w_data
    debug_aws_responses[name] = raw_res


# ============================================
# 2. 대기오염 최신 데이터
# ============================================
AIR_URL = "https://apis.data.go.kr/B552584/ArpltnInforInqireSvc/getCtprvnRltmMesureDnsty"

pm10 = pm25 = o3 = no2 = co = so2 = data_time = "-"
debug_air_response = {} # 대기오염 디버깅 데이터 저장

try:
    decoded_key = urllib.parse.unquote(SERVICE_KEY)
    air_params = {
        "serviceKey": decoded_key,
        "returnType": "json",
        "numOfRows": "100",
        "pageNo": "1",
        "sidoName": "경북",
        "ver": "1.0"
    }

    air_response = requests.get(AIR_URL, params=air_params, timeout=10)
    
    if air_response.status_code == 200:
        debug_air_response = air_response.json()
        items = debug_air_response.get("response", {}).get("body", {}).get("items", [])

        target = None
        for item in items:
            if "영천" in item.get("stationName", ""):
                target = item
                break

        if target:
            data_time = safe_val(target.get("dataTime"))
            pm10 = safe_val(target.get("pm10Value"))
            pm25 = safe_val(target.get("pm25Value"))
            o3 = safe_val(target.get("o3Value"))
            no2 = safe_val(target.get("no2Value"))
            co = safe_val(target.get("coValue"))
            so2 = safe_val(target.get("so2Value"))
    else:
        debug_air_response = {"error": f"HTTP Status Code {air_response.status_code}"}
except Exception as e:
    debug_air_response = {"error": str(e)}


# ==========================================================
# 페이지 설정 및 UI
# ==========================================================
st.set_page_config(
    page_title="공공 환경 데이터 기반 영천 지역 문화재 훼손 위험 예측",
    page_icon="🏛",
    layout="wide"
)

try:
    df = pd.read_csv("data/processed/yc_heritage_detail_enriched.csv")
except Exception:
    df = pd.DataFrame()

# 제목
st.markdown("<h1 style='font-size:30px;'>🏛 공공 환경 데이터 기반 영천 지역 문화재 훼손 위험 예측</h1>", unsafe_allow_html=True)
st.markdown("영천 지역 문화재와 공공 환경데이터를 분석하여 문화재 훼손 위험을 사전에 예측하는 데이터 분석 프로젝트 입니다.")
st.divider()

# 카드 스타일
aws_card_style = "background-color:#f8f9fa; padding:18px; border-radius:16px; border:1px solid #e5e7eb; box-shadow:0 4px 12px rgba(0,0,0,0.04); min-height:280px; position:relative;"
bottom_card_style = "background-color:#f8f9fa; padding:22px; border-radius:20px; border:1px solid #e5e7eb; box-shadow:0 4px 12px rgba(0,0,0,0.05); min-height:260px; position:relative;"
title_style = "font-size:18px; font-weight:700; margin-bottom:8px; color:#1f2937;"
label_style = "font-size:13px; color:#6b7280; margin-bottom:2px;"
value_style = "font-size:18px; font-weight:700; color:#111827; margin-bottom:10px;"
time_style = "font-size:12px; color:#9ca3af; margin-top:15px;"


# --------------------------------------------
# 1행 : 기상 관측소 실시간 현황
# --------------------------------------------
st.markdown('<h3 style="font-size:22px; margin-bottom:15px;">🌦 영천시 지역별 실시간 기상 현황 (AWS 10분 관측)</h3>', unsafe_allow_html=True)
aws_cols = st.columns(4)

for idx, (stn_name, w_data) in enumerate(weather_results.items()):
    with aws_cols[idx]:
        st.markdown(
            f"""
<div style="{aws_card_style}">
    <div style="{title_style}">📍 {stn_name}</div>
    <hr style="margin: 8px 0;">
    <div style="display:grid; grid-template-columns:1fr 1fr; gap:8px; margin-top:10px;">
        <div><div style="{label_style}">🌡 기온</div><div style="{value_style}">{w_data['temp']} °C</div></div>
        <div><div style="{label_style}">💧 습도</div><div style="{value_style}">{w_data['humidity']} %</div></div>
        <div><div style="{label_style}">🌧 강수량</div><div style="{value_style}">{w_data['rainfall']} mm</div></div>
        <div><div style="{label_style}">💨 풍속/풍향</div><div style="{value_style}">{w_data['wind_speed']} m/s<br><span style="font-size:13px; font-weight:normal; color:#4b5563;">({w_data['wind_dir']})</span></div></div>
    </div>
    <div style="{time_style}">⏱ 관측({stn_name}): {display_tm}</div>
</div>
            """,
            unsafe_allow_html=True
        )

st.markdown("<div style='height: 25px;'></div>", unsafe_allow_html=True)

# --------------------------------------------
# 2행 : 대기오염 현황 & 문화재 현황
# --------------------------------------------
st.markdown('<h3 style="font-size:22px; margin-bottom:15px;">🌿 대기 환경 및 문화재 현황</h3>', unsafe_allow_html=True)
bot_left, bot_right = st.columns([2.2, 1.0])

with bot_left:
    st.markdown(
        f"""
<div style="{bottom_card_style}">
    <div style="{title_style}">🌫 대기오염 현황 (영천 측정소)</div>
    <hr style="margin: 8px 0;">
    <div style="display:grid; grid-template-columns:1fr 1fr 1fr; gap:12px; margin-top:12px;">
        <div>
            <div style="{label_style}">PM10 (미세먼지)</div><div style="{value_style}">{pm10} ㎍/㎥</div>
            <div style="{label_style}">O₃ (오존)</div><div style="{value_style}">{o3} ppm</div>
        </div>
        <div>
            <div style="{label_style}">PM2.5 (초미세먼지)</div><div style="{value_style}">{pm25} ㎍/㎥</div>
            <div style="{label_style}">NO₂ (이산화질소)</div><div style="{value_style}">{no2} ppm</div>
        </div>
        <div>
            <div style="{label_style}">CO (일산화탄소)</div><div style="{value_style}">{co} ppm</div>
            <div style="{label_style}">SO₂ (아황산가스)</div><div style="{value_style}">{so2} ppm</div>
        </div>
    </div>
    <div style="{time_style}">⏱ 측정 시각 : {data_time}</div>
</div>
        """,
        unsafe_allow_html=True
    )

with bot_right:
    st.markdown(
        f"""
<div style="{bottom_card_style}">
    <div style="{title_style}">🏛 문화재 현황</div>
    <hr style="margin: 8px 0;">
    <div style="margin-top:20px;">
        <div style="{label_style}">분석 문화재 수</div>
        <div style="font-size:28px; font-weight:700; color:#111827; margin-top:5px;">{len(df)}개</div>
    </div>
</div>
        """,
        unsafe_allow_html=True
    )

st.divider()
st.caption("선화여고 - 영천 헤리티지 AI 탐구단")

# ==========================================================
# 3. 개발자 디버깅용 섹션 (여기서 에러 원인을 확인하세요!)
# ==========================================================
with st.expander("🔍 API 응답 데이터 디버깅 (개발자용 확인 - 데이터가 안뜰때 여기를 확인하세요)"):
    st.write("요청 API 시각 파라미터(tm):", api_tm)
    
    st.subheader("1. 기상청 AWS 원본 데이터")
    st.write("영천(종합) 관측소 응답:")
    st.json(debug_aws_responses.get("영천(종합)", {}))
    
    st.subheader("2. 환경공단 대기오염 원본 데이터")
    st.json(debug_air_response)
