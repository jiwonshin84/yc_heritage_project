import os
import urllib.parse
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import requests
import streamlit as st
from streamlit_autorefresh import st_autorefresh

# ============================================
# Streamlit 페이지 기본 설정
# ============================================
st.set_page_config(
    page_title="공공 환경 데이터 기반 영천 지역 실시간 환경 현황",
    page_icon="🏛",
    layout="wide"
)

# 60초(60,000밀리초)마다 자동으로 페이지를 새로고침 (최대 1,000회)
count = st_autorefresh(interval=60000, limit=1000, key="weather_auto_refresh")

# ============================================
# API KEY 및 기본 설정
# ============================================
# 기상청 API 허브 전용 인증키
KMA_AUTH_KEY = "XDdcOK8kT5C3XDivJN-Qtg"

# 한국환경공단 API 인증키
AIR_SERVICE_KEY = "feb2bfabd299d5d05e89c7aec49ba7e706112603e76549a92e868bd86ec60323"

# 기상청 AWS 매분자료 조회 API URL
AWS_MIN_URL = "https://apihub.kma.go.kr/api/typ01/cgi-bin/url/nph-aws2_min"

# 한국환경공단 시도별 실시간 대기오염 측정 정보 API URL
AIR_URL = "http://apis.data.go.kr/B552584/ArpltnInforInqireSvc/getCtprvnRltmMesureDnsty"

# 영천 주요 관측소 지점 번호
STATION_MAP = {
    "영천(종합)": "281",
    "신령": "853",
    "청통": "854",
    "화북": "855"
}

# ============================================
# 헬퍼 함수 정의
# ============================================
def get_current_kst_time():
    """현재 한국 표준시 표시용 문자열 생성"""
    now = datetime.now(ZoneInfo("Asia/Seoul"))
    return now.strftime("%Y-%m-%d %H:%M:%S")

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
    """응답 값 유효성 체크"""
    if val is None or val == "" or str(val).startswith("-99") or str(val) == "-":
        return default
    return str(val)

# ============================================
# 데이터 수집 함수 (AWS & 대기오염)
# ============================================
def get_aws_weather_data(stn_id):
    """기상청 API 허브(nph-aws2_min)를 이용한 최신 1분 관측 데이터 수집"""
    aws_data = {
        "temp": "-", "humidity": "-", "rainfall": "-", 
        "wind_speed": "-", "wind_dir": "-", "obs_time": "-"
    }
    raw_text = ""
    
    try:
        params = {
            "authKey": KMA_AUTH_KEY,
            "stn": str(stn_id),
            "disp": "1",  # 쉼표(,) 구분자 형식
            "help": "0"
        }
        response = requests.get(AWS_MIN_URL, params=params, timeout=10)
        
        if response.status_code == 200:
            raw_text = response.text
            lines = [line.strip() for line in raw_text.split("\n") if line.strip() and not line.startswith("#")]
            
            if lines:
                data_line = lines[-1]
                parts = [p.strip() for p in data_line.split(",")]
                
                # 순서: TM, STN, WD1, WS1, WDS, WSS, WD10, WS10, TA, RE, RN-15m, RN-60m, RN-12H, RN-DAY, HM, ...
                if len(parts) >= 15:
                    aws_data["obs_time"] = parts[0]                      # TM (관측시각)
                    aws_data["wind_dir"] = degree_to_direction(parts[2])  # WD1 (1분 평균 풍향)
                    aws_data["wind_speed"] = safe_val(parts[3])          # WS1 (1분 평균 풍속)
                    aws_data["temp"] = safe_val(parts[8])                # TA (1분 평균 기온)
                    aws_data["rainfall"] = safe_val(parts[10], "0.0")   # RN-15m
                    aws_data["humidity"] = safe_val(parts[14])            # HM (분 평균 상대습도)
        else:
            raw_text = f"HTTP Error {response.status_code}: {response.text}"
    except Exception as e:
        raw_text = f"Exception: {str(e)}"
        
    return aws_data, raw_text

def get_air_pollution_data():
    """한국환경공단 영천 대기오염 측정 정보 데이터 수집"""
    air_res = {
        "pm10": "-", "pm25": "-", "o3": "-", 
        "no2": "-", "co": "-", "so2": "-", "data_time": "-"
    }
    raw_json = {}
    
    try:
        decoded_key = urllib.parse.unquote(AIR_SERVICE_KEY)
        air_params = {
            "serviceKey": decoded_key,
            "returnType": "json",
            "numOfRows": "100",
            "pageNo": "1",
            "sidoName": "경북",
            "ver": "1.0"
        }
        response = requests.get(AIR_URL, params=air_params, timeout=15)
        
        if response.status_code == 200:
            raw_json = response.json()
            items = raw_json.get("response", {}).get("body", {}).get("items", [])
            
            target = None
            for item in items:
                if "영천" in item.get("stationName", ""):
                    target = item
                    break
            
            if target:
                air_res["data_time"] = safe_val(target.get("dataTime"))
                air_res["pm10"] = safe_val(target.get("pm10Value"))
                air_res["pm25"] = safe_val(target.get("pm25Value"))
                air_res["o3"] = safe_val(target.get("o3Value"))
                air_res["no2"] = safe_val(target.get("no2Value"))
                air_res["co"] = safe_val(target.get("coValue"))
                air_res["so2"] = safe_val(target.get("so2Value"))
        else:
            raw_json = {"error": f"HTTP Status Code {response.status_code}"}
    except Exception as e:
        raw_json = {"error": str(e)}
        
    return air_res, raw_json

# ============================================
# 데이터 수집 실행
# ============================================
current_kst = get_current_kst_time()

weather_results = {}
debug_aws_responses = {}

for name, stn_id in STATION_MAP.items():
    w_data, raw_res = get_aws_weather_data(stn_id)
    weather_results[name] = w_data
    debug_aws_responses[name] = raw_res

air_data, debug_air_response = get_air_pollution_data()

# ============================================
# UI 레이아웃 구성
# ============================================
try:
    df = pd.read_csv("data/processed/yc_heritage_detail_enriched.csv")
except Exception:
    df = pd.DataFrame()

st.markdown("<h1 style='font-size:30px;'>🏛 공공 환경 데이터 기반 영천 지역 실시간 환경 현황</h1>", unsafe_allow_html=True)
st.markdown("영천 지역 문화재 보존 관리를 위한 실시간 기상 관측 데이터(AWS) 및 대기 오염 현황 모니터링 페이지입니다.")

# 자동 새로고침 상태 안내 표시
st.info(f"🔄 **실시간 자동 갱신 중** (마지막 화면 동기화: {current_kst}) — 1분마다 최신 관측값으로 자동 업데이트됩니다.")

st.divider()

# 카드 스타일 정의
aws_card_style = "background-color:#f8f9fa; padding:18px; border-radius:16px; border:1px solid #e5e7eb; box-shadow:0 4px 12px rgba(0,0,0,0.04); min-height:280px;"
bottom_card_style = "background-color:#f8f9fa; padding:22px; border-radius:20px; border:1px solid #e5e7eb; box-shadow:0 4px 12px rgba(0,0,0,0.05); min-height:260px;"
title_style = "font-size:18px; font-weight:700; margin-bottom:8px; color:#1f2937;"
label_style = "font-size:13px; color:#6b7280; margin-bottom:2px;"
value_style = "font-size:18px; font-weight:700; color:#111827; margin-bottom:10px;"
time_style = "font-size:12px; color:#9ca3af; margin-top:15px;"

# --------------------------------------------
# 1행 : 기상 관측소 실시간 현황 (최신 1분 데이터)
# --------------------------------------------
st.markdown('<h3 style="font-size:22px; margin-bottom:15px;">🌦 영천시 지역별 실시간 기상 현황 (AWS 1분 관측)</h3>', unsafe_allow_html=True)
aws_cols = st.columns(4)

for idx, (stn_name, w_data) in enumerate(weather_results.items()):
    obs_time_fmt = w_data['obs_time']
    if len(obs_time_fmt) == 12:
        obs_time_fmt = f"{obs_time_fmt[:4]}-{obs_time_fmt[4:6]}-{obs_time_fmt[6:8]} {obs_time_fmt[8:10]}:{obs_time_fmt[10:12]}"
        
    with aws_cols[idx]:
        st.markdown(
            f"""
<div style="{aws_card_style}">
    <div style="{title_style}">📍 {stn_name}</div>
    <hr style="margin: 8px 0;">
    <div style="display:grid; grid-template-columns:1fr 1fr; gap:8px; margin-top:10px;">
        <div><div style="{label_style}">🌡 기온 (1분)</div><div style="{value_style}">{w_data['temp']} °C</div></div>
        <div><div style="{label_style}">💧 습도 (1분)</div><div style="{value_style}">{w_data['humidity']} %</div></div>
        <div><div style="{label_style}">🌧 강수량</div><div style="{value_style}">{w_data['rainfall']} mm</div></div>
        <div><div style="{label_style}">💨 풍속/풍향</div><div style="{value_style}">{w_data['wind_speed']} m/s<br><span style="font-size:13px; font-weight:normal; color:#4b5563;">({w_data['wind_dir']})</span></div></div>
    </div>
    <div style="{time_style}">⏱ 관측 시각: {obs_time_fmt}</div>
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
            <div style="{label_style}">PM10 (미세먼지)</div><div style="{value_style}">{air_data['pm10']} ㎍/㎥</div>
            <div style="{label_style}">O₃ (오존)</div><div style="{value_style}">{air_data['o3']} ppm</div>
        </div>
        <div>
            <div style="{label_style}">PM2.5 (초미세먼지)</div><div style="{value_style}">{air_data['pm25']} ㎍/㎥</div>
            <div style="{label_style}">NO₂ (이산화질소)</div><div style="{value_style}">{air_data['no2']} ppm</div>
        </div>
        <div>
            <div style="{label_style}">CO (일산화탄소)</div><div style="{value_style}">{air_data['co']} ppm</div>
            <div style="{label_style}">SO₂ (아황산가스)</div><div style="{value_style}">{air_data['so2']} ppm</div>
        </div>
    </div>
    <div style="{time_style}">⏱ 측정 시각 : {air_data['data_time']}</div>
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

# --------------------------------------------
# 3행 : 개발자 디버깅용 확장 섹션
# --------------------------------------------
with st.expander("🔍 API 응답 데이터 디버깅 (개발자용 확인)"):
    st.write("페이지 갱신 횟수 (Auto-Refresh Count):", count)
    st.write("현재 시스템 시각:", current_kst)
    
    st.subheader("1. 기상청 AWS 최신 1분 원본 데이터 (영천 종합)")
    st.code(debug_aws_responses.get("영천(종합)", ""), language="text")
    
    st.subheader("2. 환경공단 대기오염 원본 데이터")
    st.json(debug_air_response)
