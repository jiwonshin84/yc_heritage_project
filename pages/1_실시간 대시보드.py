from datetime import datetime
import os
import urllib.parse
from zoneinfo import ZoneInfo

import folium
import pandas as pd
import requests
import streamlit as st
from streamlit_autorefresh import st_autorefresh
from streamlit_folium import st_folium

# ============================================
# Streamlit 페이지 기본 설정
# ============================================
st.set_page_config(
    page_title="공공 환경 데이터 기반 영천 지역 실시간 환경 현황",
    page_icon="🏠",
    layout="wide",
)

# 상단 기본 여백을 완전히 없애서 최상단부터 채우도록 설정
st.markdown(
    """
    <style>
        .block-container {
            padding-top: 0rem !important;
            padding-bottom: 2rem !important;
            margin-top: 0rem !important;
        }
        header {
            visibility: hidden;
        }
    </style>
""",
    unsafe_allow_html=True,
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

# 관측소 좌표 설정
STATION_MAP = {
    "신녕": {"id": "853", "lat": 36.0150, "lon": 128.6100},
    "청통": {"id": "854", "lat": 36.0250, "lon": 128.7800},
    "화북": {"id": "855", "lat": 36.1700, "lon": 128.9300},
    "영천(종합)": {"id": "281", "lat": 35.9650, "lon": 128.9400},
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
        "북",
        "북북동",
        "북동",
        "동북동",
        "동",
        "남동",
        "남",
        "남남서",
        "남서",
        "서남서",
        "서",
        "서북서",
        "북서",
        "북북서",
        "북",
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
# 데이터 수집 함수 (AWS & 대기오염 + 캐싱 적용)
# ============================================
@st.cache_data(ttl=60)
def get_aws_weather_data(stn_id):
  """기상청 API 허브(nph-aws2_min)를 이용한 최신 1분 관측 데이터 수집 (60초 캐싱)"""
  aws_data = {
      "temp": "-",
      "humidity": "-",
      "rainfall": "-",
      "wind_speed": "-",
      "wind_dir": "-",
      "obs_time": "-",
  }

  try:
    params = {"authKey": KMA_AUTH_KEY, "stn": str(stn_id), "disp": "1", "help": "0"}
    response = requests.get(AWS_MIN_URL, params=params, timeout=10)

    if response.status_code == 200:
      lines = [
          line.strip()
          for line in response.text.split("\n")
          if line.strip() and not line.startswith("#")
      ]

      if lines:
        data_line = lines[-1]
        parts = [p.strip() for p in data_line.split(",")]

        if len(parts) >= 15:
          aws_data["obs_time"] = parts[0]
          aws_data["wind_dir"] = degree_to_direction(parts[2])
          aws_data["wind_speed"] = safe_val(parts[3])
          aws_data["temp"] = safe_val(parts[8])
          aws_data["rainfall"] = safe_val(parts[10], "0.0")
          aws_data["humidity"] = safe_val(parts[14])
  except Exception:
    pass

  return aws_data


@st.cache_data(ttl=600)
def get_air_pollution_data():
  """한국환경공단 영천 대기오염 측정 정보 데이터 수집 (10분 캐싱)"""
  air_res = {
      "pm10": "-",
      "pm25": "-",
      "o3": "-",
      "no2": "-",
      "co": "-",
      "so2": "-",
      "data_time": "-",
  }

  try:
    decoded_key = urllib.parse.unquote(AIR_SERVICE_KEY)
    air_params = {
        "serviceKey": decoded_key,
        "returnType": "json",
        "numOfRows": "100",
        "pageNo": "1",
        "sidoName": "경북",
        "ver": "1.0",
    }
    response = requests.get(AIR_URL, params=air_params, timeout=15)

    if response.status_code == 200:
      items = (
          response.json().get("response", {}).get("body", {}).get("items", [])
      )

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
  except Exception:
    pass

  return air_res


# ============================================
# 데이터 수집 실행
# ============================================
current_kst = get_current_kst_time()

weather_results = {}
for name, info in STATION_MAP.items():
  weather_results[name] = get_aws_weather_data(info["id"])

air_data = get_air_pollution_data()

# ============================================
# UI 레이아웃 구성
# ============================================
try:
  df = pd.read_csv("data/processed/yc_heritage_detail_enriched.csv")
except Exception:
  df = pd.DataFrame()

st.markdown(
    "<h1 style='font-size:28px; margin-top:0px; margin-bottom:5px;'>🏛 공공 환경"
    " 데이터 기반 영천 지역 실시간 환경 현황</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='margin-bottom:10px; color:#4b5563;'>영천 지역 문화재 보존"
    " 관리를 위한 실시간 기상 관측 데이터(AWS) 및 대기 오염 현황 모니터링"
    " 페이지입니다.</p>",
    unsafe_allow_html=True,
)

# 자동 새로고침 상태 안내 표시
st.info(
    f"🔄 **실시간 자동 동기화 중** (마지막 화면 동기화: {current_kst}) — 관측소"
    " 수집 전송 지연에 따라 실측 시각과 차이가 발생할 수 있습니다."
)

st.divider()

# 카드 스타일 정의
title_style = (
    "font-size:16px; font-weight:700; margin-bottom:6px; color:#1f2937;"
)
label_style = "font-size:12px; color:#6b7280; margin-bottom:2px;"

# --------------------------------------------
# 1행 : [좌측] 실시간 날씨 지도 & [우측] 대기오염 및 문화재 현황
# --------------------------------------------
map_col, right_col = st.columns([1.8, 1.0])

with map_col:
  m = folium.Map(location=[36.0650, 128.8740], zoom_start=10.5)

  # 상자 크기 (고정값으로 앵커 계산에 사용)
  BOX_W = 170
  BOX_H = 100

  # 각 상자 배치
  for name, info in STATION_MAP.items():
    w_data = weather_results[name]

    obs_time_fmt = w_data["obs_time"]
    if len(obs_time_fmt) == 12:
      obs_time_fmt = (
          f"{obs_time_fmt[:4]}-{obs_time_fmt[4:6]}-{obs_time_fmt[6:8]}"
          f" {obs_time_fmt[8:10]}:{obs_time_fmt[10:12]}"
      )

    # 위치 미세 조정
    if name == "신녕":
      anchor_val = (-120, BOX_H + 100)
    elif name == "화북":
      anchor_val = (BOX_W - 210, -40)
    elif name == "영천(종합)":
      anchor_val = (BOX_W - 140, BOX_H)
    else:  # 청통
      anchor_val = (0, 0)

    label_html = f"""
        <div style="
            background-color: white; 
            border: 2px solid #1e3a8a; 
            border-radius: 8px; 
            padding: 6px 10px; 
            font-size: 11px; 
            font-weight: bold; 
            color: #1f2937; 
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            width: {BOX_W}px;
            text-align: left;
        ">
            <div style="color: #1d4ed8; border-bottom: 1px solid #e5e7eb; padding-bottom: 2px; margin-bottom: 4px;">📍 {name}</div>
            <div style="display: flex; justify-content: space-between; margin-bottom: 2px;">
                <span>🌡 {w_data['temp']}°C</span>
                <span>💧 {w_data['humidity']}%</span>
            </div>
            <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                <span>🌧 {w_data['rainfall']}mm</span>
                <span>💨 {w_data['wind_speed']}m/s ({w_data['wind_dir']})</span>
            </div>
            <div style="font-size: 9px; color: #9ca3af; border-top: 1px solid #e5e7eb; padding-top: 2px;">⏱ {obs_time_fmt}</div>
        </div>
        """

    folium.Marker(
        location=[info["lat"], info["lon"]],
        icon=folium.DivIcon(
            html=label_html, icon_size=(BOX_W, BOX_H), icon_anchor=anchor_val
        ),
    ).add_to(m)

  st_folium(m, width="100%", height=460, key="weather_map")

with right_col:
  # 1. 대기현황 카드
  st.markdown(
      f"""
<div style="background-color:#f8f9fa; padding:12px 14px; border-radius:14px; border:1px solid #e5e7eb; box-shadow:0 4px 12px rgba(0,0,0,0.04); margin-bottom:10px;">
    <div style="{title_style}">🌫 대기오염 현황 (영천 측정소)</div>
    <hr style="margin: 4px 0;">
    <div style="display:grid; grid-template-columns:1fr 1fr; gap:6px; margin-top:4px;">
        <div>
            <div style="{label_style}">PM10 (미세먼지)</div><div style="font-size:14px; font-weight:700; color:#111827; margin-bottom:3px;">{air_data['pm10']} ㎍/㎥</div>
            <div style="{label_style}">O₃ (오존)</div><div style="font-size:14px; font-weight:700; color:#111827; margin-bottom:3px;">{air_data['o3']} ppm</div>
            <div style="{label_style}">CO (일산화탄소)</div><div style="font-size:14px; font-weight:700; color:#111827;">{air_data['co']} ppm</div>
        </div>
        <div>
            <div style="{label_style}">PM2.5 (초미세먼지)</div><div style="font-size:14px; font-weight:700; color:#111827; margin-bottom:3px;">{air_data['pm25']} ㎍/㎥</div>
            <div style="{label_style}">NO₂ (이산화질소)</div><div style="font-size:14px; font-weight:700; color:#111827; margin-bottom:3px;">{air_data['no2']} ppm</div>
            <div style="{label_style}">SO₂ (아황산가스)</div><div style="font-size:14px; font-weight:700; color:#111827;">{air_data['so2']} ppm</div>
        </div>
    </div>
    <div style="font-size:10px; color:#9ca3af; margin-top:6px;">⏱ 측정 시각 : {air_data['data_time']}</div>
</div>
    """,
      unsafe_allow_html=True,
  )

  # 2. 대기현황 밑에 위치한 문화재 수 카드
  st.markdown(
      f"""
<div style="background-color:#f8f9fa; padding:12px 14px; border-radius:14px; border:1px solid #e5e7eb; box-shadow:0 4px 12px rgba(0,0,0,0.04);">
    <div style="{title_style}">🏛 문화재 보존 관리 현황</div>
    <hr style="margin: 4px 0;">
    <div style="display:flex; justify-content:space-between; align-items:center; margin-top:6px;">
        <div style="{label_style}">실시간 모니터링 대상 문화재 총 수</div>
        <div style="font-size:18px; font-weight:700; color:#1f2937;">{len(df)}개소</div>
    </div>
</div>
    """,
      unsafe_allow_html=True,
  )

st.divider()
st.caption("선화여고 - 영천 헤리티지 AI 탐구단")
