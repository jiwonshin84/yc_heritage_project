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

st.set_page_config(
    page_title="공공 환경 데이터 기반 영천 지역 실시간 환경 현황",
    page_icon="🏠",
    layout="wide",
)

st.markdown(
    """
    <style>
        .block-container {
            padding-top: 1rem !important;
            padding-bottom: 2rem !important;
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
KMA_AUTH_KEY = st.secrets.get("KMA_AUTH_KEY", "")

# 한국환경공단 API 인증키
AIR_SERVICE_KEY = st.secrets.get("AIR_SERVICE_KEY", "")

# 기상청 AWS 매분자료 조회 API URL
AWS_MIN_URL = "https://apihub.kma.go.kr/api/typ01/cgi-bin/url/nph-aws2_min"

# 한국환경공단 시도별 실시간 대기오염 측정 정보 API URL
AIR_URL = "https://apis.data.go.kr/B552584/ArpltnInforInqireSvc/getCtprvnRltmMesureDnsty"

# Streamlit Secrets 설정 확인
if not KMA_AUTH_KEY:
  st.error("❌ Streamlit Secrets에 KMA_AUTH_KEY가 없습니다. Streamlit Cloud의 Settings → Secrets에 인증키를 등록해주세요.")
  st.stop()

if not AIR_SERVICE_KEY:
  st.error("❌ Streamlit Secrets에 AIR_SERVICE_KEY가 없습니다. Streamlit Cloud의 Settings → Secrets에 인증키를 등록해주세요.")
  st.stop()

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


def _normalize_risk_label(value):
  """예측 결과의 다양한 등급 표현을 안전/주의/위험으로 통일"""
  if pd.isna(value):
    return None

  # 숫자형 0/1/2
  try:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
      ivalue = int(value)
      return {0: "안전", 1: "주의", 2: "위험"}.get(ivalue)
  except Exception:
    pass

  text_value = str(value).strip().lower()

  mapping = {
      "0": "안전",
      "1": "주의",
      "2": "위험",
      "안전": "안전",
      "safe": "안전",
      "normal": "안전",
      "주의": "주의",
      "caution": "주의",
      "warning": "주의",
      "위험": "위험",
      "danger": "위험",
      "risk": "위험",
  }

  if text_value in mapping:
    return mapping[text_value]

  # 문자열 안에 등급명이 포함된 경우
  if "안전" in text_value or "safe" in text_value:
    return "안전"
  if "주의" in text_value or "caution" in text_value or "warning" in text_value:
    return "주의"
  if "위험" in text_value or "danger" in text_value:
    return "위험"

  return None


def _extract_prediction_counts(pred_df):
  """
  예측 DataFrame에서 안전/주의/위험 개수를 계산.
  여러 날짜가 들어 있으면 가장 최근 날짜만 사용.
  """
  if pred_df is None or not isinstance(pred_df, pd.DataFrame) or pred_df.empty:
    return None

  work = pred_df.copy()

  # 가장 최근 예측 시각/날짜만 선택
  date_candidates = [
      "prediction_datetime",
      "prediction_time",
      "prediction_date",
      "target_date",
      "date",
      "예측시각",
      "예측일시",
      "예측일",
      "기준일",
  ]

  latest_label = None

  for col in date_candidates:
    if col in work.columns:
      parsed = pd.to_datetime(work[col], errors="coerce")
      if parsed.notna().any():
        latest_dt = parsed.max()
        work = work.loc[parsed == latest_dt].copy()
        latest_label = latest_dt.strftime("%Y-%m-%d")
        break

  # 예측 등급 컬럼 탐색
  class_candidates = [
      "risk_label",
      "risk_level",
      "prediction_label",
      "predicted_label",
      "prediction",
      "pred_class",
      "class",
      "예측등급",
      "위험등급",
      "환경취약도등급",
      "등급",
  ]

  class_col = next(
      (col for col in class_candidates if col in work.columns),
      None,
  )

  if class_col is None:
    return None

  labels = work[class_col].apply(_normalize_risk_label)

  counts = {
      "안전": int((labels == "안전").sum()),
      "주의": int((labels == "주의").sum()),
      "위험": int((labels == "위험").sum()),
  }

  # 등급을 하나도 읽지 못한 경우
  if sum(counts.values()) == 0:
    return None

  return {
      "counts": counts,
      "latest_label": latest_label,
      "total": sum(counts.values()),
  }


def get_latest_prediction_status():
  """
  마지막 예측 결과를 찾는다.

  우선순위
  1) 현재 Streamlit 세션의 예측 결과
  2) 저장된 최신 예측 CSV

  예측 페이지와 실시간 대시보드가 같은 세션이면 즉시 반영되고,
  CSV를 저장해 둔 경우 앱 재시작 후에도 복원 가능하다.
  """

  # 1. Session State 우선
  session_candidates = [
      "prediction_result",
      "prediction_df",
      "latest_prediction",
      "latest_prediction_df",
      "heritage_prediction_result",
      "prediction_results",
  ]

  for key in session_candidates:
    if key in st.session_state:
      result = _extract_prediction_counts(st.session_state.get(key))
      if result is not None:
        result["source"] = "최근 예측 세션"
        return result

  # 2. 로컬/GitHub 저장소에 존재할 수 있는 최신 예측 파일
  file_candidates = [
      "data/processed/latest_prediction.csv",
      "data/processed/latest_prediction_result.csv",
      "data/processed/heritage_prediction_latest.csv",
      "data/processed/latest_heritage_prediction.csv",
      "data/processed/prediction_result.csv",
      "data/processed/prediction_results.csv",
      "data/processed/예측결과.csv",
      "latest_prediction.csv",
  ]

  for file_path in file_candidates:
    if os.path.exists(file_path):
      try:
        pred_df = pd.read_csv(file_path)
        result = _extract_prediction_counts(pred_df)
        if result is not None:
          result["source"] = file_path
          return result
      except Exception:
        pass

  return {
      "counts": {
          "안전": None,
          "주의": None,
          "위험": None,
      },
      "latest_label": None,
      "total": 0,
      "source": None,
  }


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

# 마지막 문화재 예측 결과
latest_prediction_status = get_latest_prediction_status()
latest_prediction_counts = latest_prediction_status["counts"]

safe_count = latest_prediction_counts["안전"]
caution_count = latest_prediction_counts["주의"]
danger_count = latest_prediction_counts["위험"]

prediction_date_text = (
    latest_prediction_status["latest_label"]
    if latest_prediction_status["latest_label"]
    else "최근 예측 결과 없음"
)

safe_text = f"{safe_count}개" if safe_count is not None else "-"
caution_text = f"{caution_count}개" if caution_count is not None else "-"
danger_text = f"{danger_count}개" if danger_count is not None else "-"

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
    f"🔄 **실시간 자동 동기화 중** (마지막 화면 동기화: {current_kst}) "
    "— 관측소 수집 전송 지연에 따라 실측 시각과 차이가 발생할 수 있습니다."
)
# 카드 스타일 정의
title_style = (
    "font-size:20px; font-weight:700; margin-bottom:6px; color:#1f2937;"
)
label_style = "font-size:18px; color:#6b7280; margin-bottom:2px;"

# --------------------------------------------
# 1행 : 실시간 날씨 지도 / 대기오염 및 문화재 현황
# --------------------------------------------
map_col, right_col = st.columns([1.8, 1.0])

with map_col:
  m = folium.Map(location=[36.0650, 128.8740], zoom_start=10.5)

  
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
      anchor_val = (-40, BOX_H + 40)
    elif name == "화북":
      anchor_val = (BOX_W - 210, -40)
    elif name == "영천(종합)":
      anchor_val = (BOX_W - 140, BOX_H)
    else:  # 청통
      anchor_val = (120, 10)

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
            <div style="font-size: 10px; color: #9ca3af; border-top: 1px solid #e5e7eb; padding-top: 2px;">⏱ {obs_time_fmt}</div>
        </div>
        """

    folium.Marker(
        location=[info["lat"], info["lon"]],
        icon=folium.DivIcon(
            html=label_html, icon_size=(BOX_W, BOX_H), icon_anchor=anchor_val
        ),
    ).add_to(m)

  st_folium(m, width="100%", height=600, key="weather_map")

with right_col:
  # 1. 대기현황 카드
  st.markdown(
      f"""
<div style="background-color:#f8f9fa; padding:12px 14px; border-radius:14px; border:1px solid #e5e7eb; box-shadow:0 4px 12px rgba(0,0,0,0.04); margin-bottom:10px;">
    <div style="{title_style}">🌫 대기오염 현황 (영천 측정소)</div>
    <hr style="margin: 4px 0;">
    <div style="display:grid; grid-template-columns:1fr 1fr; gap:6px; margin-top:4px;">
        <div>
            <div style="{label_style}">PM10 (미세먼지)</div><div style="font-size:18px; font-weight:700; color:#111827; margin-bottom:3px;">{air_data['pm10']} ㎍/㎥</div>
            <div style="{label_style}">O₃ (오존)</div><div style="font-size:18px; font-weight:700; color:#111827; margin-bottom:3px;">{air_data['o3']} ppm</div>
            <div style="{label_style}">CO (일산화탄소)</div><div style="font-size:18px; font-weight:700; color:#111827;">{air_data['co']} ppm</div>
        </div>
        <div>
            <div style="{label_style}">PM2.5 (초미세먼지)</div><div style="font-size:18px; font-weight:700; color:#111827; margin-bottom:3px;">{air_data['pm25']} ㎍/㎥</div>
            <div style="{label_style}">NO₂ (이산화질소)</div><div style="font-size:18px; font-weight:700; color:#111827; margin-bottom:3px;">{air_data['no2']} ppm</div>
            <div style="{label_style}">SO₂ (아황산가스)</div><div style="font-size:18px; font-weight:700; color:#111827;">{air_data['so2']} ppm</div>
        </div>
    </div>
    <div style="font-size:16px; color:#9ca3af; margin-top:6px;">⏱ 측정 시각 : {air_data['data_time']}</div>
</div>
    """,
      unsafe_allow_html=True,
  )

  # 2. 문화재 보존 관리 + 최근 예측 현황 카드
  if latest_prediction_status["source"] is not None:
    prediction_html = f"""
    <div style="margin-top:10px; padding-top:9px; border-top:1px solid #e5e7eb;">
      <div style="display:flex; justify-content:space-between; align-items:center; gap:8px; margin-bottom:7px;">
        <div style="font-size:16px; font-weight:700; color:#374151;">🤖 마지막 예측 결과</div>
        <div style="font-size:12px; color:#9ca3af; white-space:nowrap;">{prediction_date_text}</div>
      </div>
      <div style="display:grid; grid-template-columns:repeat(3, minmax(0, 1fr)); gap:7px;">
        <div style="background:#ecfdf5; border:1px solid #a7f3d0; border-radius:10px; padding:8px 5px; text-align:center;">
          <div style="font-size:13px; font-weight:700; color:#047857;">✅ 안전</div>
          <div style="font-size:20px; font-weight:800; color:#065f46; margin-top:2px;">{safe_text}</div>
        </div>
        <div style="background:#fffbeb; border:1px solid #fde68a; border-radius:10px; padding:8px 5px; text-align:center;">
          <div style="font-size:13px; font-weight:700; color:#b45309;">⚠️ 주의</div>
          <div style="font-size:20px; font-weight:800; color:#92400e; margin-top:2px;">{caution_text}</div>
        </div>
        <div style="background:#fef2f2; border:1px solid #fecaca; border-radius:10px; padding:8px 5px; text-align:center;">
          <div style="font-size:13px; font-weight:700; color:#b91c1c;">🚨 위험</div>
          <div style="font-size:20px; font-weight:800; color:#991b1b; margin-top:2px;">{danger_text}</div>
        </div>
      </div>
      <div style="font-size:11px; color:#9ca3af; margin-top:6px; text-align:right;">
        최근 실행된 문화재 환경 취약도 분류 결과
      </div>
    </div>
    """
  else:
    # latest_prediction.csv가 없고 현재 세션에도 예측 결과가 없는 경우
    prediction_html = """
    <div style="margin-top:10px; padding-top:9px; border-top:1px solid #e5e7eb;">
      <div style="font-size:16px; font-weight:700; color:#374151; margin-bottom:6px;">
        🤖 마지막 예측 결과
      </div>
      <div style="
        background:#f9fafb;
        border:1px dashed #d1d5db;
        border-radius:10px;
        padding:12px 10px;
        text-align:center;
        color:#6b7280;
        font-size:13px;
        line-height:1.55;
      ">
        아직 저장된 예측 결과가 없습니다.<br>
        <span style="font-size:12px; color:#9ca3af;">
          예측 페이지에서 환경 취약도 예측을 실행하면<br>
          안전·주의·위험 현황이 여기에 자동 표시됩니다.
        </span>
      </div>
    </div>
    """

  heritage_card_html = f"""
  <div style="
    background-color:#f8f9fa;
    padding:12px 14px;
    border-radius:14px;
    border:1px solid #e5e7eb;
    box-shadow:0 4px 12px rgba(0,0,0,0.04);
  ">
    <div style="{title_style}">🏛 문화재 보존 관리 현황</div>
    <hr style="margin:4px 0;">
    <div style="display:flex; justify-content:space-between; align-items:center; gap:10px; margin-top:6px;">
      <div style="{label_style}">실시간 모니터링 대상 문화재</div>
      <div style="font-size:18px; font-weight:700; color:#1f2937; white-space:nowrap;">{len(df)}개</div>
    </div>
    {prediction_html}
  </div>
  """

  st.html(heritage_card_html)


st.caption("선화여고 - 영천 헤리티지 AI 탐구단")
