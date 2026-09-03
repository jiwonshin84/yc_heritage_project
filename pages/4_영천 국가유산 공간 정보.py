import re
from datetime import datetime
import urllib.parse
from zoneinfo import ZoneInfo

import folium
import pandas as pd
import requests
import streamlit as st
from streamlit_folium import st_folium

# =================================================
# 페이지 설정
# =================================================
st.set_page_config(
    page_title="영천 국가유산 공간 정보",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 상단 기본 여백 및 사이드바 유지 스타일 설정
st.markdown(
    """
    <style>
        .block-container {
            padding-top: 1rem !important;
            padding-bottom: 2rem !important;
        }
        .stButton>button {
            text-align: left;
            border-radius: 8px;
            padding: 10px;
            margin-bottom: -5px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# =================================================
# API KEY 및 관측소 설정 (실시간 대시보드 연동)
# =================================================
KMA_AUTH_KEY = "XDdcOK8kT5C3XDivJN-Qtg"
AWS_MIN_URL = "https://apihub.kma.go.kr/api/typ01/cgi-bin/url/nph-aws2_min"

STATION_MAP = {
    "신녕": {"id": "853", "lat": 36.0150, "lon": 128.6100},
    "청통": {"id": "854", "lat": 36.0250, "lon": 128.7800},
    "화북": {"id": "855", "lat": 36.1700, "lon": 128.9300},
    "영천(종합)": {"id": "281", "lat": 35.9650, "lon": 128.9400},
}


# =================================================
# 헬퍼 함수 정의
# =================================================
def degree_to_direction(deg):
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
  if val is None or val == "" or str(val).startswith("-99") or str(val) == "-":
    return default
  return str(val)


@st.cache_data(ttl=60)
def get_aws_weather_data(stn_id):
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


# 날씨 데이터 일괄 수집
weather_results = {}
for name, info in STATION_MAP.items():
  weather_results[name] = get_aws_weather_data(info["id"])


# =================================================
# 데이터 로드 및 전처리 (캐싱 적용)
# =================================================
@st.cache_data
def load_data():
  df = pd.read_csv("data/processed/yc_heritage_detail_enriched.csv")
  df.columns = df.columns.str.strip()
  df = df.dropna(subset=["위도", "경도"])

  def simplify_era(text):
    if pd.isna(text):
      return "기타"
    text = str(text).strip()
    if "청동기" in text:
      return "청동기"
    elif any(x in text for x in ["통일신라", "신라시대 후기"]):
      return "통일신라"
    elif "신라" in text:
      return "신라"
    elif "고려" in text:
      if any(x in text for x in ["초기", "전기"]):
        return "고려초기"
      if any(x in text for x in ["말기", "후기"]):
        return "고려후기"
      return "고려"
    elif any(
        k in text
        for k in [
            "세종",
            "태조",
            "태종",
            "문종",
            "단종",
            "세조",
            "성종",
            "연산군",
            "중종",
            "인종",
        ]
    ):
      return "조선초기"
    elif any(
        k in text for k in ["숙종", "영조", "정조", "순조", "철종", "고종", "광해군"]
    ):
      return "조선후기"
    elif "조선" in text:
      if any(x in text for x in ["초기", "전기"]):
        return "조선초기"
      if any(x in text for x in ["말기", "후기"]):
        return "조선후기"
      return "조선"
    elif "대한제국" in text:
      return "대한제국"

    year_match = re.search(r"\d{4}", text)
    if year_match:
      yr = int(year_match.group())
      if yr < 700:
        return "신라"
      elif yr < 1400:
        return "고려"
      elif yr < 1600:
        return "조선초기"
      elif yr < 1910:
        return "조선후기"
    return "기타"

  df["시대그룹"] = df["시대"].apply(simplify_era)
  df["국가유산종목"] = df.get("국가유산종목", "미상").fillna("미상").astype(str)
  df["소재지상세"] = df.get("소재지상세", "-").fillna("-").astype(str)
  return df


df = load_data()

# =================================================
# 제목
# =================================================
st.title("🛰️ 영천 국가유산 공간 정보 및 실시간 환경 현황")

# =================================================
# 사이드바 필터 및 검색
# =================================================
st.sidebar.header("🔎 검색 및 필터")
search_query = st.sidebar.text_input("유산 명칭 검색", placeholder="명칭을 입력하세요")

era_order = [
    "청동기",
    "신라",
    "통일신라",
    "고려초기",
    "고려",
    "고려후기",
    "조선초기",
    "조선",
    "조선후기",
    "대한제국",
    "기타",
]
existing_eras = [e for e in era_order if e in df["시대그룹"].unique()]
selected_era = st.sidebar.selectbox("시대 선택", ["전체"] + existing_eras)

type_options = ["전체"] + sorted(df["국가유산종목"].unique().tolist())
selected_type = st.sidebar.selectbox("종목 선택", type_options)

# 데이터 필터링
filtered_df = df.copy()
if search_query:
  filtered_df = filtered_df[
      filtered_df["문화재명(국문)"].str.contains(search_query, na=False)
  ]
if selected_era != "전체":
  filtered_df = filtered_df[filtered_df["시대그룹"] == selected_era]
if selected_type != "전체":
  filtered_df = filtered_df[filtered_df["국가유산종목"] == selected_type]

st.sidebar.metric("검색 결과", f"{len(filtered_df)} 건")

if filtered_df.empty:
  st.warning("조건에 맞는 유산이 없습니다.")
  st.stop()

# =================================================
# 세션 상태 관리 (중앙 좌표 및 선택 마커)
# =================================================
if (
    "selected_heritage" not in st.session_state
    or st.session_state.selected_heritage
    not in filtered_df["문화재명(국문)"].values
):
  st.session_state.selected_heritage = filtered_df.iloc[0]["문화재명(국문)"]

# 현재 선택된 데이터 (지도 중앙으로 보낼 좌표 추출)
selected_row = filtered_df[
    filtered_df["문화재명(국문)"] == st.session_state.selected_heritage
].iloc[0]
center_lat = selected_row["위도"]
center_lon = selected_row["경도"]

# =================================================
# 레이아웃 구성
# =================================================
map_col, list_col = st.columns([3.3, 1.2])

with map_col:
  # 기본 OpenStreetMap 방식 지도 생성 (화남면 중심 좌표 적용 가능)
  m = folium.Map(location=[36.0650, 128.8740], zoom_start=10.5)

  # 1. 문화유산 마커 추가
  for _, row in filtered_df.iterrows():
    name = row["문화재명(국문)"]
    is_selected = (name == st.session_state.selected_heritage)

    img_url = str(row.get("이미지URL", "")).replace("http://", "https://")
    img_tag = (
        f'<img src="{img_url}" style="width:100%; height:180px;'
        ' object-fit:cover; border-radius:8px;">'
        if img_url and img_url.lower() != "nan"
        else '<div'
        ' style="width:100%;height:150px;background:#eee;border-radius:8px;display:flex;justify-content:center;align-items:center;">이미지'
        " 없음</div>"
    )

    popup_content = f"""
        <div style="width:260px; font-family:sans-serif;">
            <h4 style="margin:0 0 10px 0;">{name}</h4>
            {img_tag}
            <div style="font-size:12px; margin-top:10px;">
                <b>시대:</b> {row['시대그룹']} | <b>종목:</b> {row['국가유산종목']}<br>
                <b>주소:</b> {row['소재지상세']}
            </div>
        </div>
        """

    folium.Marker(
        location=[row["위도"], row["경도"]],
        popup=folium.Popup(popup_content, max_width=300),
        tooltip=name,
        icon=folium.Icon(
            color="red" if is_selected else "blue",
            icon="university" if is_selected else "info-sign",
            prefix="fa" if is_selected else "glyphicon",
        ),
    ).add_to(m)

  # 2. 실시간 대시보드 스타일의 날씨/환경 관측소 카드 오버레이 추가
  BOX_W = 170
  BOX_H = 100

  for name, info in STATION_MAP.items():
    w_data = weather_results[name]
    obs_time_fmt = w_data["obs_time"]
    if len(obs_time_fmt) == 12:
      obs_time_fmt = (
          f"{obs_time_fmt[:4]}-{obs_time_fmt[4:6]}-{obs_time_fmt[6:8]}"
          f" {obs_time_fmt[8:10]}:{obs_time_fmt[10:12]}"
      )

    # 대시보드와 동일한 위치 미세 조정 값 적용
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
            border-radius: 8px; -webkit-border-radius: 8px;
            padding: 6px 10px; 
            width: {BOX_W}px;
            text-align: left;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        ">
            <div style="color: #1d4ed8; font-size: 13px; font-weight: 800; border-bottom: 1px solid #e5e7eb; padding-bottom: 2px; margin-bottom: 4px;">
                📍 {name} (관측소)
            </div>
            <div style="display: flex; justify-content: space-between; margin-bottom: 2px; font-size: 11px; font-weight: 600; color: #374151;">
                <span>🌡 {w_data['temp']}°C</span>
                <span>💧 {w_data['humidity']}%</span>
            </div>
            <div style="display: flex; justify-content: space-between; margin-bottom: 4px; font-size: 11px; font-weight: 600; color: #374151;">
                <span>🌧 {w_data['rainfall']}mm</span>
                <span>💨 {w_data['wind_speed']}m/s ({w_data['wind_dir']})</span>
            </div>
            <div style="font-size: 9px; font-weight: 500; color: #6b7280; border-top: 1px solid #e5e7eb; padding-top: 2px;">
                ⏱ {obs_time_fmt}
            </div>
        </div>
        """

    folium.Marker(
        location=[info["lat"], info["lon"]],
        icon=folium.DivIcon(
            html=label_html, icon_size=(BOX_W, BOX_H), icon_anchor=anchor_val
        ),
    ).add_to(m)

  # 지도 출력
  st_folium(m, width="100%", height=720, key="gis_map")

with list_col:
  st.subheader("📋 유산 목록")
  st.caption("클릭 시 해당 위치로 지도가 이동합니다.")

  list_container = st.container(height=670)
  with list_container:
    for idx, row in filtered_df.iterrows():
      name = row["문화재명(국문)"]
      addr = row["소재지상세"]
      is_selected = (name == st.session_state.selected_heritage)

      btn_label = f"🚩 {name}" if is_selected else f"🏛️ {name}"

      if st.button(btn_label, key=f"list_btn_{idx}", use_container_width=True):
        st.session_state.selected_heritage = name
        st.rerun()

      st.caption(f"{addr}")

# 하단 정보 바 (통계)
st.divider()
c1, c2, c3 = st.columns(3)
with c1:
  st.info(f"📊 **주요 시대:** {filtered_df['시대그룹'].mode()[0]}")
with c2:
  st.info(f"📂 **종목 다양성:** {filtered_df['국가유산종목'].nunique()}종")
with c3:
  st.success(f"현재 위치: **{st.session_state.selected_heritage}**")
