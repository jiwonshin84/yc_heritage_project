# ==========================================================
# 라이브러리
# ==========================================================
import streamlit as st
import pandas as pd
import requests
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo


# ============================================
# API KEY
# ============================================
SERVICE_KEY = "feb2bfabd299d5d05e89c7aec49ba7e706112603e76549a92e868bd86ec60323"


# ============================================
# 1. 기상청 지상(AWS) 10분 단위 실시간 관측 데이터
# ============================================
AWS_URL = "http://apis.data.go.kr/1360000/SfcMtsInfoService/getAws10Min"

# 영천 주요 관측소 목록 (지점 ID)
STATION_MAP = {
    "영천(종합)": "281",
    "신령": "853",
    "청통": "854",
    "화북": "855"
}

def get_aws_latest_tm():
    """가장 최근 10분 단위 관측 시각(YYYYMMDDHHMM) 생성"""
    now = datetime.now(ZoneInfo("Asia/Seoul"))
    # 최신 관측값 수집 오차 감안하여 약 2분 전 기준 10분 단위 절삭
    target_time = now - timedelta(minutes=2)
    minute = (target_time.minute // 10) * 10
    target = target_time.replace(minute=minute, second=0, microsecond=0)
    
    api_tm = target.strftime("%Y%m%d%H%M")
    display_tm = target.strftime("%Y-%m-%d %H:%M")
    return api_tm, display_tm

def degree_to_direction(deg):
    """풍향(도) -> 16방위 변환"""
    try:
        deg = float(deg)
        if deg < 0:
            return "-"
        dirs = [
            "북", "북북동", "북동", "동북동", 
            "동", "남동", "남", "남남서", 
            "남서", "서남서", "서", "서북서", 
            "북서", "북북서", "북"
        ]
        idx = int((deg + 11.25) / 22.5) % 16
        return dirs[idx]
    except:
        return "-"

def get_aws_weather_data(stn_id, tm_str):
    """특정 지점의 AWS 10분 단위 데이터 수집"""
    aws_data = {
        "temp": "-",
        "humidity": "-",
        "rainfall": "-",
        "wind_speed": "-",
        "wind_dir": "-",
        "time": "-"
    }
    
    try:
        params = {
            "serviceKey": SERVICE_KEY,
            "pageNo": "1",
            "numOfRows": "10",
            "dataType": "JSON",
            "stnId": str(stn_id),
            "tm": tm_str
        }

        response = requests.get(AWS_URL, params=params, timeout=10)
        
        if response.status_code == 200:
            res_json = response.json()
            items = res_json.get("response", {}).get("body", {}).get("items", {}).get("item", [])
            
            if items:
                item = items[0] if isinstance(items, list) else items
                aws_data["temp"] = item.get("ta", "-")         # 기온 (°C)
                aws_data["humidity"] = item.get("hm", "-")     # 습도 (%)
                aws_data["rainfall"] = item.get("rn1hr", item.get("rn10m", "-")) # 강수량 (mm)
                aws_data["wind_speed"] = item.get("ws", "-")   # 풍속 (m/s)
                aws_data["wind_dir"] = degree_to_direction(item.get("wd", "-")) # 풍향
                aws_data["time"] = item.get("tm", "-")
    except Exception as e:
        print(f"AWS 데이터 조회 실패 (지점: {stn_id}): {e}")
        
    return aws_data


# 관측 시각 산출
api_tm, display_tm = get_aws_latest_tm()


# ============================================
# 2. 대기오염 최신 데이터 (기존 유지)
# ============================================
AIR_URL = (
    "https://apis.data.go.kr/"
    "B552584/ArpltnInforInqireSvc/"
    "getCtprvnRltmMesureDnsty"
)

pm10 = "-"
pm25 = "-"
o3 = "-"
no2 = "-"
co = "-"
so2 = "-"
data_time = "-"

try:
    air_params = {
        "serviceKey": SERVICE_KEY,
        "returnType": "json",
        "numOfRows": "100",
        "pageNo": "1",
        "sidoName": "경북",
        "ver": "1.0"
    }

    air_response = requests.get(AIR_URL, params=air_params, timeout=30)
    air_data = air_response.json()

    items = air_data.get("response", {}).get("body", {}).get("items", [])

    target = None
    for item in items:
        if "영천" in item.get("stationName", ""):
            target = item
            break

    if target:
        data_time = target.get("dataTime", "-")
        pm10 = target.get("pm10Value", "-")
        pm25 = target.get("pm25Value", "-")
        o3 = target.get("o3Value", "-")
        no2 = target.get("no2Value", "-")
        co = target.get("coValue", "-")
        so2 = target.get("so2Value", "-")

except Exception as e:
    print("대기오염 데이터 조회 실패:", e)


# ==========================================================
# 페이지 설정
# ==========================================================
st.set_page_config(
    page_title="공공 환경 데이터 기반 영천 지역 문화재 훼손 위험 예측",
    page_icon="🏛",
    layout="wide"
)

# ==========================================================
# 데이터 불러오기
# ==========================================================
try:
    df = pd.read_csv("data/processed/yc_heritage_detail_enriched.csv")
except:
    df = pd.DataFrame()


# ==========================================================
# 제목
# ==========================================================
st.markdown("""
<h1 style='font-size:30px;'>
🏛 공공 환경 데이터 기반 영천 지역 문화재 훼손 위험 예측
</h1>
""", unsafe_allow_html=True)
st.markdown("""
영천 지역 문화재와 공공 환경데이터를 분석하여 문화재 훼손 위험을 사전에 예측하는 데이터 분석 프로젝트 입니다.
""")

st.divider()


# ============================================
# 상단 환경 대시보드
# ============================================
st.markdown("""
<h3 style="font-size:25px; margin-bottom:10px;">
🌿 영천시 실시간 환경 데이터 및 문화재 현황
</h3>
""", unsafe_allow_html=True)

# 메인 영역 (열 레이아웃)
left, center, right = st.columns([1.4, 2.0, 1.0])


# ============================================
# 공통 카드 스타일
# ============================================
card_style = """
background-color:#f8f9fa;
padding:22px;
border-radius:20px;
border:1px solid #e5e7eb;
box-shadow:0 4px 12px rgba(0,0,0,0.05);
height:380px;
"""

title_style = """
font-size:22px;
font-weight:700;
margin-bottom:10px;
color:#1f2937;
"""

label_style = """
font-size:14px;
color:#6b7280;
margin-bottom:4px;
"""

value_style = """
font-size:20px;
font-weight:700;
color:#111827;
margin-bottom:14px;
"""

time_style = """
font-size:13px;
color:#9ca3af;
margin-top:12px;
position:absolute;
bottom:20px;
"""


# ============================================
# 1열 : 기상 환경 (AWS 4개 지점 선택 기능 추가)
# ============================================
with left:
    # 관측소 선택 드롭다운
    selected_stn_name = st.selectbox(
        "📍 AWS 기상 관측소 선택",
        options=list(STATION_MAP.keys()),
        index=0,
        key="aws_station_select"
    )
    
    selected_stn_id = STATION_MAP[selected_stn_name]
    weather = get_aws_weather_data(selected_stn_id, api_tm)

    st.markdown(
        f"""
<div style="{card_style}; position:relative;">

<div style="{title_style}">
🌦 실시간 기상 환경 ({selected_stn_name})
</div>

<hr>

<div style="
display:grid;
grid-template-columns:1fr 1fr;
gap:12px;
margin-top:10px;
">

<div>
<div style="{label_style}">🌡 기온</div>
<div style="{value_style}">{weather['temp']} °C</div>
</div>

<div>
<div style="{label_style}">💧 습도</div>
<div style="{value_style}">{weather['humidity']} %</div>
</div>

<div>
<div style="{label_style}">🌧 강수량 (1시간)</div>
<div style="{value_style}">{weather['rainfall']} mm</div>
</div>

<div>
<div style="{label_style}">💨 풍속 / 풍향</div>
<div style="{value_style}">{weather['wind_speed']} m/s ({weather['wind_dir']})</div>
</div>

</div>

<div style="{time_style}">
⏱ 관측 시각 : {display_tm}
</div>

</div>
        """,
        unsafe_allow_html=True
    )


# ============================================
# 2열 : 대기오염 현황 (기존 그대로)
# ============================================
with center:
    # 1열의 selectbox 높이 상쇄용 더미 공간
    st.markdown("<div style='height: 42px;'></div>", unsafe_allow_html=True)
    
    st.markdown(
        f"""
<div style="{card_style}; position:relative;">

<div style="{title_style}">
🌫 대기오염 현황 (영천 측정소)
</div>

<hr>

<div style="
display:grid;
grid-template-columns:1fr 1fr 1fr;
gap:16px;
margin-top:10px;
">

<div>
<div style="{label_style}">PM10</div>
<div style="{value_style}">{pm10} ㎍/㎥</div>

<div style="{label_style}">O₃</div>
<div style="{value_style}">{o3} ppm</div>
</div>

<div>
<div style="{label_style}">PM2.5</div>
<div style="{value_style}">{pm25} ㎍/㎥</div>

<div style="{label_style}">NO₂</div>
<div style="{value_style}">{no2} ppm</div>
</div>

<div>
<div style="{label_style}">CO</div>
<div style="{value_style}">{co} ppm</div>

<div style="{label_style}">SO₂</div>
<div style="{value_style}">{so2} ppm</div>
</div>

</div>

<div style="{time_style}">
⏱ 측정 시각 : {data_time}
</div>

</div>
        """,
        unsafe_allow_html=True
    )


# ============================================
# 3열 : 문화재 현황 (기존 그대로)
# ============================================
with right:
    st.markdown("<div style='height: 42px;'></div>", unsafe_allow_html=True)
    
    st.markdown(
        f"""
<div style="{card_style}; position:relative;">

<div style="{title_style}">
🏛 문화재 현황
</div>

<hr>

<div style="margin-top:20px;">

<div style="{label_style}">
분석 문화재 수
</div>

<div style="{value_style}">
{len(df)}개
</div>

</div>

</div>
        """,
        unsafe_allow_html=True
    )

st.divider()

# ==========================================================
# 하단 안내
# ==========================================================
st.caption(
    "제6회 학생 SW·AI 인재양성 프로젝트 | 선화여고 - 영천 헤리티지 AI 탐구단"
)
