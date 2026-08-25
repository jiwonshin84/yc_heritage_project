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


def get_latest_base_time():

    now = datetime.now(ZoneInfo("Asia/Seoul"))

    target = now.replace(minute=0, second=0, microsecond=0)

    # 초단기실황은 정시+40분 이후 공개
    if now.minute < 40:
        target -= timedelta(hours=1)

    api_date = target.strftime("%Y%m%d")
    api_time = target.strftime("%H00")

    display_date = target.strftime("%Y-%m-%d")

    return api_date, api_time, display_date


# ============================================
# 1. 기상청 초단기실황 최신 자료
# ============================================

ULTRA_URL = (
    "https://apis.data.go.kr/"
    "1360000/VilageFcstInfoService_2.0/"
    "getUltraSrtNcst"
)

# 영천시 중심 격자
NX = "92"
NY = "106"

# 최근 발표 시각
api_date, api_time, display_date = get_latest_base_time()

# 기본값
tm = f"{display_date} {api_time[:2]}:00"

temp = "-"
humidity = "-"

rainfall = "-"
wind_speed = "-"

rain_type = "-"
wind_dir = "-"

# 풍향 변환 함수
def degree_to_direction(deg):

    try:
        deg = float(deg)

        dirs = [
            "북", "북동", "동", "남동",
            "남", "남서", "서", "북서"
        ]

        return dirs[
            round(deg / 45) % 8
        ]

    except:
        return "-"


try:

    ultra_params = {
        "serviceKey": SERVICE_KEY,
        "pageNo": "1",
        "numOfRows": "1000",
        "dataType": "JSON",

        "base_date": api_date,
        "base_time": api_time,

        "nx": NX,
        "ny": NY
    }

    response = requests.get(
        ULTRA_URL,
        params=ultra_params,
        timeout=30
    )

    print("초단기실황 응답코드:", response.status_code)

    data = response.json()

    items = (
        data["response"]["body"]
        ["items"]["item"]
    )

    for item in items:

        category = item["category"]
        value = item["obsrValue"]

        if category == "T1H":
            temp = value

        elif category == "REH":
            humidity = value

        elif category == "RN1":
            rainfall = value

        elif category == "WSD":
            wind_speed = value

        elif category == "VEC":
            wind_dir = degree_to_direction(value)

        elif category == "PTY":

            rain_map = {
                "0": "없음",
                "1": "비",
                "2": "비/눈",
                "3": "눈",
                "5": "빗방울",
                "6": "빗방울눈날림",
                "7": "눈날림"
            }

            rain_type = rain_map.get(
                str(value),
                str(value)
            )

    print()
    print("===== 초단기실황 =====")

    print("관측시각:", tm)
    print("기온:", temp)
    print("습도:", humidity)
    print("강수량:", rainfall)
    print("풍속:", wind_speed)
    print("풍향:", wind_dir)
    print("강수형태:", rain_type)

except Exception as e:

    print("초단기실황 조회 실패")
    print(e)

# ============================================
# 2. 대기오염 최신 데이터
# ============================================

AIR_URL = (
    "https://apis.data.go.kr/"
    "B552584/ArpltnInforInqireSvc/"
    "getCtprvnRltmMesureDnsty"
)

# 기본값
pm10 = "-"
pm25 = "-"

o3 = "-"
no2 = "-"

co = "-"
so2 = "-"

data_time = "-"

# ============================================
# 대기오염 API 요청
# ============================================

try:

    air_params = {
        "serviceKey": SERVICE_KEY,
        "returnType": "json",

        "numOfRows": "100",
        "pageNo": "1",

        # 경북
        "sidoName": "경북",

        "ver": "1.0"
    }

    air_response = requests.get(
        AIR_URL,
        params=air_params,
        timeout=30
    )

    print("대기오염 응답코드:", air_response.status_code)

    air_data = air_response.json()

    print(air_data)

    items = air_data["response"]["body"]["items"]

    # 영천 측정소 찾기
    target = None

    for item in items:

        if "영천" in item["stationName"]:
            target = item
            break

    if target:

        data_time = target["dataTime"]

        pm10 = target["pm10Value"]
        pm25 = target["pm25Value"]

        o3 = target["o3Value"]
        no2 = target["no2Value"]

        co = target["coValue"]
        so2 = target["so2Value"]

        print()
        print("===== 최신 대기오염 데이터 =====")

        print("측정시각:", data_time)

        print("PM10:", pm10)
        print("PM2.5:", pm25)

        print("O3:", o3)
        print("NO2:", no2)

        print("CO:", co)
        print("SO2:", so2)

except Exception as e:

    print("대기오염 데이터 조회 실패")
    print(e)


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
df = pd.read_csv(
    "data/processed/yc_heritage_detail_enriched.csv"
)

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
<h3 style="
    font-size:25px;
    margin-bottom:10px;
">
🌿 영천시 실시간 환경 데이터 및 문화재 현황
</h3>
""", unsafe_allow_html=True)

# 메인 영역
left, center, right = st.columns([1.4, 2.0, 1.0])

# ============================================
# 공통 스타일
# ============================================

card_style = """
background-color:#f8f9fa;
padding:22px;
border-radius:20px;
border:1px solid #e5e7eb;
box-shadow:0 4px 12px rgba(0,0,0,0.05);
height:350px;
"""

title_style = """
font-size:24px;
font-weight:700;
margin-bottom:14px;
color:#1f2937;
"""

label_style = """
font-size:14px;
color:#6b7280;
margin-bottom:4px;
"""

value_style = """
font-size:22px;
font-weight:700;
color:#111827;
margin-bottom:18px;
"""

time_style = """
font-size:13px;
color:#9ca3af;
margin-top:12px;
position:absolute;
bottom:20px;
"""

# ============================================
# 1열 : 기상 환경
# ============================================

with left:

    st.markdown(
        f"""
<div style="{card_style}; position:relative;">

<div style="{title_style}">
🌦 기상 환경
</div>

<hr>

<div style="
display:grid;
grid-template-columns:1fr 1fr;
gap:16px;
margin-top:20px;
">

<div>
<div style="{label_style}">🌡 기온</div>
<div style="{value_style}">{temp} °C</div>
</div>

<div>
<div style="{label_style}">💧 습도</div>
<div style="{value_style}">{humidity} %</div>
</div>

<div>
<div style="{label_style}">🌧 강수량</div>
<div style="{value_style}">{rainfall} mm</div>
</div>

<div>
<div style="{label_style}">💨 풍속</div>
<div style="{value_style}">{wind_speed} m/s</div>
</div>

</div>

<div style="{time_style}">
⏱ 측정 시각 : {tm}
</div>

</div>
        """,
        unsafe_allow_html=True
    )

# ============================================
# 2열 : 대기오염 현황
# ============================================

with center:

    st.markdown(
        f"""
<div style="{card_style}; position:relative;">

<div style="{title_style}">
🌫 대기오염 현황
</div>

<hr>

<div style="
display:grid;
grid-template-columns:1fr 1fr 1fr;
gap:20px;
margin-top:20px;
">

<div>

<div style="{label_style}">PM10</div>
<div style="{value_style}">{pm10}</div>

<div style="{label_style}">O₃</div>
<div style="{value_style}">{o3}</div>

</div>

<div>

<div style="{label_style}">PM2.5</div>
<div style="{value_style}">{pm25}</div>

<div style="{label_style}">NO₂</div>
<div style="{value_style}">{no2}</div>

</div>

<div>

<div style="{label_style}">CO</div>
<div style="{value_style}">{co}</div>

<div style="{label_style}">SO₂</div>
<div style="{value_style}">{so2}</div>

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
# 3열 : 문화재 현황
# ============================================

with right:

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

<br>

<div style="{label_style}">

</div>

<div style="{value_style}">

</div>

</div>

<div>
- 
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
