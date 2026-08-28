import streamlit as st
import pandas as pd
import folium
import re
from streamlit_folium import st_folium
from folium.plugins import HeatMap, MarkerCluster

# =================================================
# 페이지 설정
# =================================================
st.set_page_config(
    page_title="영천 국가유산 공간 정보",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 커스텀 CSS (목록 가독성 및 버튼 스타일)
st.markdown("""
    <style>
    .stButton>button {
        text-align: left;
        border-radius: 8px;
        padding: 10px;
        margin-bottom: -5px;
    }
    /* 선택된 버튼 스타일 강조 */
    div[data-testid="column"]:nth-child(2) button[kind="secondary"]:focus {
        border-color: #ff4b4b !important;
        color: #ff4b4b !important;
    }
    </style>
    """, unsafe_allow_html=True)

# =================================================
# 데이터 로드 및 전처리 (캐싱 적용)
# =================================================
@st.cache_data
def load_data():
    df = pd.read_csv("data/processed/yc_heritage_detail_enriched.csv")
    df.columns = df.columns.str.strip()
    df = df.dropna(subset=["위도", "경도"])
    
    def simplify_era(text):
        if pd.isna(text): return "기타"
        text = str(text).strip()
        if "청동기" in text: return "청동기"
        elif any(x in text for x in ["통일신라", "신라시대 후기"]): return "통일신라"
        elif "신라" in text: return "신라"
        elif "고려" in text:
            if any(x in text for x in ["초기", "전기"]): return "고려초기"
            if any(x in text for x in ["말기", "후기"]): return "고려후기"
            return "고려"
        elif any(k in text for k in ["세종", "태조", "태종", "문종", "단종", "세조", "성종", "연산군", "중종", "인종"]):
            return "조선초기"
        elif any(k in text for k in ["숙종", "영조", "정조", "순조", "철종", "고종", "광해군"]):
            return "조선후기"
        elif "조선" in text:
            if any(x in text for x in ["초기", "전기"]): return "조선초기"
            if any(x in text for x in ["말기", "후기"]): return "조선후기"
            return "조선"
        elif "대한제국" in text: return "대한제국"
        
        year_match = re.search(r"\d{4}", text)
        if year_match:
            yr = int(year_match.group())
            if yr < 700: return "신라"
            elif yr < 1400: return "고려"
            elif yr < 1600: return "조선초기"
            elif yr < 1910: return "조선후기"
        return "기타"

    df["시대그룹"] = df["시대"].apply(simplify_era)
    df["국가유산종목"] = df.get("국가유산종목", "미상").fillna("미상").astype(str)
    df["소재지상세"] = df.get("소재지상세", "-").fillna("-").astype(str)
    return df

df = load_data()

# =================================================
# 제목
# =================================================
st.title("🛰️ 영천 국가유산 공간 정보")

# =================================================
# 사이드바 필터 및 검색
# =================================================
st.sidebar.header("🔎 검색 및 필터")
search_query = st.sidebar.text_input("유산 명칭 검색", placeholder="명칭을 입력하세요")

era_order = ["청동기", "신라", "통일신라", "고려초기", "고려", "고려후기", "조선초기", "조선", "조선후기", "대한제국", "기타"]
existing_eras = [e for e in era_order if e in df["시대그룹"].unique()]
selected_era = st.sidebar.selectbox("시대 선택", ["전체"] + existing_eras)

type_options = ["전체"] + sorted(df["국가유산종목"].unique().tolist())
selected_type = st.sidebar.selectbox("종목 선택", type_options)

# 데이터 필터링
filtered_df = df.copy()
if search_query:
    filtered_df = filtered_df[filtered_df["문화재명(국문)"].str.contains(search_query, na=False)]
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
if "selected_heritage" not in st.session_state or \
   st.session_state.selected_heritage not in filtered_df["문화재명(국문)"].values:
    st.session_state.selected_heritage = filtered_df.iloc[0]["문화재명(국문)"]

# 현재 선택된 데이터 (지도 중앙으로 보낼 좌표 추출)
selected_row = filtered_df[filtered_df["문화재명(국문)"] == st.session_state.selected_heritage].iloc[0]
center_lat = selected_row["위도"]
center_lon = selected_row["경도"]

# =================================================
# 레이아웃 구성
# =================================================
map_col, list_col = st.columns([3.3, 1.2])

with map_col:
    # 지도를 선택된 좌표 중심으로 생성
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=15, # 중앙으로 올 때 더 자세히 보이도록 조정
        tiles="CartoDB positron"
    )

    # 마커 클러스터 및 히트맵
    marker_cluster = MarkerCluster(name="국가유산 마커").add_to(m)
    heat_data = filtered_df[["위도", "경도"]].values.tolist()
    HeatMap(heat_data, radius=18, blur=13, name="밀집도").add_to(m)

    for _, row in filtered_df.iterrows():
        name = row["문화재명(국문)"]
        is_selected = (name == st.session_state.selected_heritage)
        
        # 팝업 HTML
        img_url = str(row.get("이미지URL", "")).replace("http://", "https://")
        img_tag = f'<img src="{img_url}" style="width:100%; height:180px; object-fit:cover; border-radius:8px;">' if img_url and img_url.lower() != "nan" else '<div style="width:100%;height:150px;background:#eee;border-radius:8px;display:flex;justify-content:center;align-items:center;">이미지 없음</div>'

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
        
        # 선택된 유산은 붉은색 마커, 나머지는 파란색
        folium.Marker(
            location=[row["위도"], row["경도"]],
            popup=folium.Popup(popup_content, max_width=300),
            tooltip=name,
            icon=folium.Icon(
                color="red" if is_selected else "blue", 
                icon="university", 
                prefix="fa"
            )
        ).add_to(marker_cluster)

    # 레이어 컨트롤 (히트맵 끄기 등 가능)
    folium.LayerControl().add_to(m)

    # 지도 출력 (세션 상태가 바뀔 때마다 다시 렌더링되어 중앙 이동)
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
            
            # 버튼 표시 (선택된 항목은 이모지 변경 및 강조)
            btn_label = f"🚩 {name}" if is_selected else f"🏛️ {name}"
            
            if st.button(btn_label, key=f"list_btn_{idx}", use_container_width=True):
                # 1. 세션 상태 업데이트
                st.session_state.selected_heritage = name
                # 2. 리프레시를 통해 지도 중심(location) 재설정
                st.rerun()
            
            st.caption(f"{addr}")
            #st.markdown("---")

# 하단 정보 바 (통계)
st.divider()
c1, c2, c3 = st.columns(3)
with c1: st.info(f"📊 **주요 시대:** {filtered_df['시대그룹'].mode()[0]}")
with c2: st.info(f"📂 **종목 다양성:** {filtered_df['국가유산종목'].nunique()}종")
with c3: st.success(f"현재 위치: **{st.session_state.selected_heritage}**")
