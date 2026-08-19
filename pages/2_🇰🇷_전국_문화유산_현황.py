import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# =================================================
# 페이지 설정
# =================================================
st.set_page_config(
    page_title="전국 문화유산 현황",
    page_icon="🏛",
    layout="wide"
)

# =================================================
# 제목
# =================================================
st.markdown("""
<h1 style="font-size:34px; margin-bottom:5px;">
🇰🇷 전국 문화유산 현황
</h1>
<div style="font-size:17px; color:#6b7280; margin-bottom:20px;">
국가유산 공공데이터와 2025년 최신 인구 통계를 활용한 시각화
</div>
""", unsafe_allow_html=True)

st.divider()

# =================================================
# 데이터 불러오기
# =================================================
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("data/raw/all_heritage.csv")
        df.columns = df.columns.str.strip()
        return df
    except Exception as e:
        st.error(f"데이터 파일을 찾을 수 없습니다: {e}")
        return None

df = load_data()

if df is not None:
    # =================================================
    # 데이터 전처리 (경북 데이터 추출)
    # =================================================
    gb_df = df[df["시도명"] == "경북"].copy()

    # =================================================
    # 주요 지표 (Metrics)
    # =================================================
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("전체 문화유산", f"{len(df):,}개")
    with c2:
        st.metric("문화유산 종목 수", f"{df['국가유산종목'].nunique()}종")
    with c3:
        st.metric("시도 수", f"{df['시도명'].nunique()}개")
    with c4:
        st.metric("가장 많은 지역", df["시도명"].value_counts().idxmax())

    st.markdown("<br>", unsafe_allow_html=True)
    st.divider()
    # =================================================
    # 1행: 전국 분포 (Treemap) & 종목별 현황 (Bubble)
    # =================================================
    row1_left, row1_right = st.columns([1.2, 1])

    with row1_left:
        st.markdown("### 🗺 시도별 문화유산 분포")
        region_count = df["시도명"].value_counts().reset_index()
        region_count.columns = ["시도명", "개수"]

        fig1 = px.treemap(
            region_count, path=["시도명"], values="개수",
            color="개수", color_continuous_scale="GnBu"
        )
        fig1.update_layout(margin=dict(t=20, l=10, r=10, b=10), height=450)
        fig1.update_traces(texttemplate="<b>%{label}</b><br>%{value}개", textfont_size=16)
        st.plotly_chart(fig1, use_container_width=True)

    with row1_right:
        st.markdown("### 🫧 국가유산 종목별 현황")
        type_count = df["국가유산종목"].value_counts().reset_index()
        type_count.columns = ["국가유산종목", "개수"]
        
        # 버블에 표시할 텍스트 컬럼 추가 (줄바꿈 <br> 처리)
        type_count["표시텍스트"] = type_count["국가유산종목"] + "<br>(" + type_count["개수"].astype(str) + "건)"
        
        n = len(type_count)
        cols = 4
        type_count["x"] = [i % cols for i in range(n)]
        type_count["y"] = [-(i // cols) for i in range(n)]
        
        fig2 = px.scatter(
            type_count, x="x", y="y", size="개수", color="개수",
            text="표시텍스트", size_max=60, color_continuous_scale="Blues"
        )
        fig2.update_traces(textposition="middle center", marker=dict(opacity=0.85, line=dict(width=1, color="white")))
        fig2.update_layout(
            xaxis=dict(visible=False), 
            yaxis=dict(visible=False), 
            margin=dict(t=20, l=10, r=10, b=10), 
            height=450, 
            showlegend=False
        )
        st.plotly_chart(fig2, use_container_width=True)
        '''
        st.markdown("### 🫧 국가유산 종목별 현황")
        type_count = df["국가유산종목"].value_counts().reset_index()
        type_count.columns = ["국가유산종목", "개수"]

        n = len(type_count)
        cols = 4
        type_count["x"] = [i % cols for i in range(n)]
        type_count["y"] = [-(i // cols) for i in range(n)]

        fig2 = px.scatter(
            type_count, x="x", y="y", size="개수", color="개수",
            text="국가유산종목", size_max=60, color_continuous_scale="Blues"
        )
        fig2.update_traces(textposition="middle center", marker=dict(opacity=0.85, line=dict(width=1, color="white")))
        fig2.update_layout(xaxis=dict(visible=False), yaxis=dict(visible=False), 
                          margin=dict(t=20, l=10, r=10, b=10), height=450, showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)
        '''

    # =================================================
    # 2행: 경북 상세 (Polar Bar & Heatmap)
    # =================================================
    st.markdown("<br>", unsafe_allow_html=True)
    row2_left, row2_right = st.columns([1, 1])

    with row2_left:
        st.markdown("### 🌀 경북 지역 문화유산 분포 (상위 15개)")
        city_count = gb_df["시군구명"].value_counts().head(15).reset_index()
        city_count.columns = ["시군구명", "개수"]

        fig3 = px.bar_polar(
            city_count, r="개수", theta="시군구명",
            color="개수", color_continuous_scale="Tealgrn"
        )
        fig3.update_layout(height=500, margin=dict(t=50, b=50), coloraxis_showscale=False)
        st.plotly_chart(fig3, use_container_width=True)

    with row2_right:
        st.markdown("### 🌡 경북 시군구별 종목 현황")
        heatmap_df = pd.pivot_table(gb_df, index="시군구명", columns="국가유산종목", aggfunc="size", fill_value=0)
        top_cities = gb_df["시군구명"].value_counts().head(15).index
        heatmap_df = heatmap_df.loc[top_cities]
    
        # 각 시군구별 총 합계 수 계산 및 인덱스명 변경
        row_totals = heatmap_df.sum(axis=1)
        heatmap_df.index = [f"{city} ({row_totals[city]}건)" for city in heatmap_df.index]
    
        fig4 = px.imshow(heatmap_df, text_auto=True, color_continuous_scale="YlGnBu", aspect="auto")
        fig4.update_layout(height=500, margin=dict(t=20, l=10, r=10, b=10), coloraxis_showscale=False)
        st.plotly_chart(fig4, use_container_width=True)
        
    # =================================================
    # 3행: 영천시 특징 (Radar) & 인구 대비 밀도 (Bubble)
    # =================================================
    st.markdown("<br>", unsafe_allow_html=True)
    row3_left, row3_right = st.columns(2)

    with row3_left:
        st.markdown("### 🎯 영천 문화유산 종목 특징")
        yc_df = gb_df[gb_df["시군구명"] == "영천시"]
        if not yc_df.empty:
            type_ratio = yc_df["국가유산종목"].value_counts(normalize=True).reset_index()
            type_ratio.columns = ["종목", "비율"]
            type_ratio["비율"] = type_ratio["비율"] * 100

            
            fig5 = px.line_polar(type_ratio.head(8), r="비율", theta="종목", line_close=True)
            fig5.update_traces(fill="toself", line_color="#008080")
            fig5.update_layout(height=500, margin=dict(t=30, b=30))
            st.plotly_chart(fig5, use_container_width=True)
        else:
            st.warning("영천시 데이터가 존재하지 않습니다.")

    with row3_right:
        st.markdown("### 👥 인구 대비 문화유산 밀도")
        
        heritage_count = gb_df["시군구명"].value_counts().reset_index()
        heritage_count.columns = ["시군구명", "문화유산수"]

        # 2025년 인구 데이터 - KOSIS 국가통계포털 주민등록인구
        actual_pop_data = {
            "시군구명": ["포항시", "경주시", "김천시", "안동시", "구미시", "영주시", "영천시", "상주시", 
                        "문경시", "경산시", "의성군", "청송군", "영양군", "영덕군", "청도군", "고령군", 
                        "성주군", "칠곡군", "예천군", "봉화군", "울진군", "울릉군"],
            "인구": [488707, 244055, 133791, 152610, 403883, 97162, 95185, 89888, 
                    65348, 263853, 47902, 23363, 15941, 32698, 40117, 29667, 
                    40720, 104842, 53887, 28315, 45896, 8696]
        }
        pop_df = pd.DataFrame(actual_pop_data)
        density_df = pd.merge(heritage_count, pop_df, on="시군구명", how="inner")
        density_df["밀도"] = (density_df["문화유산수"] / density_df["인구"]) * 10000

        # 영천시 강조를 위한 색상 및 라인 설정
        # 영천시는 붉은색(#FF4B4B), 나머지는 청록색 계열
        colors = ['#FF4B4B' if city == '영천시' else '#008080' for city in density_df['시군구명']]
        
        
        fig6 = go.Figure()

        fig6.add_trace(go.Scatter(
            x=density_df["인구"],
            y=density_df["문화유산수"],
            mode='markers+text',
            marker=dict(
                size=density_df["밀도"] * 1.5, 
                color=colors,
                opacity=0.7,
                line=dict(width=2, color='White')
            ),
            text=density_df["시군구명"],
            textposition="top center",
            hovertemplate="<b>%{text}</b><br>인구: %{x}<br>보유수: %{y}<br>밀도: %{marker.size}<extra></extra>"
        ))

        fig6.update_layout(
            height=500,
            margin=dict(t=20, l=10, r=10, b=10),
            xaxis_title="2025년 주민등록 인구 수(출처:KOSIS 국가통계포털)",
            yaxis_title="문화유산 보유 수",
            showlegend=False
        )
        st.plotly_chart(fig6, use_container_width=True)

    # =================================================
    # 하단 설명
    # =================================================
    st.divider()
    
    st.info("""
    #### 💡 데이터 기반 지역 문화유산 분석 및 전략적 통찰
    
    * **📌 전국 유산 거버넌스 (Treemap & Bubble)**
        전국적 분포와 종목별 비중을 통해 대한민국 문화유산의 거시적 현황을 파악하고, 각 유산이 가진 위상과 규모를 직관적으로 비교합니다.
    
    * **📌 지역별 자산 포트폴리오 (Polar & Heatmap)**
        경상북도 내 시군별 문화유산 편차와 종목 구성을 분석하여, 각 지역이 보유한 문화 자산의 고유한 정체성과 집중도를 상세히 진단합니다.
    
    * **📌 영천시 특화 분석 및 경쟁력 진단 (Radar & Density)**
        * **🎯 종목 특징:**
            영천시가 보유한 문화유산의 종목별 강점을 분석하여 지역 특화 콘텐츠 개발을 위한 기초 데이터를 제공합니다.
        * **🌀 밀도 분석:**
            2025년 최신 인구 데이터 대비 유산 보유량을 산출하여 인구 소멸 위기 속 문화유산의 보존 가치와 잠재적 밀도를 측정하였습니다.  
            &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;특히 경북 내 타 시군 대비 영천시의 문화적 위상과 자산 효율성을 한눈에 비교할 수 있도록 설계했습니다.
    """)
