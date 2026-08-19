import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.preprocessing import StandardScaler

# =================================================
# 페이지 설정
# =================================================
st.set_page_config(
    page_title="영천 국가유산 위험예측 군집분석",
    layout="wide"
)

st.title("🛡️ 국가유산 지능형 위험관리 시스템")
st.markdown("공간 정보와 유산 가치를 결합하여 **환경 재난 대비 관리 지구**를 도출합니다.")
st.divider()

# =================================================
# 데이터 로드 및 전처리
# =================================================
@st.cache_data
def load_base_data():
    df = pd.read_csv("data/processed/yc_clustering.csv")
    df = df.dropna(subset=['위도', '경도', '가치점수', '시대점수'])
    # 주소 내 탭 문자 제거
    df['소재지상세'] = df['소재지상세'].fillna("-").astype(str).str.replace('\t', ' ', regex=True).str.strip()
    return df

df_base = load_base_data()

# =================================================
# 사이드바 설정
# =================================================
st.sidebar.header("⚙️ 분석 엔진 설정")
k_value = st.sidebar.slider("군집 수(k) 설정", min_value=2, max_value=10, value=3)

st.sidebar.divider()
st.sidebar.subheader("💡 지표 산출 로직 안내")

with st.sidebar.expander("💎 가치 점수 배점", expanded=False):
    st.caption("""
    * **10점:** 국보 / **9점:** 보물
    * **8점:** 사적, 천연기념물
    * **7점:** 국가민속문화유산
    * **6점:** 국가등록문화유산
    * **5점:** 경북 유형문화유산
    * **4점:** 경북 기념물/민속
    * **3점:** 경북 문화유산자료
    """)

with st.sidebar.expander("⏳ 시대 점수 산출", expanded=False):
    st.caption("""
    * **15점:** 선사시대 / **14점:** 삼한시대
    * **13점:** 신라/통일신라 / **11점:** 고려시대
    * **9점:** 조선 초기 / **7점:** 조선 중후기
    * **4점:** 근대/일제강점기 / **5점:** 기타
    """)

# =================================================
# 실시간 K-Means 분석 수행 (공간 정보 강화 모델)
# =================================================
# 1. 분석 피처 선택
features_raw = df_base[['위도', '경도', '가치점수', '시대점수']]

# 2. 데이터 표준화 (위경도 데이터의 영향력을 점수 데이터와 동등하게 맞춤)
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features_raw)

# 3. K-Means 군집화
kmeans = KMeans(n_clusters=k_value, init='k-means++', random_state=42, n_init=10)
df_base['cluster_num'] = kmeans.fit_predict(features_scaled) + 1
df_base['cluster'] = df_base['cluster_num'].astype(str)

# 4. 실루엣 계수 계산
sil_avg = silhouette_score(features_scaled, df_base['cluster'])
df_base['silhouette_val'] = silhouette_samples(features_scaled, df_base['cluster'])

# =================================================
# 상단 요약 지표 (Metrics)
# =================================================
c1, c2, c3, c4 = st.columns(4)
with c1: st.metric("분석 대상 전수", "105건")
with c2: st.metric("위험 관리 지구(k)", f"{k_value}개")
with c3: st.metric("전체 평균 가치", f"{df_base['가치점수'].mean():.2f}")
with c4: st.metric("분석 신뢰도(실루엣)", f"{sil_avg:.3f}")

st.divider()

# =================================================
# 메인 분석 시각화
# =================================================
col_left, col_right = st.columns([1.2, 1])

with col_left:
    st.subheader(f"📍 위험관리 구역 분포 (Zone 1~{k_value})")
    # 표준화를 거쳤기 때문에 이제 색상(군집)이 지도상에서 구역별로 뭉치게 됩니다.
    fig_scatter = px.scatter(
        df_base, x="경도", y="위도", color="cluster", size="가치점수",
        hover_data=["문화재명(국문)", "국가유산종목", "시대"],
        color_discrete_sequence=px.colors.qualitative.Bold,
        template="plotly_white",
        category_orders={"cluster": [str(i) for i in range(1, k_value + 1)]}
    )
    fig_scatter.update_layout(legend_title_text='관리 지구')
    st.plotly_chart(fig_scatter, use_container_width=True)

with col_right:
    st.subheader("📊 지구별 환경 노출 특성")
    
    summary = df_base.groupby("cluster").agg({
        "가치점수": "mean", "시대점수": "mean", "문화재명(국문)": "count"
    }).rename(columns={"문화재명(국문)": "유산수"}).reset_index()
    
    summary['cluster_int'] = summary['cluster'].astype(int)
    summary = summary.sort_values('cluster_int')

    # 방사형 차트
    fig_radar = go.Figure()
    max_count = summary['유산수'].max()

    for i, row in summary.iterrows():
        fig_radar.add_trace(go.Scatterpolar(
            r=[row['가치점수'], row['시대점수'], (row['유산수']/max_count*10), row['가치점수']],
            theta=['평균 가치', '평균 시대', '지구 규모', '평균 가치'],
            fill='toself',
            name=f"지구 {row['cluster']}"
        ))

    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 15])),
        showlegend=True, template="plotly_white", margin=dict(l=40, r=40, t=40, b=40)
    )
    st.plotly_chart(fig_radar, use_container_width=True)

# =================================================
# 하단: 군집별 상세 목록
# =================================================
st.divider()
st.subheader("🔍 지구별 관리 대상 목록")

cluster_labels = [str(i) for i in range(1, k_value + 1)]
tabs = st.tabs([f"지구 {c}" for c in cluster_labels])

for i, tab in enumerate(tabs):
    cluster_id = cluster_labels[i]
    with tab:
        cluster_df = df_base[df_base['cluster'] == cluster_id].sort_values("가치점수", ascending=False)
        
        cnt = len(cluster_df)
        avg_v = cluster_df['가치점수'].mean()
        avg_e = cluster_df['시대점수'].mean()
        
        st.markdown(f"🚩 **현재 지구 정보** | 대상 수: **{cnt}건** | 지구 평균 가치: **{avg_v:.2f}** | 지구 평균 시대: **{avg_e:.2f}**")
        
        st.dataframe(
            cluster_df[["문화재명(국문)", "국가유산종목", "시대", "소재지상세"]],
            use_container_width=True, hide_index=True
        )

st.sidebar.warning("""
**⚠️ 위험예측 분석 가이드:**
현재 모델은 **데이터 표준화**를 통해 공간 정보의 중요도를 높였습니다. 
지도상에서 같은 색으로 묶인 지구는 향후 산불이나 홍수 등 환경 재난 시 **동일한 위험권역**으로 분석될 가능성이 매우 높습니다.
""")
