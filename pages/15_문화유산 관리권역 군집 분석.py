import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.preprocessing import StandardScaler


# =============================================================
# 1. 페이지 설정
# =============================================================
st.set_page_config(
    page_title="영천 문화유산 관리권역 군집 분석",
    page_icon="🗺️",
    layout="wide",
)

st.title("🗺️ 영천 문화유산 공간·가치 특성 기반 관리권역 분석")
st.caption(
    "문화유산의 위치(위도·경도), 가치점수, 시대점수를 표준화한 뒤 K-Means 군집분석을 적용하여 "
    "서로 유사한 공간·가치 특성을 가진 문화유산 관리권역을 탐색합니다."
)
st.info(
    "📌 본 군집은 환경 취약도·산불·홍수 등 재난 위험도를 예측한 결과가 아닙니다. "
    "군집 번호는 위험도 순위나 관리 우선순위를 의미하지 않습니다."
)
st.divider()


# =============================================================
# 2. 데이터 로드 및 전처리
# =============================================================
DATA_PATH = "data/processed/yc_clustering.csv"
REQUIRED_COLS = [
    "위도",
    "경도",
    "가치점수",
    "시대점수",
    "문화재명(국문)",
    "국가유산종목",
    "시대",
    "소재지상세",
]


@st.cache_data
def load_base_data():
    df = pd.read_csv(DATA_PATH)

    missing_cols = [col for col in REQUIRED_COLS if col not in df.columns]
    if missing_cols:
        raise ValueError(
            "군집 분석에 필요한 열이 없습니다: " + ", ".join(missing_cols)
        )

    numeric_cols = ["위도", "경도", "가치점수", "시대점수"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=numeric_cols).copy()

    df["소재지상세"] = (
        df["소재지상세"]
        .fillna("-")
        .astype(str)
        .str.replace("\t", " ", regex=False)
        .str.strip()
    )

    return df.reset_index(drop=True)


try:
    df_base = load_base_data()
except Exception as e:
    st.error(f"군집 분석 데이터 로드 실패: {e}")
    st.stop()

if len(df_base) < 3:
    st.error("군집 분석을 수행하려면 유효한 문화유산 데이터가 최소 3건 이상 필요합니다.")
    st.stop()


# =============================================================
# 3. 사이드바 설정
# =============================================================
st.sidebar.header("⚙️ 군집 분석 설정")

max_k = min(10, len(df_base) - 1)
default_k = min(3, max_k)

k_value = st.sidebar.slider(
    "군집 수(k) 설정",
    min_value=2,
    max_value=max_k,
    value=default_k,
)

st.sidebar.divider()
st.sidebar.subheader("💡 분석 지표 안내")

with st.sidebar.expander("💎 가치 점수 배점", expanded=False):
    st.caption(
        """
        * **10점:** 국보 / **9점:** 보물
        * **8점:** 사적, 천연기념물
        * **7점:** 국가민속문화유산
        * **6점:** 국가등록문화유산
        * **5점:** 경북 유형문화유산
        * **4점:** 경북 기념물/민속
        * **3점:** 경북 문화유산자료
        """
    )

with st.sidebar.expander("⏳ 시대 점수 산출", expanded=False):
    st.caption(
        """
        * **15점:** 선사시대 / **14점:** 삼한시대
        * **13점:** 신라/통일신라 / **11점:** 고려시대
        * **9점:** 조선 초기 / **7점:** 조선 중후기
        * **4점:** 근대/일제강점기 / **5점:** 기타
        """
    )

with st.sidebar.expander("📐 표준화와 군집분석", expanded=False):
    st.caption(
        "위도·경도·가치점수·시대점수는 단위와 범위가 서로 다르므로 StandardScaler로 "
        "각 변수를 표준화한 뒤 K-Means를 적용합니다. 따라서 특정 변수의 단위가 크다는 이유만으로 "
        "군집 결과를 지배하는 현상을 줄일 수 있습니다."
    )

with st.sidebar.expander("📊 실루엣 계수", expanded=False):
    st.caption(
        "실루엣 계수는 각 문화유산이 자신의 군집과 얼마나 잘 결합되어 있고 다른 군집과 얼마나 "
        "분리되어 있는지를 나타내는 군집 분리도 지표입니다. 예측 정확도나 신뢰확률은 아닙니다."
    )


# =============================================================
# 4. K-Means 군집 분석
# =============================================================
FEATURE_COLS = ["위도", "경도", "가치점수", "시대점수"]
features_raw = df_base[FEATURE_COLS].copy()

# 각 변수의 스케일을 맞춤
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features_raw)

# K-Means 군집화
kmeans = KMeans(
    n_clusters=k_value,
    init="k-means++",
    random_state=42,
    n_init=10,
)

cluster_zero_based = kmeans.fit_predict(features_scaled)
df_base["cluster_num"] = cluster_zero_based + 1
df_base["cluster"] = df_base["cluster_num"].astype(str)

# 실루엣 계수는 실제 0-based label을 사용해도 동일하지만,
# 군집 번호 표시와 독립적으로 원래 label을 사용한다.
sil_avg = silhouette_score(features_scaled, cluster_zero_based)
df_base["silhouette_val"] = silhouette_samples(
    features_scaled,
    cluster_zero_based,
)


# =============================================================
# 5. 군집별 요약 데이터
# =============================================================
summary = (
    df_base.groupby("cluster", as_index=False)
    .agg(
        가치점수=("가치점수", "mean"),
        시대점수=("시대점수", "mean"),
        유산수=("문화재명(국문)", "count"),
        평균군집분리도=("silhouette_val", "mean"),
    )
)
summary["cluster_int"] = summary["cluster"].astype(int)
summary = summary.sort_values("cluster_int").reset_index(drop=True)


# =============================================================
# 6. 상단 요약 지표
# =============================================================
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("분석 대상", f"{len(df_base):,}건")
with c2:
    st.metric("관리권역 수(k)", f"{k_value}개")
with c3:
    st.metric("전체 평균 가치점수", f"{df_base['가치점수'].mean():.2f}")
with c4:
    st.metric("군집 분리도(실루엣)", f"{sil_avg:.3f}")

st.caption(
    "※ 실루엣 계수는 군집 구조의 분리 정도를 확인하는 지표이며, 모델의 예측 정확도나 위험도 신뢰도를 의미하지 않습니다."
)
st.divider()


# =============================================================
# 7. 메인 분석 시각화
# =============================================================
col_left, col_right = st.columns([1.2, 1])

with col_left:
    st.subheader(f"📍 문화유산 관리권역 분포 (권역 1~{k_value})")
    st.caption(
        "점의 색은 K-Means로 구분된 군집을 나타내며, 점의 크기는 가치점수를 나타냅니다. "
        "같은 색은 유사한 입력 특성을 가진 군집이라는 뜻입니다."
    )

    fig_scatter = px.scatter(
        df_base,
        x="경도",
        y="위도",
        color="cluster",
        size="가치점수",
        hover_data=[
            "문화재명(국문)",
            "국가유산종목",
            "시대",
            "가치점수",
            "시대점수",
        ],
        color_discrete_sequence=px.colors.qualitative.Bold,
        template="plotly_white",
        category_orders={
            "cluster": [str(i) for i in range(1, k_value + 1)]
        },
    )
    fig_scatter.update_layout(legend_title_text="관리권역")
    st.plotly_chart(fig_scatter, use_container_width=True)

with col_right:
    st.subheader("📊 관리권역별 특성 비교")
    st.caption(
        "가치점수·시대점수와 권역 내 문화유산 수를 비교하여 각 군집의 상대적 특성을 확인합니다."
    )

    fig_radar = go.Figure()
    max_count = summary["유산수"].max()

    for _, row in summary.iterrows():
        size_score = (row["유산수"] / max_count * 10) if max_count else 0

        fig_radar.add_trace(
            go.Scatterpolar(
                r=[
                    row["가치점수"],
                    row["시대점수"],
                    size_score,
                    row["가치점수"],
                ],
                theta=[
                    "평균 가치",
                    "평균 시대",
                    "권역 규모",
                    "평균 가치",
                ],
                fill="toself",
                name=f"권역 {row['cluster']}",
            )
        )

    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 15],
            )
        ),
        showlegend=True,
        template="plotly_white",
        margin=dict(l=40, r=40, t=40, b=40),
    )
    st.plotly_chart(fig_radar, use_container_width=True)


# =============================================================
# 8. 군집별 요약표
# =============================================================
st.divider()
st.subheader("📋 관리권역별 요약")

summary_display = summary.rename(
    columns={
        "cluster": "관리권역",
        "가치점수": "평균 가치점수",
        "시대점수": "평균 시대점수",
        "유산수": "문화유산 수",
        "평균군집분리도": "평균 군집 분리도",
    }
)[
    [
        "관리권역",
        "문화유산 수",
        "평균 가치점수",
        "평균 시대점수",
        "평균 군집 분리도",
    ]
]

st.dataframe(
    summary_display,
    use_container_width=True,
    hide_index=True,
    column_config={
        "평균 가치점수": st.column_config.NumberColumn(format="%.2f"),
        "평균 시대점수": st.column_config.NumberColumn(format="%.2f"),
        "평균 군집 분리도": st.column_config.NumberColumn(format="%.3f"),
    },
)


# =============================================================
# 9. 군집별 상세 목록
# =============================================================
st.divider()
st.subheader("🔍 관리권역별 문화유산 목록")

cluster_labels = [str(i) for i in range(1, k_value + 1)]
tabs = st.tabs([f"권역 {c}" for c in cluster_labels])

for i, tab in enumerate(tabs):
    cluster_id = cluster_labels[i]

    with tab:
        cluster_df = (
            df_base[df_base["cluster"] == cluster_id]
            .sort_values("가치점수", ascending=False)
            .copy()
        )

        cnt = len(cluster_df)
        avg_v = cluster_df["가치점수"].mean()
        avg_e = cluster_df["시대점수"].mean()
        avg_s = cluster_df["silhouette_val"].mean()

        st.markdown(
            f"🚩 **권역 {cluster_id} 특성** | "
            f"대상 수: **{cnt}건** | "
            f"평균 가치점수: **{avg_v:.2f}** | "
            f"평균 시대점수: **{avg_e:.2f}** | "
            f"평균 군집 분리도: **{avg_s:.3f}**"
        )

        st.dataframe(
            cluster_df[
                [
                    "문화재명(국문)",
                    "국가유산종목",
                    "시대",
                    "소재지상세",
                    "가치점수",
                    "시대점수",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )


# =============================================================
# 10. 해석 안내
# =============================================================
st.sidebar.info(
    """
    **🗺️ 군집 분석 해석 가이드**

    - 입력 변수: 위도, 경도, 가치점수, 시대점수
    - 같은 권역: 네 변수의 표준화된 특성이 상대적으로 유사한 문화유산 집단
    - 권역 번호: 단순 군집 식별번호
    - 군집 분리도: 군집 간 분리 정도를 확인하는 참고지표

    본 분석에는 산불·홍수·기상·대기환경 위험 변수가 포함되지 않았으므로
    각 권역을 재난 위험권역으로 해석하지 않습니다.
    """
)

st.caption(
    "※ 본 결과는 공간·가치 특성을 이용한 탐색적 K-Means 군집분석입니다. "
    "문화유산의 실제 훼손 위험, 재난 발생 가능성 또는 관리 우선순위를 직접 예측하지 않습니다."
)
