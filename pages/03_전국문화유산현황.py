import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


# =================================================
# 페이지 설정
# =================================================
st.set_page_config(
    page_title="전국 및 경북 문화유산 분포 현황",
    page_icon="🏛",
    layout="wide"
)


# =================================================
# 페이지 제목
# =================================================
st.markdown(
    """
    <h1 style="font-size:34px; margin-bottom:5px;">
        전국 및 경북 문화유산 분포 현황
    </h1>

    <div style="
        font-size:17px;
        color:#6b7280;
        margin-bottom:20px;
    ">
        국가유산 공공데이터와 2025년 주민등록인구 자료를 활용한
        지역별 문화유산 분포 분석
    </div>
    """,
    unsafe_allow_html=True
)

st.divider()


# =================================================
# 데이터 불러오기
# =================================================
@st.cache_data
def load_data():
    try:
        df = pd.read_csv(
            "data/raw/all_heritage.csv"
        )

        df.columns = (
            df.columns
            .str.strip()
        )

        return df

    except Exception as e:
        st.error(
            f"데이터 파일을 찾을 수 없습니다: {e}"
        )

        return None


df = load_data()


# =================================================
# 데이터가 정상적으로 로드된 경우
# =================================================
if df is not None:

    # =================================================
    # 데이터 전처리
    # =================================================

    required_columns = [
        "시도명",
        "시군구명",
        "국가유산종목",
    ]

    missing_columns = [
        col
        for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        st.error(
            "필요한 컬럼이 없습니다: "
            f"{missing_columns}"
        )

        st.stop()


    # 결측 제거
    df = df.dropna(
        subset=[
            "시도명",
            "국가유산종목",
        ]
    ).copy()


    # 경북 데이터 추출
    gb_df = (
        df[
            df["시도명"] == "경북"
        ]
        .copy()
    )


    # 영천 데이터 추출
    yc_df = (
        gb_df[
            gb_df["시군구명"] == "영천시"
        ]
        .copy()
    )


    # =================================================
    # 주요 지표
    # =================================================
    st.subheader("📌 전국 문화유산 주요 현황")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "전체 문화유산",
            f"{len(df):,}개"
        )

    with c2:
        st.metric(
            "문화유산 종목 수",
            f"{df['국가유산종목'].nunique()}종"
        )

    with c3:
        st.metric(
            "시도 수",
            f"{df['시도명'].nunique()}개"
        )

    with c4:

        if not df.empty:
            top_region = (
                df["시도명"]
                .value_counts()
                .idxmax()
            )

            st.metric(
                "문화유산이 가장 많은 시도",
                top_region
            )

        else:
            st.metric(
                "문화유산이 가장 많은 시도",
                "-"
            )


    st.markdown("<br>", unsafe_allow_html=True)
    st.divider()


    # =================================================
    # 1행
    # 전국 시도별 분포 / 국가유산 종목별 현황
    # =================================================

    row1_left, row1_right = st.columns(
        [1.2, 1]
    )


    # -------------------------------------------------
    # 1-1. 시도별 문화유산 분포
    # -------------------------------------------------
    with row1_left:

        st.markdown(
            "### 🗺 시도별 문화유산 분포"
        )

        region_count = (
            df["시도명"]
            .value_counts()
            .reset_index()
        )

        region_count.columns = [
            "시도명",
            "개수",
        ]

        fig1 = px.treemap(
            region_count,
            path=["시도명"],
            values="개수",
            color="개수",
            color_continuous_scale="GnBu",
        )

        fig1.update_layout(
            margin=dict(
                t=20,
                l=10,
                r=10,
                b=10,
            ),
            height=450,
            coloraxis_showscale=False,
        )

        fig1.update_traces(
            texttemplate=(
                "<b>%{label}</b><br>"
                "%{value}개"
            ),
            textfont_size=16,
        )

        st.plotly_chart(
            fig1,
            use_container_width=True,
        )


    # -------------------------------------------------
    # 1-2. 국가유산 종목별 현황
    # -------------------------------------------------
    with row1_right:

        st.markdown(
            "### 🫧 국가유산 종목별 현황"
        )

        type_count = (
            df["국가유산종목"]
            .value_counts()
            .reset_index()
        )

        type_count.columns = [
            "국가유산종목",
            "개수",
        ]

        type_count[
            "표시텍스트"
        ] = (
            type_count["국가유산종목"]
            + "<br>("
            + type_count["개수"].astype(str)
            + "건)"
        )

        n = len(type_count)

        cols = 4

        type_count["x"] = [
            i % cols
            for i in range(n)
        ]

        type_count["y"] = [
            -(i // cols)
            for i in range(n)
        ]

        fig2 = px.scatter(
            type_count,
            x="x",
            y="y",
            size="개수",
            color="개수",
            text="표시텍스트",
            size_max=60,
            color_continuous_scale="Blues",
            hover_name="국가유산종목",
            hover_data={
                "개수": True,
                "x": False,
                "y": False,
                "표시텍스트": False,
            },
        )

        fig2.update_traces(
            textposition="middle center",
            marker=dict(
                opacity=0.85,
                line=dict(
                    width=1,
                    color="white",
                ),
            ),
        )

        fig2.update_layout(
            xaxis=dict(
                visible=False
            ),
            yaxis=dict(
                visible=False
            ),
            margin=dict(
                t=20,
                l=10,
                r=10,
                b=10,
            ),
            height=450,
            showlegend=False,
            coloraxis_showscale=False,
        )

        st.plotly_chart(
            fig2,
            use_container_width=True,
        )


    # =================================================
    # 2행
    # 경북 시군별 분포 / 종목별 Heatmap
    # =================================================

    st.markdown(
        "<br>",
        unsafe_allow_html=True,
    )

    row2_left, row2_right = st.columns(
        [1, 1]
    )


    # -------------------------------------------------
    # 2-1. 경북 지역 문화유산 분포
    # -------------------------------------------------
    with row2_left:

        st.markdown(
            "### 🌀 경북 지역 문화유산 분포 (상위 15개)"
        )

        city_count = (
            gb_df["시군구명"]
            .value_counts()
            .head(15)
            .reset_index()
        )

        city_count.columns = [
            "시군구명",
            "개수",
        ]

        if not city_count.empty:

            fig3 = px.bar_polar(
                city_count,
                r="개수",
                theta="시군구명",
                color="개수",
                color_continuous_scale="Teal",
            )

            fig3.update_layout(
                height=500,
                margin=dict(
                    t=50,
                    b=50,
                ),
                coloraxis_showscale=False,
            )

            st.plotly_chart(
                fig3,
                use_container_width=True,
            )

        else:

            st.warning(
                "경북 지역 문화유산 데이터가 없습니다."
            )


    # -------------------------------------------------
    # 2-2. 경북 시군구별 종목 현황
    # -------------------------------------------------
    with row2_right:

        st.markdown(
            "### 🌡 경북 시군구별 종목 현황"
        )

        heatmap_df = pd.pivot_table(
            gb_df,
            index="시군구명",
            columns="국가유산종목",
            aggfunc="size",
            fill_value=0,
        )

        top_cities = (
            gb_df["시군구명"]
            .value_counts()
            .head(15)
            .index
        )

        available_cities = [
            city
            for city in top_cities
            if city in heatmap_df.index
        ]

        heatmap_df = (
            heatmap_df
            .loc[available_cities]
        )

        row_totals = (
            heatmap_df
            .sum(axis=1)
        )

        # y축 표시용 라벨
        display_labels = []

        for city in heatmap_df.index:

            total = row_totals[city]

            if city == "영천시":
                display_labels.append(
                    f"★ 영천시 ({total}건)"
                )
            else:
                display_labels.append(
                    f"{city} ({total}건)"
                )


        heatmap_display = (
            heatmap_df.copy()
        )

        heatmap_display.index = (
            display_labels
        )


        fig4 = px.imshow(
            heatmap_display,
            text_auto=True,
            color_continuous_scale="YlGnBu",
            aspect="auto",
        )


        # 영천시 행 강조
        city_list = list(
            heatmap_df.index
        )

        if "영천시" in city_list:

            yc_idx = (
                city_list
                .index("영천시")
            )

            fig4.add_shape(
                type="rect",
                x0=-0.5,
                x1=len(
                    heatmap_display.columns
                ) - 0.5,
                y0=yc_idx - 0.5,
                y1=yc_idx + 0.5,
                line=dict(
                    color="#e74c3c",
                    width=3,
                ),
                fillcolor="rgba(0,0,0,0)",
            )


        fig4.update_layout(
            height=500,
            margin=dict(
                t=20,
                l=10,
                r=10,
                b=10,
            ),
            coloraxis_showscale=False,
        )

        st.plotly_chart(
            fig4,
            use_container_width=True,
        )


    # =================================================
    # 3행
    # 영천시 종목 특징 / 인구 대비 문화유산 밀도
    # =================================================

    st.markdown(
        "<br>",
        unsafe_allow_html=True,
    )

    row3_left, row3_right = st.columns(
        2
    )


    # -------------------------------------------------
    # 3-1. 영천 문화유산 종목 특징
    # -------------------------------------------------
    with row3_left:

        st.markdown(
            "### 🎯 영천 문화유산 종목 구성 특징"
        )

        if not yc_df.empty:

            type_ratio = (
                yc_df[
                    "국가유산종목"
                ]
                .value_counts(
                    normalize=True
                )
                .reset_index()
            )

            type_ratio.columns = [
                "종목",
                "비율",
            ]

            type_ratio["비율"] = (
                type_ratio["비율"]
                * 100
            )


            top_type_ratio = (
                type_ratio
                .head(8)
            )


            fig5 = px.line_polar(
                top_type_ratio,
                r="비율",
                theta="종목",
                line_close=True,
                markers=True,
            )

            fig5.update_traces(
                fill="toself",
            )

            fig5.update_layout(
                height=500,
                margin=dict(
                    t=30,
                    b=30,
                ),
            )

            st.plotly_chart(
                fig5,
                use_container_width=True,
            )

        else:

            st.warning(
                "영천시 문화유산 데이터가 존재하지 않습니다."
            )


    # -------------------------------------------------
    # 3-2. 인구 대비 문화유산 밀도
    # -------------------------------------------------
    with row3_right:

        st.markdown(
            "### 👥 인구 대비 문화유산 밀도"
        )


        heritage_count = (
            gb_df[
                "시군구명"
            ]
            .value_counts()
            .reset_index()
        )

        heritage_count.columns = [
            "시군구명",
            "문화유산수",
        ]


        # -------------------------------------------------
        # 2025년 주민등록인구 데이터
        # KOSIS 국가통계포털 기준
        # -------------------------------------------------

        actual_pop_data = {

            "시군구명": [
                "포항시",
                "경주시",
                "김천시",
                "안동시",
                "구미시",
                "영주시",
                "영천시",
                "상주시",
                "문경시",
                "경산시",
                "의성군",
                "청송군",
                "영양군",
                "영덕군",
                "청도군",
                "고령군",
                "성주군",
                "칠곡군",
                "예천군",
                "봉화군",
                "울진군",
                "울릉군",
            ],

            "인구": [
                488707,
                244055,
                133791,
                152610,
                403883,
                97162,
                95185,
                89888,
                65348,
                263853,
                47902,
                23363,
                15941,
                32698,
                40117,
                29667,
                40720,
                104842,
                53887,
                28315,
                45896,
                8696,
            ],
        }


        pop_df = (
            pd.DataFrame(
                actual_pop_data
            )
        )


        density_df = pd.merge(
            heritage_count,
            pop_df,
            on="시군구명",
            how="inner",
        )


        # 인구 1만 명당 문화유산 수
        density_df[
            "문화유산밀도"
        ] = (
            density_df[
                "문화유산수"
            ]
            / density_df[
                "인구"
            ]
            * 10000
        )


        # -------------------------------------------------
        # 버블 크기용 값
        # 최소 크기를 보정하여 작은 값도 보이도록 설정
        # -------------------------------------------------

        density_df[
            "버블크기"
        ] = (
            density_df[
                "문화유산밀도"
            ]
            .clip(
                lower=1
            )
            * 4
        )


        # -------------------------------------------------
        # 영천시 강조 색상
        # -------------------------------------------------

        colors = [
            "#FF4B4B"
            if city == "영천시"
            else "#008080"
            for city
            in density_df[
                "시군구명"
            ]
        ]


        fig6 = go.Figure()


        fig6.add_trace(
            go.Scatter(
                x=density_df["인구"],
                y=density_df["문화유산수"],
                mode="markers+text",

                marker=dict(
                    size=density_df[
                        "버블크기"
                    ],
                    color=colors,
                    opacity=0.7,
                    line=dict(
                        width=2,
                        color="White",
                    ),
                ),

                text=density_df[
                    "시군구명"
                ],

                textposition=(
                    "top center"
                ),

                customdata=(
                    density_df[
                        [
                            "문화유산밀도",
                        ]
                    ]
                    .values
                ),

                hovertemplate=(
                    "<b>%{text}</b><br>"
                    "2025년 주민등록인구: "
                    "%{x:,}명<br>"
                    "문화유산 보유 수: "
                    "%{y}개<br>"
                    "인구 1만 명당 문화유산: "
                    "%{customdata[0]:.2f}개"
                    "<extra></extra>"
                ),
            )
        )


        fig6.update_layout(

            height=500,

            margin=dict(
                t=20,
                l=10,
                r=10,
                b=10,
            ),

            xaxis_title=(
                "2025년 주민등록인구 "
                "(출처: KOSIS 국가통계포털)"
            ),

            yaxis_title=(
                "문화유산 보유 수"
            ),

            showlegend=False,
        )


        st.plotly_chart(
            fig6,
            use_container_width=True,
        )


        st.caption(
            "※ 버블의 크기는 인구 1만 명당 문화유산 수를 "
            "시각적으로 표현한 값입니다."
        )


    # =================================================
    # 분석 해석
    # =================================================

    st.divider()


    st.info(
        """
### 💡 데이터 기반 지역 문화유산 분석

**📌 전국 문화유산 분포**

시도별 문화유산 수와 국가유산 종목별 분포를 통해
대한민국 문화유산의 지역별 분포 특성과 주요 종목 구성을
직관적으로 확인할 수 있다.

**📌 경북 지역 문화유산 분포**

경상북도 시군별 문화유산 보유 수와 국가유산 종목 구성을
비교함으로써 지역별 문화유산 분포 차이와 종목별 특징을
파악할 수 있다.

**📌 영천시 문화유산 특성**

영천시가 보유한 문화유산의 종목별 구성 비율을 통해
영천 지역 문화유산의 상대적 특징을 확인할 수 있다.

**📌 인구 대비 문화유산 밀도**

2025년 주민등록인구 대비 문화유산 보유 수를 산출하여
경북 각 시군의 **인구 1만 명당 문화유산 수**를 비교하였다.

이를 통해 단순한 문화유산 보유 수뿐 아니라
인구 규모를 함께 고려한 지역별 문화유산 분포 특성을
확인할 수 있으며, 영천시의 상대적인 문화유산 보유 특성을
다른 시군과 비교할 수 있다.
"""
    )


    # =================================================
    # 데이터 출처
    # =================================================

    st.divider()

    st.subheader(
        "🗂 데이터 출처 및 해석 유의사항"
    )


    st.markdown(
        """
- **문화유산 데이터:** 국가유산 관련 공공데이터
- **지역 구분:** 전국 시도 및 경상북도 시군구
- **인구 데이터:** 2025년 주민등록인구 자료
- **인구 대비 문화유산 밀도:**  
  `문화유산 수 ÷ 주민등록인구 × 10,000`

※ 본 페이지의 인구 대비 문화유산 밀도는
지역의 문화적 가치나 문화유산 관리 수준을 직접 측정하는
지표가 아니라, **지역별 문화유산 분포를 인구 규모와 함께
비교하기 위한 상대적 지표**입니다.
"""
    )


    st.caption(
        "선화여고 · 영천 헤리티지 AI 탐구단"
    )
