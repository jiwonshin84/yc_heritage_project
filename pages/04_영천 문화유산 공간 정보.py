import re

import folium
import pandas as pd
import streamlit as st
from folium.plugins import HeatMap, MarkerCluster
from streamlit_folium import st_folium


# ==========================================================
# 페이지 설정
# ==========================================================
st.set_page_config(
    page_title="영천 국가유산 공간 정보",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ==========================================================
# 페이지 제목
# ==========================================================
st.markdown(
    """
    <h1 style="font-size:34px; margin-bottom:5px;">
        🏛️ 영천 국가유산 공간 정보
    </h1>
    <div style="font-size:16px; color:#6b7280; margin-bottom:8px;">
        영천 지역 국가유산의 위치, 시대, 종목, 소재지 정보를 지도 기반으로 탐색하고
        문화유산의 공간적 분포를 확인합니다.
    </div>
    """,
    unsafe_allow_html=True,
)

st.caption(
    "※ 시대 구분은 원자료의 시대명을 지도 시각화와 비교 분석을 위해 "
    "간소화한 분류입니다."
)


# ==========================================================
# 데이터 로드 및 전처리
# ==========================================================
@st.cache_data(show_spinner=False)
def load_data():
    file_path = "data/processed/yc_heritage_detail_enriched.csv"

    try:
        df = pd.read_csv(
            file_path,
            encoding="utf-8-sig",
        )

    except UnicodeDecodeError:
        df = pd.read_csv(
            file_path,
            encoding="utf-8",
        )

    except Exception as e:
        raise RuntimeError(
            f"문화유산 데이터 파일을 불러오지 못했습니다: {e}"
        ) from e

    # 컬럼명 공백 정리
    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
    )

    # ------------------------------------------------------
    # 필수 컬럼 확인
    # ------------------------------------------------------
    required_columns = [
        "문화재명(국문)",
        "위도",
        "경도",
    ]

    missing_columns = [
        col
        for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "문화유산 데이터에 필요한 컬럼이 없습니다: "
            f"{missing_columns}"
        )

    # ------------------------------------------------------
    # 위도 / 경도 숫자형 변환
    # ------------------------------------------------------
    df["위도"] = pd.to_numeric(
        df["위도"],
        errors="coerce",
    )

    df["경도"] = pd.to_numeric(
        df["경도"],
        errors="coerce",
    )

    # 좌표 없는 데이터 제거
    df = (
        df
        .dropna(
            subset=[
                "위도",
                "경도",
            ]
        )
        .copy()
    )

    # ------------------------------------------------------
    # 선택 컬럼이 없을 경우 기본값 생성
    # ------------------------------------------------------
    if "시대" not in df.columns:
        df["시대"] = "미상"

    if "국가유산종목" not in df.columns:
        df["국가유산종목"] = "미상"
    else:
        df["국가유산종목"] = (
            df["국가유산종목"]
            .fillna("미상")
            .astype(str)
            .str.strip()
        )

    if "소재지상세" not in df.columns:
        df["소재지상세"] = "-"
    else:
        df["소재지상세"] = (
            df["소재지상세"]
            .fillna("-")
            .astype(str)
            .str.strip()
        )

    if "이미지URL" not in df.columns:
        df["이미지URL"] = ""

    # ------------------------------------------------------
    # 문화유산명 정리
    # ------------------------------------------------------
    df["문화재명(국문)"] = (
        df["문화재명(국문)"]
        .fillna("명칭 미상")
        .astype(str)
        .str.strip()
    )

    # ------------------------------------------------------
    # 시대 간소화 함수
    # ------------------------------------------------------
    def simplify_era(text):
        if pd.isna(text):
            return "기타"

        text = str(text).strip()

        if not text or text.lower() == "nan":
            return "기타"

        if "청동기" in text:
            return "청동기"

        if any(
            x in text
            for x in [
                "통일신라",
                "신라시대 후기",
            ]
        ):
            return "통일신라"

        if "신라" in text:
            return "신라"

        if "고려" in text:
            if any(
                x in text
                for x in [
                    "초기",
                    "전기",
                ]
            ):
                return "고려초기"

            if any(
                x in text
                for x in [
                    "말기",
                    "후기",
                ]
            ):
                return "고려후기"

            return "고려"

        # 조선 초기 왕명
        if any(
            k in text
            for k in [
                "태조",
                "정종",
                "태종",
                "세종",
                "문종",
                "단종",
                "세조",
                "예종",
                "성종",
                "연산군",
                "중종",
                "인종",
            ]
        ):
            return "조선초기"

        # 조선 후기 왕명
        if any(
            k in text
            for k in [
                "광해군",
                "인조",
                "효종",
                "현종",
                "숙종",
                "경종",
                "영조",
                "정조",
                "순조",
                "헌종",
                "철종",
                "고종",
                "순종",
            ]
        ):
            return "조선후기"

        if "조선" in text:
            if any(
                x in text
                for x in [
                    "초기",
                    "전기",
                ]
            ):
                return "조선초기"

            if any(
                x in text
                for x in [
                    "말기",
                    "후기",
                ]
            ):
                return "조선후기"

            return "조선"

        if "대한제국" in text:
            return "대한제국"

        # 숫자 연도가 포함된 경우 보조적으로 분류
        year_match = re.search(
            r"(?:^|[^0-9])(\d{4})(?:[^0-9]|$)",
            text,
        )

        if year_match:
            try:
                yr = int(
                    year_match.group(1)
                )

                if yr < 700:
                    return "신라"
                elif yr < 936:
                    return "통일신라"
                elif yr < 1392:
                    return "고려"
                elif yr < 1600:
                    return "조선초기"
                elif yr < 1897:
                    return "조선후기"
                elif yr < 1910:
                    return "대한제국"

            except ValueError:
                pass

        return "기타"

    df["시대그룹"] = (
        df["시대"]
        .apply(simplify_era)
    )

    # ------------------------------------------------------
    # 좌표 범위 기본 점검
    # 너무 비정상적인 값은 제거
    # ------------------------------------------------------
    df = df[
        df["위도"].between(
            -90,
            90,
        )
        &
        df["경도"].between(
            -180,
            180,
        )
    ].copy()

    return (
        df
        .reset_index(drop=True)
    )


# ==========================================================
# 데이터 로드
# ==========================================================
try:
    df = load_data()

except Exception as e:
    st.error(
        f"❌ 데이터 로드 실패: {e}"
    )
    st.stop()


if df.empty:
    st.warning(
        "표시할 영천 국가유산 데이터가 없습니다."
    )
    st.stop()


# ==========================================================
# 화면 스타일
# ==========================================================
st.markdown(
    """
    <style>
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #ffffff;
        border: 1px solid #b8d4fc !important;
        border-radius: 12px;
        padding: 10px 10px;
        margin-bottom: 25px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ==========================================================
# 검색 / 필터
# ==========================================================
with st.container(
    border=True
):
    f_col1, f_col2, f_col3, f_col4 = st.columns(
        [
            2,
            1.5,
            1.5,
            1,
        ]
    )

    # ------------------------------------------------------
    # 문화유산명 검색
    # ------------------------------------------------------
    with f_col1:
        search_query = st.text_input(
            "유산 명칭 검색",
            placeholder="명칭을 입력하세요",
        )

    # ------------------------------------------------------
    # 시대 필터
    # ------------------------------------------------------
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

    existing_eras = [
        era
        for era in era_order
        if era in df["시대그룹"].unique()
    ]

    with f_col2:
        selected_era = st.selectbox(
            "시대 선택",
            [
                "전체",
                *existing_eras,
            ],
        )

    # ------------------------------------------------------
    # 종목 필터
    # ------------------------------------------------------
    type_options = [
        "전체",
        *sorted(
            df[
                "국가유산종목"
            ]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        ),
    ]

    with f_col3:
        selected_type = st.selectbox(
            "종목 선택",
            type_options,
        )

    # ------------------------------------------------------
    # 필터 적용
    # ------------------------------------------------------
    filtered_df = (
        df.copy()
    )

    if search_query.strip():
        filtered_df = filtered_df[
            filtered_df[
                "문화재명(국문)"
            ]
            .astype(str)
            .str.contains(
                search_query.strip(),
                case=False,
                na=False,
            )
        ]

    if selected_era != "전체":
        filtered_df = filtered_df[
            filtered_df[
                "시대그룹"
            ] == selected_era
        ]

    if selected_type != "전체":
        filtered_df = filtered_df[
            filtered_df[
                "국가유산종목"
            ] == selected_type
        ]

    # ------------------------------------------------------
    # 결과 건수
    # ------------------------------------------------------
    with f_col4:
        st.markdown(
            """
            <div style="
                font-size:14px;
                font-weight:400;
                color:transparent;
                margin-bottom:6px;
                user-select:none;
            ">
                맞춤 정렬용
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.html(
            f"""
            <div style="
                height:40px;
                display:flex;
                flex-direction:column;
                justify-content:center;
                align-items:center;
                background-color:#ffffff;
                border:1px solid #b8d4fc;
                border-radius:8px;
                box-shadow:0 1px 3px rgba(0,0,0,0.05);
            ">
                <span style="
                    font-size:25px;
                    color:#1d4ed8;
                    font-weight:bold;
                ">
                    {len(filtered_df):,}건
                </span>
            </div>
            """
        )


# ==========================================================
# 필터 결과 없음
# ==========================================================
if filtered_df.empty:
    st.warning(
        "조건에 맞는 국가유산이 없습니다."
    )
    st.stop()


# ==========================================================
# 필터 결과 간단 요약
# ==========================================================
summary1, summary2, summary3 = st.columns(
    3
)

summary1.metric(
    "📍 표시 유산",
    f"{len(filtered_df):,}개",
)

summary2.metric(
    "🕰 시대 그룹",
    f"{filtered_df['시대그룹'].nunique():,}개",
)

summary3.metric(
    "🏷 종목",
    f"{filtered_df['국가유산종목'].nunique():,}종",
)

st.markdown("<br>", unsafe_allow_html=True)


# ==========================================================
# 세션 상태 관리
# ==========================================================
available_names = (
    filtered_df[
        "문화재명(국문)"
    ]
    .astype(str)
    .tolist()
)

if (
    "selected_heritage"
    not in st.session_state
    or
    st.session_state.selected_heritage
    not in available_names
):
    st.session_state.selected_heritage = (
        filtered_df
        .iloc[0][
            "문화재명(국문)"
        ]
    )


# 현재 선택된 문화유산 정보
selected_rows = filtered_df[
    filtered_df[
        "문화재명(국문)"
    ]
    == st.session_state.selected_heritage
]

if selected_rows.empty:
    selected_row = (
        filtered_df
        .iloc[0]
    )

    st.session_state.selected_heritage = (
        selected_row[
            "문화재명(국문)"
        ]
    )

else:
    selected_row = (
        selected_rows
        .iloc[0]
    )


center_lat = float(
    selected_row["위도"]
)

center_lon = float(
    selected_row["경도"]
)


# ==========================================================
# 지도 + 목록 레이아웃
# ==========================================================
map_col, list_col = st.columns(
    [
        3.3,
        1.2,
    ]
)


# ==========================================================
# 지도
# ==========================================================
with map_col:

    st.subheader(
        "🗺️ 국가유산 위치 및 공간 분포"
    )

    st.caption(
        "파란색 마커는 일반 유산, 빨간색 마커는 현재 선택한 유산입니다. "
        "HeatMap은 국가유산의 공간적 밀집도를 나타냅니다."
    )

    # ------------------------------------------------------
    # 기본 지도
    # ------------------------------------------------------
    m = folium.Map(
        location=[
            center_lat,
            center_lon,
        ],
        zoom_start=15,
        control_scale=True,
    )

    # ------------------------------------------------------
    # 마커 클러스터
    # ------------------------------------------------------
    marker_cluster = MarkerCluster(
        name="국가유산 마커"
    ).add_to(m)

    # ------------------------------------------------------
    # HeatMap
    # 단순 위치 밀집도이며 위험도를 의미하지 않음
    # ------------------------------------------------------
    heat_data = (
        filtered_df[
            [
                "위도",
                "경도",
            ]
        ]
        .astype(float)
        .values
        .tolist()
    )

    if heat_data:
        HeatMap(
            heat_data,
            radius=18,
            blur=13,
            min_opacity=0.25,
            name="국가유산 밀집도",
        ).add_to(m)

    # ------------------------------------------------------
    # 마커 생성
    # ------------------------------------------------------
    for _, row in filtered_df.iterrows():

        name = str(
            row["문화재명(국문)"]
        )

        is_selected = (
            name
            == st.session_state.selected_heritage
        )

        # 이미지
        img_url = str(
            row.get(
                "이미지URL",
                "",
            )
        ).strip()

        if (
            img_url
            and img_url.lower()
            not in [
                "nan",
                "none",
                "-",
            ]
        ):
            img_tag = f"""
            <img
                src="{img_url}"
                style="
                    width:100%;
                    height:180px;
                    object-fit:cover;
                    border-radius:8px;
                "
                onerror="
                    this.style.display='none';
                    this.nextElementSibling.style.display='flex';
                "
            >
            <div style="
                display:none;
                width:100%;
                height:150px;
                background:#eeeeee;
                border-radius:8px;
                justify-content:center;
                align-items:center;
                color:#6b7280;
            ">
                이미지 없음
            </div>
            """

        else:
            img_tag = """
            <div style="
                width:100%;
                height:150px;
                background:#eeeeee;
                border-radius:8px;
                display:flex;
                justify-content:center;
                align-items:center;
                color:#6b7280;
            ">
                이미지 없음
            </div>
            """

        era = str(
            row.get(
                "시대그룹",
                "기타",
            )
        )

        heritage_type = str(
            row.get(
                "국가유산종목",
                "미상",
            )
        )

        address = str(
            row.get(
                "소재지상세",
                "-",
            )
        )

        # --------------------------------------------------
        # 팝업 HTML
        # --------------------------------------------------
        popup_content = f"""
        <div style="
            width:270px;
            font-family:
                Pretendard,
                Arial,
                sans-serif;
        ">
            <h4 style="
                margin:0 0 10px 0;
                font-size:16px;
                line-height:1.35;
            ">
                {name}
            </h4>

            {img_tag}

            <div style="
                font-size:12px;
                line-height:1.6;
                margin-top:10px;
                color:#374151;
            ">
                <b>시대:</b> {era}<br>
                <b>종목:</b> {heritage_type}<br>
                <b>주소:</b> {address}
            </div>
        </div>
        """

        # --------------------------------------------------
        # 선택 유산 빨간색 / 일반 유산 파란색
        # --------------------------------------------------
        folium.Marker(
            location=[
                float(row["위도"]),
                float(row["경도"]),
            ],
            popup=folium.Popup(
                popup_content,
                max_width=310,
            ),
            tooltip=name,
            icon=folium.Icon(
                color=(
                    "red"
                    if is_selected
                    else "blue"
                ),
                icon="info-sign",
            ),
        ).add_to(
            marker_cluster
        )

    # ------------------------------------------------------
    # 선택 유산 주변 원 표시
    # ------------------------------------------------------
    folium.Circle(
        location=[
            center_lat,
            center_lon,
        ],
        radius=120,
        color="#ef4444",
        weight=2,
        fill=True,
        fill_color="#ef4444",
        fill_opacity=0.08,
        tooltip=(
            f"선택 유산: "
            f"{st.session_state.selected_heritage}"
        ),
    ).add_to(m)

    # ------------------------------------------------------
    # 레이어 컨트롤
    # ------------------------------------------------------
    folium.LayerControl(
        collapsed=False
    ).add_to(m)

    # ------------------------------------------------------
    # 지도 출력
    # ------------------------------------------------------
    st_folium(
        m,
        width="100%",
        height=540,
        key="gis_map",
    )


# ==========================================================
# 우측 유산 목록
# ==========================================================
with list_col:

    st.subheader(
        "📋 유산 목록"
    )

    st.caption(
        "목록을 선택하면 해당 유산을 중심으로 지도가 이동합니다."
    )

    # 현재 선택 유산 카드
    st.info(
        f"📍 현재 선택\n\n"
        f"**{st.session_state.selected_heritage}**"
    )

    list_container = st.container(
        height=430
    )

    with list_container:

        for idx, row in filtered_df.iterrows():

            name = str(
                row["문화재명(국문)"]
            )

            addr = str(
                row.get(
                    "소재지상세",
                    "-",
                )
            )

            is_selected = (
                name
                == st.session_state.selected_heritage
            )

            btn_label = (
                f"🚩 {name}"
                if is_selected
                else f"🏛️ {name}"
            )

            if st.button(
                btn_label,
                key=f"list_btn_{idx}",
                use_container_width=True,
                type=(
                    "primary"
                    if is_selected
                    else "secondary"
                ),
            ):
                st.session_state.selected_heritage = (
                    name
                )

                st.rerun()

            st.caption(
                f"📍 {addr}"
            )


# ==========================================================
# 선택 유산 상세 정보
# ==========================================================
st.divider()

st.subheader(
    "🔎 선택 국가유산 상세 정보"
)

detail1, detail2, detail3, detail4 = st.columns(
    4
)

detail1.metric(
    "🏛️ 유산명",
    str(
        selected_row[
            "문화재명(국문)"
        ]
    ),
)

detail2.metric(
    "🕰 시대",
    str(
        selected_row.get(
            "시대그룹",
            "기타",
        )
    ),
)

detail3.metric(
    "🏷 종목",
    str(
        selected_row.get(
            "국가유산종목",
            "미상",
        )
    ),
)

detail4.metric(
    "📍 좌표",
    (
        f"{center_lat:.5f}, "
        f"{center_lon:.5f}"
    ),
)

st.caption(
    "※ 지도상의 HeatMap은 문화유산 위치의 공간적 밀집도를 표현한 것으로, "
    "환경 취약도나 훼손 위험도를 의미하지 않습니다."
)

st.caption(
    "선화여고 · 영천 헤리티지 AI 탐구단"
)
