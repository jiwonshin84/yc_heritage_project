import json

import google.generativeai as genai
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components


# =====================================================
# 1. 페이지 및 보안 설정
# =====================================================
st.set_page_config(
    page_title="영천 AI 문화재 해설사",
    page_icon="🤖",
    layout="wide",
)

# Secrets에서 키를 가져와 Gemini 설정
if "GEMINI_API_KEY" in st.secrets:
    genai.configure(
        api_key=st.secrets["GEMINI_API_KEY"]
    )

    model = genai.GenerativeModel(
        "gemini-2.5-flash"
    )

else:
    st.error(
        "API 키가 설정되지 않았습니다. "
        "Streamlit Cloud의 Settings → Secrets에 "
        "GEMINI_API_KEY를 입력해주세요."
    )
    st.stop()


# =====================================================
# 2. 세션 상태(Session State) 초기화
# =====================================================
session_defaults = {
    "last_heritage": "",
    "open_docent": False,
    "open_qa": False,
    "current_user_q": "",
}

for key, default_value in session_defaults.items():
    if key not in st.session_state:
        st.session_state[key] = default_value


# =====================================================
# 3. 공통 헬퍼 함수
# =====================================================
def clean(val):
    """화면 표시용 결측값 정리"""
    if pd.isna(val) or str(val).strip() == "":
        return "-"

    return str(val).strip()


def get_content_text(row):
    """
    Gemini에 전달할 국가유산 원문 설명.
    내용이 없으면 빈 문자열을 반환하여 AI가 임의로 생성하지 않도록 처리.
    """
    raw_content = row.get("내용")

    if pd.isna(raw_content):
        return ""

    text = str(raw_content).strip()

    if text == "" or text.lower() == "nan":
        return ""

    return text


def clear_ai_cache():
    """문화유산 변경 시 이전 AI 결과와 팝업 상태 초기화"""
    for key in [
        "docent_cache",
        "qa_cache",
    ]:
        if key in st.session_state:
            del st.session_state[key]

    st.session_state.open_docent = False
    st.session_state.open_qa = False
    st.session_state.current_user_q = ""


# =====================================================
# 4. 음성 출력용 JavaScript 함수
# =====================================================
def speak_text(text):
    """
    브라우저의 SpeechSynthesis API를 이용하여
    생성된 해설을 한국어로 읽어준다.

    json.dumps를 사용하여 따옴표, 줄바꿈, 특수문자 때문에
    JavaScript 문자열이 깨지는 문제를 방지한다.
    """
    if not text:
        return

    js_text = json.dumps(
        str(text),
        ensure_ascii=False,
    )

    tts_script = f"""
    <script>
        window.speechSynthesis.cancel();

        const msg = new SpeechSynthesisUtterance({js_text});
        msg.lang = "ko-KR";
        msg.rate = 1.0;
        msg.pitch = 1.0;

        window.speechSynthesis.speak(msg);
    </script>
    """

    components.html(
        tts_script,
        height=0,
    )


# =====================================================
# 5. Gemini 호출 함수
# =====================================================
def generate_docent_text(heritage, content_text):
    """
    국가유산 공공데이터의 원문 설명만을 근거로
    3~4문장의 도슨트 해설을 생성한다.
    """
    prompt = f"""
당신은 영천 지역 국가유산을 소개하는 AI 해설사입니다.

아래에 제공된 국가유산 자료만을 근거로 답변하세요.
자료에 없는 역사적 사실, 연도, 인물, 사건을 임의로 추가하거나 추측하지 마세요.
제공된 자료만으로 확인하기 어려운 내용은 만들어내지 마세요.

문화유산명: {heritage}

[국가유산 공공데이터 원문]
{content_text}

위 자료를 바탕으로 일반 방문객이 이해하기 쉽도록
역사적 배경과 주요 특징을 중심으로 3~4문장으로 설명하세요.
과도하게 전문적인 용어는 피하고 자연스러운 한국어로 작성하세요.
""".strip()

    response = model.generate_content(
        prompt
    )

    if not response or not getattr(response, "text", None):
        raise RuntimeError(
            "Gemini에서 해설 결과를 반환하지 않았습니다."
        )

    return response.text.strip()


def generate_qa_text(heritage, user_q, content_text):
    """
    사용자의 질문에 대해 제공된 국가유산 원문 범위 안에서 답변한다.
    """
    prompt = f"""
당신은 영천 지역 국가유산을 소개하는 AI 해설사입니다.

다음 국가유산 공공데이터 원문만을 근거로 사용자의 질문에 답하세요.
자료에 없는 내용을 추측하거나 만들어내지 마세요.
질문의 답을 원문에서 확인할 수 없는 경우에는
'제공된 국가유산 자료에서는 확인하기 어렵습니다.'라고 명확히 답하세요.

문화유산명: {heritage}

[국가유산 공공데이터 원문]
{content_text}

[사용자 질문]
{user_q}

답변은 간결하고 이해하기 쉬운 한국어로 작성하세요.
""".strip()

    response = model.generate_content(
        prompt
    )

    if not response or not getattr(response, "text", None):
        raise RuntimeError(
            "Gemini에서 답변 결과를 반환하지 않았습니다."
        )

    return response.text.strip()


# =====================================================
# 6. 팝업(Dialog) 함수
# =====================================================
@st.dialog(
    "✨ AI 도슨트 해설",
    width="large",
)
def show_docent_dialog(
    heritage,
    content_text,
):
    # 이미 생성된 결과가 없을 때만 API 호출
    if "docent_cache" not in st.session_state:

        with st.spinner(
            "AI 해설사가 원고를 작성하고 있습니다..."
        ):
            try:
                st.session_state.docent_cache = (
                    generate_docent_text(
                        heritage,
                        content_text,
                    )
                )

            except Exception as e:
                st.error(
                    f"AI 해설 생성 중 오류가 발생했습니다: {e}"
                )
                return

    docent_text = (
        st.session_state.docent_cache
    )

    st.info(
        docent_text
    )

    st.caption(
        "※ AI 해설은 국가유산 공공데이터를 기반으로 생성한 보조 설명입니다. "
        "정확한 학술·역사 정보는 국가유산청 공식 자료를 확인하시기 바랍니다."
    )

    btn_col1, btn_col2 = st.columns(
        2
    )

    with btn_col1:
        if st.button(
            "🔊 해설 듣기",
            use_container_width=True,
            key="docent_speak_btn",
        ):
            speak_text(
                docent_text
            )

    with btn_col2:
        if st.button(
            "확인",
            use_container_width=True,
            key="docent_close_btn",
        ):
            if "docent_cache" in st.session_state:
                del st.session_state.docent_cache

            st.session_state.open_docent = False

            st.rerun()


@st.dialog(
    "💬 AI 질문 답변",
    width="large",
)
def show_qa_dialog(
    heritage,
    user_q,
    content_text,
):
    # 이미 생성된 결과가 없을 때만 API 호출
    if "qa_cache" not in st.session_state:

        with st.spinner(
            "답변 생성 중..."
        ):
            try:
                st.session_state.qa_cache = (
                    generate_qa_text(
                        heritage,
                        user_q,
                        content_text,
                    )
                )

            except Exception as e:
                st.error(
                    f"AI 답변 생성 중 오류가 발생했습니다: {e}"
                )
                return

    qa_text = (
        st.session_state.qa_cache
    )

    st.success(
        qa_text
    )

    st.caption(
        "※ AI 답변은 현재 선택한 국가유산의 공공데이터 원문을 바탕으로 생성됩니다."
    )

    if st.button(
        "확인",
        use_container_width=True,
        key="qa_close_btn",
    ):
        if "qa_cache" in st.session_state:
            del st.session_state.qa_cache

        st.session_state.open_qa = False

        st.rerun()


# =====================================================
# 7. 데이터 처리 함수
# =====================================================
@st.cache_data(
    show_spinner=False
)
def load_data():
    file_path = (
        "data/processed/"
        "yc_heritage_detail_enriched.csv"
    )

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

    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
    )

    return df


# =====================================================
# 8. 메인 UI 렌더링
# =====================================================
try:
    df = load_data()

    # -------------------------------------------------
    # 필수 컬럼 확인
    # -------------------------------------------------
    required_columns = [
        "문화재명(국문)",
    ]

    missing_columns = [
        col
        for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        st.error(
            "데이터에 필요한 컬럼이 없습니다: "
            f"{missing_columns}"
        )
        st.stop()

    # -------------------------------------------------
    # 종목 컬럼 결정
    # -------------------------------------------------
    if "종목" in df.columns:
        category_col = "종목"

    elif "국가유산종목" in df.columns:
        category_col = "국가유산종목"

    else:
        st.error(
            "데이터에 '종목' 또는 '국가유산종목' 컬럼이 없습니다."
        )
        st.stop()

    # -------------------------------------------------
    # 기본 문자열 정리
    # -------------------------------------------------
    df["문화재명(국문)"] = (
        df["문화재명(국문)"]
        .fillna("명칭 미상")
        .astype(str)
        .str.strip()
    )

    df[category_col] = (
        df[category_col]
        .fillna("미상")
        .astype(str)
        .str.strip()
    )

    # =================================================
    # 상단 헤더
    # 기존 화면 구조 유지
    # =================================================
    header_col1, header_col2, header_col3 = st.columns(
        [
            1.3,
            0.6,
            2.0,
        ],
        gap="medium",
        vertical_alignment="center",
    )

    with header_col1:
        st.markdown(
            """
            <h2 style="
                margin:0;
                color:#2c3e50;
            ">
                🤖 AI 문화재 해설 가이드
            </h2>
            """,
            unsafe_allow_html=True,
        )

    with header_col2:
        docent_clicked = st.button(
            "✨ AI 도슨트 해설 생성",
            use_container_width=True,
        )

    with header_col3:
        q_in_col, q_btn_col = st.columns(
            [
                3,
                1,
            ],
            gap="small",
        )

        with q_in_col:
            user_q = st.text_input(
                "질문하기",
                placeholder="궁금한 점을 입력하세요",
                label_visibility="collapsed",
            )

        with q_btn_col:
            question_clicked = st.button(
                "질문 전송",
                use_container_width=True,
            )

    st.caption(
        "국가유산 공공데이터의 상세 설명을 기반으로 생성형 AI가 "
        "영천 지역 문화유산을 이해하기 쉽게 해설하고 질문에 답변합니다."
    )

    st.markdown("---")

    # =================================================
    # 문화재 종목 / 문화재 선택
    # =================================================
    col_sel1, col_sel2 = st.columns(
        2,
        gap="medium",
    )

    category_options = (
        sorted(
            df[category_col]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )
    )

    if not category_options:
        st.warning(
            "선택할 수 있는 문화재 종목 데이터가 없습니다."
        )
        st.stop()

    with col_sel1:
        category = st.selectbox(
            "📂 문화재 품목 선택",
            category_options,
        )

    filtered_df = (
        df[
            df[category_col] == category
        ]
        .copy()
    )

    heritage_options = (
        filtered_df[
            "문화재명(국문)"
        ]
        .dropna()
        .astype(str)
        .tolist()
    )

    if not heritage_options:
        st.warning(
            "선택한 종목에 해당하는 문화재가 없습니다."
        )
        st.stop()

    with col_sel2:
        heritage = st.selectbox(
            "🏛 문화재 선택",
            heritage_options,
        )

    # =================================================
    # 문화재 변경 시 이전 AI 결과 초기화
    # =================================================
    if (
        st.session_state.last_heritage
        != heritage
    ):
        st.session_state.last_heritage = (
            heritage
        )

        clear_ai_cache()

    # =================================================
    # 현재 문화재 행 선택
    # =================================================
    matched_rows = filtered_df[
        filtered_df[
            "문화재명(국문)"
        ] == heritage
    ]

    if matched_rows.empty:
        st.error(
            "선택한 문화재 정보를 찾을 수 없습니다."
        )
        st.stop()

    row = (
        matched_rows
        .iloc[0]
    )

    content_text = (
        get_content_text(
            row
        )
    )

    # =================================================
    # AI 도슨트 버튼
    # =================================================
    if docent_clicked:

        if not content_text:
            st.warning(
                "이 문화재는 AI 해설 생성에 사용할 "
                "원문 설명 자료가 없습니다."
            )

        else:
            st.session_state.open_docent = True

    # =================================================
    # AI 질문 버튼
    # =================================================
    if question_clicked:

        if not user_q.strip():
            st.warning(
                "질문을 입력해주세요."
            )

        elif not content_text:
            st.warning(
                "이 문화재는 AI 질문 답변에 사용할 "
                "원문 설명 자료가 없습니다."
            )

        else:
            st.session_state.open_qa = True
            st.session_state.current_user_q = (
                user_q.strip()
            )

            # 새 질문일 때 이전 답변 캐시 제거
            if "qa_cache" in st.session_state:
                del st.session_state.qa_cache

    # =================================================
    # Dialog 실행
    # =================================================
    if st.session_state.open_docent:
        show_docent_dialog(
            heritage,
            content_text,
        )

    if st.session_state.open_qa:
        user_q_val = (
            st.session_state.get(
                "current_user_q",
                "",
            )
        )

        show_qa_dialog(
            heritage,
            user_q_val,
            content_text,
        )

    # =================================================
    # 상세 정보 영역
    # =================================================
    st.markdown(
        "<br>",
        unsafe_allow_html=True,
    )

    left_col, right_col = st.columns(
        2,
        gap="medium",
    )

    # -------------------------------------------------
    # 왼쪽: 문화재 이미지
    # -------------------------------------------------
    with left_col:

        image_url = (
            row.get(
                "이미지URL"
            )
        )

        if (
            pd.notna(image_url)
            and str(image_url).strip() != ""
            and str(image_url).strip().lower()
            not in [
                "nan",
                "none",
                "-",
            ]
        ):
            st.image(
                str(image_url).strip(),
                use_container_width=True,
            )

            st.caption(
                f"출처: 국가유산청 - {heritage}"
            )

        else:
            st.info(
                "🖼 등록된 이미지가 없습니다."
            )

    # -------------------------------------------------
    # 오른쪽: 상세 정보
    # -------------------------------------------------
    with right_col:

        st.markdown(
            f"""
            <h3 style="
                margin-top:0;
                color:#2c3e50;
            ">
                📋 {heritage} 상세 정보
            </h3>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <style>
                .info-table {{
                    width:100%;
                    border-collapse:collapse;
                    margin-top:10px;
                    border:1px solid #f0f0f0;
                }}

                .info-tr {{
                    border-bottom:1px solid #eeeeee;
                }}

                .info-key {{
                    width:25%;
                    padding:12px 10px;
                    font-weight:bold;
                    color:#34495e;
                    background-color:#f8f9fa;
                    font-size:15px;
                }}

                .info-val {{
                    width:75%;
                    padding:12px 15px;
                    color:#2c3e50;
                    font-size:15px;
                    line-height:1.5;
                }}
            </style>

            <table class="info-table">

                <tr class="info-tr">
                    <td class="info-key">
                        종목
                    </td>
                    <td class="info-val">
                        {clean(row.get(category_col))}
                    </td>
                </tr>

                <tr class="info-tr">
                    <td class="info-key">
                        분류
                    </td>
                    <td class="info-val">
                        {clean(row.get('국가유산분류'))}
                        ({clean(row.get('국가유산분류2'))})
                    </td>
                </tr>

                <tr class="info-tr">
                    <td class="info-key">
                        한자명
                    </td>
                    <td class="info-val">
                        {clean(row.get('문화재명(한자)'))}
                    </td>
                </tr>

                <tr class="info-tr">
                    <td class="info-key">
                        시대
                    </td>
                    <td class="info-val">
                        {clean(row.get('시대'))}
                    </td>
                </tr>

                <tr class="info-tr">
                    <td class="info-key">
                        소재지
                    </td>
                    <td class="info-val">
                        {clean(row.get('소재지상세'))}
                    </td>
                </tr>

                <tr class="info-tr">
                    <td class="info-key">
                        소유/관리
                    </td>
                    <td class="info-val">
                        {clean(row.get('소유자'))}
                        /
                        {clean(row.get('관리자'))}
                    </td>
                </tr>

            </table>
            """,
            unsafe_allow_html=True,
        )

        with st.expander(
            "📖 원문 설명 보기",
            expanded=True,
        ):
            if content_text:
                st.write(
                    content_text
                )

            else:
                st.info(
                    "등록된 원문 설명이 없습니다."
                )

    # =================================================
    # 하단 안내
    # =================================================
    st.divider()

    st.caption(
        "※ 생성형 AI의 해설과 답변은 국가유산 공공데이터를 기반으로 생성한 "
        "보조 정보이며, 공식적인 학술·역사 해석을 대체하지 않습니다."
    )

    st.caption(
        "선화여고 · 영천 헤리티지 AI 탐구단"
    )


except Exception as e:
    st.error(
        f"오류가 발생했습니다: {e}"
    )
