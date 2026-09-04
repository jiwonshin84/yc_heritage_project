import google.generativeai as genai
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

# =====================================================
# 1. 페이지 및 보안 설정
# =====================================================
st.set_page_config(
    page_title="영천 AI 문화재 해설사", page_icon="🤖", layout="wide"
)

# Secrets에서 키를 가져와 Gemini 설정
if "GEMINI_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    model = genai.GenerativeModel("gemini-2.5-flash")
else:
    st.error(
        "API 키가 설정되지 않았습니다. Streamlit Cloud의 Settings -> Secrets에"
        " GEMINI_API_KEY를 입력해주세요."
    )
    st.stop()

# =====================================================
# 2. 세션 상태(Session State) 초기화
# =====================================================
if "last_heritage" not in st.session_state:
    st.session_state.last_heritage = ""
if "open_docent" not in st.session_state:
    st.session_state.open_docent = False
if "open_qa" not in st.session_state:
    st.session_state.open_qa = False


# =====================================================
# 3. 음성 출력용 JavaScript 함수
# =====================================================
def speak_text(text):
    """브라우저의 TTS 엔진을 사용하여 텍스트를 읽어주는 JS 코드를 삽입합니다."""
    js_text = text.replace("'", "\\'").replace("\n", " ")
    tts_script = f"""
        <script>
            var msg = new SpeechSynthesisUtterance('{js_text}');
            msg.lang = 'ko-KR';
            msg.rate = 1.0;
            window.speechSynthesis.speak(msg);
        </script>
    """
    components.html(tts_script, height=0)


# =====================================================
# 4. 팝업(Dialog) 함수 정의 (캐시 적용으로 재호출 방지)
# =====================================================
@st.dialog("✨ AI 도슨트 해설", width="large")
def show_docent_dialog(heritage, content_text):
    # 이미 생성된 결과가 없을 때만 API 호출
    if "docent_cache" not in st.session_state:
        with st.spinner("AI 해설사가 원고를 작성하고 음성을 준비 중입니다..."):
            prompt = (
                f"당신은 영천 문화재 도슨트입니다. '{heritage}'를 역사적 배경과"
                " 특징을 중심으로 친절하게 설명해주세요. 3~4문장 정도로 핵심만"
                f" 말해주세요. 자료: {content_text}"
            )
            response = model.generate_content(prompt)
            st.session_state.docent_cache = response.text

    st.info(st.session_state.docent_cache)
    speak_text(st.session_state.docent_cache)

    # 확인 버튼 클릭 시 캐시를 지우고 팝업 플래그를 꺼서 창을 닫음
    if st.button("확인", use_container_width=True):
        if "docent_cache" in st.session_state:
            del st.session_state.docent_cache
        st.session_state.open_docent = False
        st.rerun()


@st.dialog("💬 AI 질문 답변", width="large")
def show_qa_dialog(heritage, user_q, content_text):
    # 이미 생성된 결과가 없을 때만 API 호출
    if "qa_cache" not in st.session_state:
        with st.spinner("답변 생성 중..."):
            res = model.generate_content(
                f"{heritage} 질문: {user_q}\n내용: {content_text}"
            )
            st.session_state.qa_cache = res.text

    st.success(st.session_state.qa_cache)

    # 확인 버튼 클릭 시 캐시를 지우고 팝업 플래그를 꺼서 창을 닫음
    if st.button("확인", use_container_width=True):
        if "qa_cache" in st.session_state:
            del st.session_state.qa_cache
        st.session_state.open_qa = False
        st.rerun()


# =====================================================
# 5. 데이터 처리 함수
# =====================================================
@st.cache_data
def load_data():
    return pd.read_csv("data/processed/yc_heritage_detail_enriched.csv")


def clean(val):
    if pd.isna(val) or str(val).strip() == "":
        return "-"
    return str(val).strip()


# =====================================================
# 6. 메인 UI 렌더링
# =====================================================
try:
    df = load_data()

    header_col1, header_col2, header_col3 = st.columns(
        [1.3, 0.6, 2.0], gap="medium", vertical_alignment="center"
    )

    with header_col1:
        st.markdown(
            "<h2 style='margin: 0; color: #2c3e50;'>🤖 AI 문화재 해설 가이드</h2>",
            unsafe_allow_html=True,
        )

    with header_col2:
        docent_clicked = st.button(
            "✨ AI 도슨트 해설 생성", use_container_width=True
        )

    with header_col3:
        q_in_col, q_btn_col = st.columns([3, 1], gap="small")
        with q_in_col:
            user_q = st.text_input(
                "질문하기",
                placeholder="궁금한 점을 입력하세요",
                label_visibility="collapsed",
            )
        with q_btn_col:
            question_clicked = st.button("질문 전송", use_container_width=True)

    st.markdown("---")

    category_col = "종목" if "종목" in df.columns else "국가유산종목"
    col_sel1, col_sel2 = st.columns(2, gap="medium")

    with col_sel1:
        category = st.selectbox(
            "📂 문화재 품목 선택", sorted(df[category_col].dropna().unique())
        )

    filtered_df = df[df[category_col] == category]

    with col_sel2:
        heritage = st.selectbox("🏛 문화재 선택", filtered_df["문화재명(국문)"])

    if st.session_state.last_heritage != heritage:
        st.session_state.last_heritage = heritage
        # 문화재가 바뀌면 이전 캐시도 초기화
        if "docent_cache" in st.session_state:
            del st.session_state.docent_cache
        if "qa_cache" in st.session_state:
            del st.session_state.qa_cache

    row = filtered_df[filtered_df["문화재명(국문)"] == heritage].iloc[0]
    content_text = clean(row.get("내용"))

    if docent_clicked:
        st.session_state.open_docent = True

    if question_clicked:
        if user_q:
            st.session_state.open_qa = True
            st.session_state.current_user_q = user_q
        else:
            st.warning("질문을 입력해주세요.")

    if st.session_state.open_docent:
        show_docent_dialog(heritage, content_text)

    if st.session_state.open_qa:
        user_q_val = st.session_state.get("current_user_q", "")
        show_qa_dialog(heritage, user_q_val, content_text)

    st.markdown("<br>", unsafe_allow_html=True)

    left_col, right_col = st.columns(2, gap="medium")

    with left_col:
        image_url = row.get("이미지URL")
        if pd.notna(image_url) and str(image_url).strip() != "":
            st.image(image_url, use_container_width=True)
            st.caption(f"출처: 국가유산청 - {heritage}")
        else:
            st.info("🖼 등록된 이미지가 없습니다.")

    with right_col:
        st.markdown(
            f"<h3 style='margin-top:0; color:#2c3e50;'>📋 {heritage} 상세 정보</h3>",
            unsafe_allow_html=True,
        )
        st.markdown(f"""
            <style>
                .info-table {{ width: 100%; border-collapse: collapse; margin-top: 10px; border: 1px solid #f0f0f0; }}
                .info-tr {{ border-bottom: 1px solid #eeeeee; }}
                .info-key {{ width: 25%; padding: 12px 10px; font-weight: bold; color: #34495e; background-color: #f8f9fa; font-size: 15px; }}
                .info-val {{ width: 75%; padding: 12px 15px; color: #2c3e50; font-size: 15px; line-height: 1.5; }}
            </style>
            <table class="info-table">
                <tr class="info-tr"><td class="info-key">종목</td><td class="info-val">{clean(row.get(category_col))}</td></tr>
                <tr class="info-tr"><td class="info-key">분류</td><td class="info-val">{clean(row.get('국가유산분류'))} ({clean(row.get('국가유산분류2'))})</td></tr>
                <tr class="info-tr"><td class="info-key">한자명</td><td class="info-val">{clean(row.get('문화재명(한자)'))}</td></tr>
                <tr class="info-tr"><td class="info-key">시대</td><td class="info-val">{clean(row.get('시대'))}</td></tr>
                <tr class="info-tr"><td class="info-key">소재지</td><td class="info-val">{clean(row.get('소재지상세'))}</td></tr>
                <tr class="info-tr"><td class="info-key">소유/관리</td><td class="info-val">{clean(row.get('소유자'))} / {clean(row.get('관리자'))}</td></tr>
            </table>
        """, unsafe_allow_html=True)

        with st.expander("📖 원문 설명 보기", expanded=True):
            st.write(content_text)

except Exception as e:
    st.error(f"오류가 발생했습니다: {e}")
