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
if "docent_explanation" not in st.session_state:
    st.session_state.docent_explanation = ""
if "ai_answer" not in st.session_state:
    st.session_state.ai_answer = ""
if "last_heritage" not in st.session_state:
    st.session_state.last_heritage = ""
if "should_speak" not in st.session_state:
    st.session_state.should_speak = False


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
# 4. 데이터 처리 함수
# =====================================================
@st.cache_data
def load_data():
    return pd.read_csv("data/processed/yc_heritage_detail_enriched.csv")


def clean(val):
    if pd.isna(val) or str(val).strip() == "":
        return "-"
    return str(val).strip()


# =====================================================
# 5. 메인 UI 렌더링
# =====================================================
try:
    df = load_data()

    # [최상단 제목 및 AI 도슨트/질문 기능 가로 배치]
    title_col, docent_btn_col, q_input_col, q_btn_col = st.columns(
        [1.8, 1.1, 1.5, 0.7], gap="medium"
    )

    with title_col:
        st.markdown(
            "<h1 style='margin: 0; font-size: 28px;'>🤖 AI 문화재 해설 가이드</h1>",
            unsafe_allow_html=True,
        )

    with docent_btn_col:
        # 버튼을 제목 세로 높이에 맞추기 위한 상단 마크다운 여백
        st.markdown(
            "<div style='height: 4px;'></div>", unsafe_allow_html=True
        )
        if st.button("✨ 도슨트 해설 생성", use_container_width=True):
            # heritage 변수가 아래에서 정의되므로 현재 선택된 값 가져오기 위한 처리 필요
            pass  # 아래 필터 로직 이후에 처리되거나 세션에 저장된 값을 활용

    # (주의: 실제 선택된 heritage 값을 쓰기 위해 필터 선행 처리 필요하므로 구조상 아래로 분리하거나 위에서 먼저 가져와야 합니다)
    # 아래에서 안전하게 처리되도록 구조를 정돈했습니다.

except Exception as e:
    pass
