import html
import json

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from google import genai
from google.genai import types


# =====================================================
# 1. 페이지 및 Gemini 설정
# =====================================================
st.set_page_config(
    page_title="영천 AI 문화재 해설사",
    page_icon="🤖",
    layout="wide",
)

if "GEMINI_API_KEY" not in st.secrets:
    st.error(
        "API 키가 설정되지 않았습니다. "
        "Streamlit Cloud의 Settings → Secrets에 "
        "GEMINI_API_KEY를 입력해주세요."
    )
    st.stop()

GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]

# 필요하면 Streamlit Secrets에
# GEMINI_MODEL = "gemini-2.5-flash"
# 와 같이 지정할 수 있습니다.
GEMINI_MODEL = st.secrets.get(
    "GEMINI_MODEL",
    "gemini-2.5-flash",
)

client = genai.Client(
    api_key=GEMINI_API_KEY
)


# =====================================================
# 2. 세션 상태 초기화
# =====================================================
SESSION_DEFAULTS = {
    "last_heritage": "",
    "open_docent": False,
    "open_qa": False,
    "current_user_q": "",
}

for key, default_value in SESSION_DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = default_value


# =====================================================
# 3. 공통 함수
# =====================================================
def clean(val):
    """화면 표시용 결측값 정리"""
    if pd.isna(val) or str(val).strip() == "":
        return "-"
    return str(val).strip()


def clean_html(val):
    """HTML 표 출력용 문자열 escape"""
    return html.escape(
        clean(val),
        quote=True,
    )


def get_content_text(row):
    """AI에 전달할 국가유산청 원문 설명"""
    raw_content = row.get("내용")

    if pd.isna(raw_content):
        return ""

    text = str(raw_content).strip()

    if (
        not text
        or text.lower() == "nan"
    ):
        return ""

    return text


def build_heritage_context(row, category_col):
    """
    현재 선택 문화유산의 상세 정보를 Gemini에 전달하기 위한
    구조화 텍스트로 만든다.
    """
    fields = [
        ("문화유산명", row.get("문화재명(국문)")),
        ("종목", row.get(category_col)),
        ("분류1", row.get("국가유산분류")),
        ("분류2", row.get("국가유산분류2")),
        ("한자명", row.get("문화재명(한자)")),
        ("시대", row.get("시대")),
        ("소재지", row.get("소재지상세")),
        ("소유자", row.get("소유자")),
        ("관리자", row.get("관리자")),
        ("국가유산청 원문 설명", row.get("내용")),
    ]

    lines = []

    for key, value in fields:
        value_clean = clean(value)

        if value_clean != "-":
            lines.append(
                f"- {key}: {value_clean}"
            )

    return "\n".join(lines)


def clear_ai_cache():
    """문화유산 변경 시 이전 AI 결과와 Dialog 상태 초기화"""
    for key in [
        "docent_cache",
        "qa_local_cache",
        "qa_web_cache",
        "qa_web_sources",
        "qa_search_queries",
    ]:
        if key in st.session_state:
            del st.session_state[key]

    st.session_state.open_docent = False
    st.session_state.open_qa = False
    st.session_state.current_user_q = ""


# =====================================================
# 4. TTS
# =====================================================
def render_auto_tts_with_stop(text):
    """
    AI 도슨트 Dialog가 열리면 브라우저 TTS 자동 재생을 시도하고,
    같은 컴포넌트의 '음성 중지' 버튼으로 확실하게 중지한다.

    Chrome 계열 브라우저에서 speechSynthesis.cancel() 한 번만으로
    긴 음성이 계속 재생되는 경우를 줄이기 위해
    pause → cancel → resume → cancel을 반복 적용한다.
    """
    if not text:
        return

    js_text = json.dumps(
        str(text),
        ensure_ascii=False,
    )

    tts_html = f"""
    <div style="width:100%; font-family:Arial, sans-serif;">
        <button
            id="stopTtsButton"
            style="
                width:100%;
                height:40px;
                border:1px solid #d1d5db;
                border-radius:8px;
                background:#ffffff;
                color:#111827;
                font-size:15px;
                font-weight:600;
                cursor:pointer;
            "
        >
            ⏹ 음성 중지
        </button>

        <script>
            let currentUtterance = null;
            let stoppedByUser = false;
            let hasStarted = false;

            const synth = window.speechSynthesis;
            const stopButton =
                document.getElementById("stopTtsButton");

            function hardStopSpeech() {{
                stoppedByUser = true;

                if (!synth) {{
                    return;
                }}

                try {{
                    synth.pause();
                }} catch (e) {{}}

                try {{
                    synth.cancel();
                }} catch (e) {{}}

                /*
                 * Chrome에서 긴 utterance가 cancel 뒤에도 잠시 이어지는
                 * 경우를 막기 위해 짧은 간격으로 여러 번 취소한다.
                 */
                [0, 50, 150, 300, 600].forEach(function(delay) {{
                    setTimeout(function() {{
                        try {{
                            synth.resume();
                        }} catch (e) {{}}

                        try {{
                            synth.cancel();
                        }} catch (e) {{}}
                    }}, delay);
                }});

                currentUtterance = null;
            }}

            function findKoreanVoice() {{
                if (!synth) {{
                    return null;
                }}

                const voices = synth.getVoices();

                return (
                    voices.find(
                        voice =>
                            voice.lang &&
                            voice.lang.toLowerCase().startsWith("ko")
                    ) || null
                );
            }}

            function startSpeech() {{
                if (
                    hasStarted ||
                    stoppedByUser ||
                    !synth
                ) {{
                    return;
                }}

                hasStarted = true;

                // 이전 음성이 남아 있으면 먼저 정리
                try {{
                    synth.cancel();
                }} catch (e) {{}}

                currentUtterance =
                    new SpeechSynthesisUtterance({js_text});

                currentUtterance.lang = "ko-KR";
                currentUtterance.rate = 1.0;
                currentUtterance.pitch = 1.0;
                currentUtterance.volume = 1.0;

                const koreanVoice = findKoreanVoice();

                if (koreanVoice) {{
                    currentUtterance.voice = koreanVoice;
                }}

                currentUtterance.onstart = function() {{
                    if (!stoppedByUser) {{
                        stopButton.innerText = "⏹ 음성 중지";
                    }}
                }};

                currentUtterance.onend = function() {{
                    if (!stoppedByUser) {{
                        stopButton.innerText = "✓ 음성 재생 완료";
                    }}
                }};

                currentUtterance.onerror = function(event) {{
                    console.error("TTS 오류:", event);

                    if (!stoppedByUser) {{
                        stopButton.innerText = "⏹ 음성 중지";
                    }}
                }};

                synth.speak(
                    currentUtterance
                );
            }}

            stopButton.addEventListener(
                "click",
                function(event) {{
                    event.preventDefault();
                    event.stopPropagation();

                    hardStopSpeech();

                    stopButton.innerText =
                        "✓ 음성 중지됨";
                }}
            );

            /*
             * Dialog/컴포넌트가 사라질 때에도 음성이 백그라운드에서
             * 계속 남지 않도록 추가 정리.
             */
            window.addEventListener(
                "pagehide",
                hardStopSpeech
            );

            window.addEventListener(
                "beforeunload",
                hardStopSpeech
            );

            document.addEventListener(
                "visibilitychange",
                function() {{
                    if (document.hidden) {{
                        hardStopSpeech();
                    }}
                }}
            );

            if (synth) {{
                synth.getVoices();

                synth.onvoiceschanged =
                    function() {{
                        synth.getVoices();
                    }};

                setTimeout(
                    startSpeech,
                    250
                );
            }}
        </script>
    </div>
    """

    components.html(
        tts_html,
        height=48,
    )


# =====================================================
# 5. Gemini 응답 생성
# =====================================================
def generate_docent_text(
    heritage,
    heritage_context,
):
    """
    국가유산청 데이터와 상세정보를 바탕으로
    상세한 AI 도슨트 해설 생성.
    """
    prompt = f"""
당신은 영천 지역 국가유산을 소개하는 전문 AI 도슨트입니다.

아래 제공된 국가유산청 공공데이터와 상세정보를 중심 근거로 사용하세요.
제공된 정보에서 확인되지 않는 구체적 사실을 사실처럼 만들어내지 마세요.
다만 원문의 의미를 일반 방문객이 이해하기 쉽게 풀어서 설명하는 것은 가능합니다.

[대상 국가유산]
{heritage}

[국가유산 상세정보 및 국가유산청 원문]
{heritage_context}

다음 기준으로 상세한 도슨트 해설을 작성하세요.

1. 이 국가유산이 무엇인지 먼저 소개합니다.
2. 시대와 역사적 배경을 설명합니다.
3. 구조, 형태, 특징, 흔적, 보존 상태 등 확인되는 내용을 구체적으로 설명합니다.
4. 방문객이 현장에서 무엇을 눈여겨보면 좋은지도 설명합니다.
5. 이 유산이 지니는 역사적·문화적 의미를 이해하기 쉽게 정리합니다.
6. 7~10문장 정도로 충분히 상세하게 작성합니다.
7. 전문 용어가 나오면 쉬운 표현으로 풀어 설명합니다.
8. 자연스러운 한국어 도슨트 말투로 작성합니다.
""".strip()

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
    )

    if (
        response is None
        or not getattr(
            response,
            "text",
            None,
        )
    ):
        raise RuntimeError(
            "Gemini에서 해설 결과를 반환하지 않았습니다."
        )

    return response.text.strip()


def generate_local_qa(
    heritage,
    user_q,
    heritage_context,
):
    """
    ① 국가유산청 공공데이터/상세정보 범위에서 답변
    """
    prompt = f"""
당신은 영천 지역 국가유산을 설명하는 전문 AI 해설사입니다.

아래 '국가유산 상세정보 및 국가유산청 원문'만을 근거로
사용자의 질문에 상세히 답변하세요.

[문화유산명]
{heritage}

[국가유산 상세정보 및 국가유산청 원문]
{heritage_context}

[사용자 질문]
{user_q}

작성 기준:
1. 질문에 직접 답하세요.
2. 원문과 상세정보에서 확인되는 근거를 구체적으로 설명하세요.
3. 시대, 구조, 특징, 역사적 흔적, 소재지, 관리 정보 등 질문과 관련 있는 내용이 있으면 연결하세요.
4. 4~7문장 정도로 충분히 상세하게 작성하세요.
5. 데이터에서 확인되지 않는 내용은 추측하지 말고
   '국가유산청 제공 데이터에서는 확인되지 않습니다.'라고 명확히 밝히세요.
6. 일반 방문객과 학생이 이해하기 쉬운 한국어로 작성하세요.
""".strip()

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
    )

    if (
        response is None
        or not getattr(
            response,
            "text",
            None,
        )
    ):
        raise RuntimeError(
            "국가유산청 데이터 기반 답변을 생성하지 못했습니다."
        )

    return response.text.strip()


def extract_grounding_sources(response):
    """
    Google Search grounding 메타데이터에서
    검색 출처와 검색어를 추출한다.
    """
    sources = []
    search_queries = []

    try:
        candidate = response.candidates[0]
        metadata = candidate.grounding_metadata

        if metadata is None:
            return sources, search_queries

        queries = getattr(
            metadata,
            "web_search_queries",
            None,
        )

        if queries:
            search_queries = [
                str(q)
                for q in queries
                if str(q).strip()
            ]

        chunks = getattr(
            metadata,
            "grounding_chunks",
            None,
        )

        if chunks:
            seen = set()

            for chunk in chunks:
                web = getattr(
                    chunk,
                    "web",
                    None,
                )

                if web is None:
                    continue

                uri = getattr(
                    web,
                    "uri",
                    None,
                )

                title = getattr(
                    web,
                    "title",
                    None,
                )

                if not uri:
                    continue

                key = str(uri)

                if key in seen:
                    continue

                seen.add(key)

                sources.append(
                    {
                        "title": (
                            str(title).strip()
                            if title
                            else "웹 검색 출처"
                        ),
                        "url": str(uri).strip(),
                    }
                )

    except Exception:
        # 검색 답변 자체는 살리고
        # 메타데이터 추출 실패만 무시
        pass

    return sources, search_queries


def generate_web_qa(
    heritage,
    user_q,
    heritage_context,
):
    """
    ② Google Search grounding을 이용하여
    인터넷 공개자료·공식자료·학술자료를 검색한 확장 답변 생성.
    """
    prompt = f"""
당신은 문화유산 연구를 돕는 전문 AI 조사 해설사입니다.

사용자의 질문에 답하기 위해 Google 검색을 활용하여
공개된 인터넷 자료를 조사하세요.

대상 문화유산: {heritage}

사용자 질문:
{user_q}

참고용 국가유산 기본정보:
{heritage_context}

검색 및 답변 원칙:
1. 국가유산청, 국가유산포털, 지자체, 공공기관, 박물관, 연구기관 등의
   공식 자료를 우선적으로 확인하세요.
2. 가능하면 학술논문, 학술지, 대학·연구기관 논문/보고서,
   RISS·KCI·DBpia에서 검색되는 연구와 관련된 공개 정보를 찾아보세요.
3. 블로그, 카페, 개인 게시물보다 공식·학술 자료의 신뢰도를 높게 두세요.
4. 국가유산청 원문에 없는 내용이라도 검색 자료에서 확인되면 설명할 수 있습니다.
5. 서로 다른 자료의 내용이 충돌하면 한쪽을 사실처럼 단정하지 말고
   자료별 차이가 있음을 설명하세요.
6. 검색으로 확인되지 않는 사실은 추측하지 마세요.
7. 질문에 직접 답한 뒤, 역사적 배경·특징·관련 연구 내용·의미를 연결하여
   약 6~10문장 정도로 상세하게 설명하세요.
8. 논문이나 연구자료를 찾았다면 연구에서 확인되는 내용을
   일반인이 이해하기 쉽게 풀어서 설명하세요.
9. 답변 안에서 '검색 결과에 따르면', '공식 자료에서는',
   '연구자료에서는'처럼 정보의 성격을 구분해서 표현하세요.
10. 최종 답변에 별도의 가짜 URL이나 임의의 참고문헌을 만들어 넣지 마세요.
    실제 출처 링크는 시스템이 grounding metadata에서 별도로 표시합니다.
""".strip()

    config = types.GenerateContentConfig(
        tools=[
            types.Tool(
                google_search=types.GoogleSearch()
            )
        ]
    )

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=config,
    )

    if (
        response is None
        or not getattr(
            response,
            "text",
            None,
        )
    ):
        raise RuntimeError(
            "인터넷 검색 기반 답변을 생성하지 못했습니다."
        )

    sources, search_queries = (
        extract_grounding_sources(
            response
        )
    )

    return (
        response.text.strip(),
        sources,
        search_queries,
    )


# =====================================================
# 6. Dialog
# =====================================================
@st.dialog(
    "✨ AI 도슨트 상세 해설",
    width="large",
)
def show_docent_dialog(
    heritage,
    heritage_context,
):
    if "docent_cache" not in st.session_state:

        with st.spinner(
            "AI 도슨트가 상세 해설을 작성하고 있습니다..."
        ):
            try:
                st.session_state.docent_cache = (
                    generate_docent_text(
                        heritage,
                        heritage_context,
                    )
                )

            except Exception as e:
                st.error(
                    "AI 해설 생성 중 오류가 발생했습니다.\n\n"
                    f"{e}"
                )
                return

    docent_text = (
        st.session_state.docent_cache
    )

    st.info(
        docent_text
    )

    st.caption(
        "※ 도슨트 해설은 국가유산청 공공데이터와 상세정보를 중심으로 "
        "방문객이 이해하기 쉽게 재구성한 보조 설명입니다."
    )

    btn_col1, btn_col2 = st.columns(
        2
    )

    with btn_col1:
        # 해설 생성 후 자동 음성 재생 + 중지 버튼
        render_auto_tts_with_stop(
            docent_text
        )

    with btn_col2:
        if st.button(
            "확인",
            use_container_width=True,
            key="docent_close_btn",
        ):
            if "docent_cache" in st.session_state:
                del st.session_state["docent_cache"]

            st.session_state.open_docent = False
            st.rerun()


@st.dialog(
    "💬 AI 질문 상세 답변",
    width="large",
)
def show_qa_dialog(
    heritage,
    user_q,
    heritage_context,
):
    # -------------------------------------------------
    # ① 국가유산청 데이터 기반 답변
    # -------------------------------------------------
    if "qa_local_cache" not in st.session_state:

        with st.spinner(
            "① 국가유산청 데이터에서 질문 관련 내용을 분석하고 있습니다..."
        ):
            try:
                st.session_state.qa_local_cache = (
                    generate_local_qa(
                        heritage,
                        user_q,
                        heritage_context,
                    )
                )

            except Exception as e:
                st.session_state.qa_local_cache = (
                    "국가유산청 데이터 기반 답변 생성 중 "
                    f"오류가 발생했습니다: {e}"
                )

    # -------------------------------------------------
    # ② 인터넷/학술 검색 기반 답변
    # -------------------------------------------------
    if "qa_web_cache" not in st.session_state:

        with st.spinner(
            "② 인터넷·공식자료·학술자료를 검색하여 추가 답변을 작성하고 있습니다..."
        ):
            try:
                (
                    web_answer,
                    web_sources,
                    search_queries,
                ) = generate_web_qa(
                    heritage,
                    user_q,
                    heritage_context,
                )

                st.session_state.qa_web_cache = (
                    web_answer
                )

                st.session_state.qa_web_sources = (
                    web_sources
                )

                st.session_state.qa_search_queries = (
                    search_queries
                )

            except Exception as e:
                st.session_state.qa_web_cache = (
                    "인터넷·학술자료 검색 기반 답변 생성 중 "
                    f"오류가 발생했습니다: {e}"
                )

                st.session_state.qa_web_sources = []
                st.session_state.qa_search_queries = []

    local_answer = (
        st.session_state.qa_local_cache
    )

    web_answer = (
        st.session_state.qa_web_cache
    )

    sources = (
        st.session_state.get(
            "qa_web_sources",
            [],
        )
    )

    queries = (
        st.session_state.get(
            "qa_search_queries",
            [],
        )
    )

    # -------------------------------------------------
    # 화면 출력
    # -------------------------------------------------
    st.markdown(
        f"**질문:** {user_q}"
    )

    st.divider()

    st.markdown(
        "### ① 국가유산청 공공데이터·상세정보에서 확인되는 답변"
    )

    st.markdown(
        local_answer
    )

    st.caption(
        "위 내용은 현재 웹서비스에 저장된 국가유산청 공공데이터와 "
        "선택 문화유산의 상세정보를 근거로 생성했습니다."
    )

    st.divider()

    st.markdown(
        "### ② 인터넷 검색·공식자료·학술자료를 확장하여 확인한 답변"
    )

    st.markdown(
        web_answer
    )

    if sources:
        st.markdown(
            "#### 🔗 검색 근거 자료"
        )

        for i, source in enumerate(
            sources,
            start=1,
        ):
            title = source.get(
                "title",
                "검색 출처",
            )

            url = source.get(
                "url",
                "",
            )

            if url:
                st.markdown(
                    f"{i}. [{title}]({url})"
                )

    else:
        st.caption(
            "이번 응답에서는 검색 출처 메타데이터가 반환되지 않았습니다. "
            "검색 결과를 확인하지 못한 내용은 사실처럼 단정하지 않도록 설정되어 있습니다."
        )

    if queries:
        with st.expander(
            "🔎 Gemini가 사용한 웹 검색어 보기",
            expanded=False,
        ):
            for q in queries:
                st.write(
                    f"• {q}"
                )

    st.warning(
        "인터넷 검색 결과에는 최신 정보, 공식 자료, 연구자료뿐 아니라 "
        "품질이 서로 다른 웹 문서가 포함될 수 있습니다. "
        "중요한 학술적 판단에는 표시된 원출처를 직접 확인하세요."
    )

    if st.button(
        "확인",
        use_container_width=True,
        key="qa_close_btn",
    ):
        for key in [
            "qa_local_cache",
            "qa_web_cache",
            "qa_web_sources",
            "qa_search_queries",
        ]:
            if key in st.session_state:
                del st.session_state[key]

        st.session_state.open_qa = False
        st.rerun()


# =====================================================
# 7. 데이터 로드
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
# 8. 메인 UI
# =====================================================
try:
    df = load_data()

    # -------------------------------------------------
    # 필수 컬럼 확인
    # -------------------------------------------------
    if "문화재명(국문)" not in df.columns:
        st.error(
            "데이터에 '문화재명(국문)' 컬럼이 없습니다."
        )
        st.stop()

    if "종목" in df.columns:
        category_col = "종목"

    elif "국가유산종목" in df.columns:
        category_col = "국가유산종목"

    else:
        st.error(
            "데이터에 '종목' 또는 '국가유산종목' 컬럼이 없습니다."
        )
        st.stop()

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
        # 질문 입력창과 전송 버튼을 form으로 묶어
        # 버튼 클릭뿐 아니라 Enter 키로도 바로 전송되도록 처리
        with st.form(
            "heritage_question_form",
            clear_on_submit=False,
            border=False,
        ):
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
                    placeholder=(
                        "예: 이 문화유산의 역사적 의미와 "
                        "관련 연구 내용을 알려주세요"
                    ),
                    label_visibility="collapsed",
                )

            with q_btn_col:
                question_clicked = st.form_submit_button(
                    "질문 전송",
                    use_container_width=True,
                )

    st.caption(
        "국가유산청 공공데이터를 기반으로 AI 도슨트 해설을 제공하며, "
        "질문 답변에서는 ① 국가유산청 데이터와 "
        "② 인터넷·공식자료·학술자료 검색 결과를 구분하여 제시합니다."
    )

    st.markdown("---")

    # =================================================
    # 종목 / 문화유산 선택
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
    # 문화유산 변경
    # =================================================
    if (
        st.session_state.last_heritage
        != heritage
    ):
        st.session_state.last_heritage = (
            heritage
        )
        clear_ai_cache()

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

    row = matched_rows.iloc[0]

    content_text = get_content_text(
        row
    )

    heritage_context = (
        build_heritage_context(
            row,
            category_col,
        )
    )

    # =================================================
    # 도슨트 버튼 처리
    # =================================================
    if docent_clicked:

        if not heritage_context:
            st.warning(
                "AI 해설 생성에 사용할 상세정보가 없습니다."
            )

        else:
            st.session_state.open_docent = True
            st.session_state.open_qa = False

    # =================================================
    # 질문 전송 처리
    # =================================================
    if question_clicked:

        if not user_q.strip():
            st.warning(
                "질문을 입력해주세요."
            )

        else:
            new_question = (
                user_q.strip()
            )

            # 질문이 바뀌었을 때 이전 답변 제거
            if (
                st.session_state.get(
                    "current_user_q",
                    "",
                )
                != new_question
            ):
                for key in [
                    "qa_local_cache",
                    "qa_web_cache",
                    "qa_web_sources",
                    "qa_search_queries",
                ]:
                    if key in st.session_state:
                        del st.session_state[key]

            st.session_state.current_user_q = (
                new_question
            )

            st.session_state.open_qa = True
            st.session_state.open_docent = False

    # =================================================
    # Dialog는 한 번에 하나만
    # =================================================
    if st.session_state.open_docent:
        show_docent_dialog(
            heritage,
            heritage_context,
        )

    elif st.session_state.open_qa:
        show_qa_dialog(
            heritage,
            st.session_state.get(
                "current_user_q",
                "",
            ),
            heritage_context,
        )

    # =================================================
    # 상세 정보 화면
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
    # 이미지
    # -------------------------------------------------
    with left_col:
        image_url = row.get(
            "이미지URL"
        )

        if (
            pd.notna(image_url)
            and str(image_url).strip()
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
    # 상세 정보
    # -------------------------------------------------
    with right_col:
        st.markdown(
            f"""
            <h3 style="
                margin-top:0;
                color:#2c3e50;
            ">
                📋 {html.escape(heritage)} 상세 정보
            </h3>
            """,
            unsafe_allow_html=True,
        )

        detail_html = f"""
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
    <td class="info-key">종목</td>
    <td class="info-val">
        {clean_html(row.get(category_col))}
    </td>
</tr>

<tr class="info-tr">
    <td class="info-key">분류</td>
    <td class="info-val">
        {clean_html(row.get('국가유산분류'))}
        ({clean_html(row.get('국가유산분류2'))})
    </td>
</tr>

<tr class="info-tr">
    <td class="info-key">한자명</td>
    <td class="info-val">
        {clean_html(row.get('문화재명(한자)'))}
    </td>
</tr>

<tr class="info-tr">
    <td class="info-key">시대</td>
    <td class="info-val">
        {clean_html(row.get('시대'))}
    </td>
</tr>

<tr class="info-tr">
    <td class="info-key">소재지</td>
    <td class="info-val">
        {clean_html(row.get('소재지상세'))}
    </td>
</tr>

<tr class="info-tr">
    <td class="info-key">소유/관리</td>
    <td class="info-val">
        {clean_html(row.get('소유자'))}
        /
        {clean_html(row.get('관리자'))}
    </td>
</tr>
</table>
"""

        st.html(
            detail_html
        )

        with st.expander(
            "📖 국가유산청 원문 설명 보기",
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
        "※ AI 질문 답변의 ① 영역은 현재 저장된 국가유산청 공공데이터를, "
        "② 영역은 Gemini의 Google Search grounding을 활용한 공개 웹 검색 결과를 "
        "기반으로 생성합니다."
    )

    st.caption(
        "※ 인터넷·학술 검색 결과는 반드시 표시된 원출처를 함께 확인하시기 바랍니다."
    )

    st.caption(
        "선화여고 · 영천 헤리티지 AI 탐구단"
    )


except Exception as e:
    st.error(
        f"오류가 발생했습니다: {e}"
    )
