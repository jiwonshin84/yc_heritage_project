import streamlit as st

# 페이지 전역 설정 (모든 페이지에 공통 적용)
st.set_page_config(
    page_title="영천 지역 실시간 환경 및 문화재 모니터링",
    page_icon="🏛",
    layout="wide"
)

# 사이드바 메뉴 및 연결할 실제 파일 경로 설정
pages = {
    "메인": [
        st.Page("pages/1_실시간 대시보드.py", title="실시간 대시보드", icon="🏠"),
        st.Page("pages/2_프로젝트 개요.py", title="프로젝트 개요", icon="📋"),
    ],
    "문화유산": [
        st.Page("pages/3_전국문화유산현황.py", title="전국 문화유산 현황", icon="🤖"),
        #st.Page("pages/4_ai_explain.py", title="영천 국가유산 공간 정", icon="🧠"),
    ]
}

# 네비게이션 실행
pg = st.navigation(pages)
pg.run()
