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
        st.Page("pages/1_dashboard.py", title="실시간 대시보드", icon="🏠"),
        st.Page("pages/2_overview.py", title="프로젝트 개요", icon="📋"),
    ],
    "분석 & 예측": [
        st.Page("pages/3_ml_model.py", title="위험 예측 모델 학습", icon="🤖"),
        st.Page("pages/4_ai_explain.py", title="AI 문화재 해설", icon="🧠"),
    ]
}

# 네비게이션 실행
pg = st.navigation(pages)
pg.run()
