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
        st.Page("pages/2_프로젝트 개요.py", title="프로젝트 개요", icon="📖"),
    ],
    "문화유산": [
        st.Page("pages/3_전국문화유산현황.py", title="전국 문화유산 현황", icon="🏛"),
        st.Page("pages/4_영천 국가유산 공간 정보.py", title="영천 국가유산 공간 정보", icon="🗺"),     
        st.Page("pages/5_AI 문화재 해설.py", title="AI 문화재 해설", icon="🧠"),     
    ],
    "수집&학습&예측": [
        st.Page("pages/6_학습 데이터 수집.py", title="학습 데이터 수집", icon="🚀"),
        st.Page("pages/7_위험 예측 분류 모델 학습.py", title="위험 예측 분류 모델 학습", icon="🗂️"),     
        st.Page("pages/7_위험 예측 분류 모델 학습 최적화.py", title="위험 예측 분류 모델 학습 최적화", icon="🗂️"),             
        st.Page("pages/8_전일~7일전 데이터.py", title="전일~7일전 데이터", icon="🔍"),  
        st.Page("pages/8_예측.py", title="예측", icon="💡"),     
    ],
    "피지컬": [
        st.Page("pages/9_피코 실시간 데이터.py", title="피코 실시간 데이터", icon="🌦"),
        st.Page("pages/10_테스트.py", title="테스트", icon="🧪"),     
        st.Page("pages/11_군집 분석.py", title="군집 분석", icon="🛡️")
    ]
}

# 네비게이션 실행
pg = st.navigation(pages)
pg.run()
