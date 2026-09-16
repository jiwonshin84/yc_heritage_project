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
        st.Page("pages/01_실시간 대시보드.py", title="실시간 대시보드", icon="🏠"),
        st.Page("pages/02_프로젝트 개요.py", title="프로젝트 개요", icon="📊"),
    ],
    "문화유산 탐색": [
        st.Page("pages/03_전국문화유산현황.py", title="전국 문화유산 현황", icon="🏛️"),
        st.Page("pages/04_영천 국가유산 공간 정보.py", title="영천 국가유산 공간 정보", icon="🗺️"),     
        st.Page("pages/05_AI 문화재 해설.py", title="AI 문화재 해설", icon="🤖"),     
    ],
    "환경 취약도 분석": [        
        st.Page("pages/06_문화유산 취약도 예측.py", title="문화유산 취약도 예측", icon="💡"),     
        st.Page("pages/07_최근 40일 환경 데이터.py", title="최근 40일 환경 데이터", icon="🔵"),  
        st.Page("pages/08_학습 데이터 수집.py", title="학습 데이터 수집", icon="📥"),
        st.Page("pages/09_환경 취약도 분류 모델 학습.py", title="환경 취약도 분류 모델 학습", icon="🧠"),     
        st.Page("pages/10_모델 성능 및 최적화.py", title="모델 성능 및 최적화", icon="📈"),        
        st.Page("pages/11_2026년 모델 검증.py", title="2026년 모델 검증", icon="🧪"),          
    ],
    "실시간 모니터링": [
        st.Page("pages/12_Pico 실시간 환경 데이터.py", title=" Pico 실시간 환경 데이터", icon="📡"),
        st.Page("pages/13_실시간 환경 위험 진단.py", title="실시간 환경 위험 진단", icon="⚡"),  
        st.Page("pages/14_실측 데이터 기반 취약도 예측.py", title="실측 데이터 기반 취약도 예측", icon="🛡️"),
    ],
    
    "공간 분석": [   
        st.Page("pages/15_문화유산 관리권역 군집 분석.py", title="군집 분석", icon="🗺️"),   
    ]
}

# 네비게이션 실행
pg = st.navigation(pages)
pg.run()
