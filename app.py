import streamlit as st

# 모바일 환경에 맞게 화면 꽉 차게 설정
st.set_page_config(page_title="차트 모야", page_icon="📈", layout="centered")

# 앱 헤더 설정
st.title("🔍 차트 모야")
st.caption("종목과 기간을 선택하면 캔들 패턴을 정밀 분석해 드립니다.")

st.divider()

# 1. 종목 선택 입력창
symbol = st.text_input("분석할 종목명을 입력하세요 (예: 삼성전자, BTC-USD)", "BTC-USD")

# 2. 캔들 시간대 선택
timeframe = st.select_slider(
    "캔들 단위를 선택하세요",
    options=["1분", "5분", "15분", "1시간", "4시간", "1일", "1주"]
)

# 3. 날짜 선택 (가로 2열 배치)
col1, col2 = st.columns(2)
with col1:
    start_date = st.date_input("시작 날짜")
with col2:
    end_date = st.date_input("종료 날짜")

st.divider()

# 4. 분석 시작 버튼
if st.button("🚀 차트 패턴 분석 시작하기", use_container_width=True):
    st.success(f"'{symbol}' 종목의 [{timeframe}] 캔들 패턴 분석을 시작합니다!")
    st.info("결과 화면: 이곳에 실제 차트와 패턴 해석, 확률 표가 들어올 예정입니다.")
