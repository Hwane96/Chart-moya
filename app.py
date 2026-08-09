import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta

# -----------------------------------------------------------------------------
# 1. 앱 기본 설정 (UI 초기화) 
# -----------------------------------------------------------------------------
st.set_page_config(page_title="차트 모야", page_icon="📈", layout="wide") 

# -----------------------------------------------------------------------------
# 2. 모듈: 데이터 수집 엔진
# -----------------------------------------------------------------------------
def fetch_data(ticker, start_date, end_date, interval):
    """
    yfinance API를 통해 OHLCV 데이터를 안전하게 수집하는 함수
    """
    try:
        # yfinance는 종료일 당일 데이터를 제외하는 경향이 있어 하루를 더해줍니다.
        end_date_yf = end_date + timedelta(days=1)
        df = yf.download(ticker, start=start_date, end=end_date_yf, interval=interval)
        return df
    except Exception as e:
        st.error(f"데이터를 불러오는 중 오류가 발생했습니다: {e}")
        return None

# -----------------------------------------------------------------------------
# 3. 모듈: 차트 시각화 엔진
# -----------------------------------------------------------------------------
def draw_candlestick_chart(df, ticker):
    """
    Plotly를 사용하여 인터랙티브 캔들 차트를 렌더링하는 함수
    """
    fig = go.Figure(data=[go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name=ticker,
        increasing_line_color='red', # 한국 주식/코인 시장에 맞춘 양봉(빨강)
        decreasing_line_color='blue' # 한국 주식/코인 시장에 맞춘 음봉(파랑)
    )])

    fig.update_layout(
        title=f"{ticker} 캔들스틱 차트",
        yaxis_title="가격",
        xaxis_title="날짜",
        template="plotly_dark", # 트레이딩 전문 플랫폼 느낌의 다크 테마
        xaxis_rangeslider_visible=False, # 화면을 넓게 쓰기 위해 하단 기본 슬라이더 숨김 
        margin=dict(l=20, r=20, t=50, b=20)
    ) 

    return fig

# -----------------------------------------------------------------------------
# 4. 메인 애플리케이션 로직
# -----------------------------------------------------------------------------
def main():
    # 헤더 섹션 
    st.title("📈 차트 모야 - 형태학적 패턴 분석기")
    st.markdown("보조지표에 의존하지 않고 캔들의 고점과 저점을 분석하여 순수한 **형태학적 패턴(쌍바닥, 헤드앤숄더 등)**을 찾아냅니다.")
    st.divider()

    # UI: 사이드바 패널 (입력 폼)
    with st.sidebar:
        st.header("📊 분석 설정")

        # 종목 코드 입력 (예: 삼성전자 005930.KS, 비트코인 BTC-USD, 애플 AAPL)
        ticker = st.text_input("종목 코드 입력", value="BTC-USD")

        # 시간대 선택 interval = st.selectbox(
                        "캔들 시간대",
                        options=["1d", "1wk", "1mo", "1h", "15m", "5m"],
                        index=0
        )

        # 날짜 선택 (기본값: 최근 6개월)
        start_date = st.date_input("시작일", datetime.today() - timedelta(days=180))
        end_date = st.date_input("종료일", datetime.today())

        st.write("") # 버튼 위 여백 
        #분석 버튼
        analyze_button = st.button("🚀 분석 시작하기", type="primary", use_container_width=True)

    # 버튼 클릭 시 실행될 로직
    if analyze_button:
        # 날짜 유효성 검사
        if start_date >= end_date:
            st.warning("⚠️ 시작일은 종료일보다 이전이어야 합니다. 날짜를 다시 설정해 주세요.")
            return

        with st.spinner("데이터를 수집하고 캔들 차트를 렌더링하는 중입니다..."):
            # 1단계: 데이터 가져오기 (모듈 호출)
            df = fetch_data(ticker, start_date, end_date, interval)

            # 데이터가 정상적으로 수집되었는지 확인
            if df is not None and not df.empty:
                st.success(f"✅ [{ticker}] 데이터를 성공적으로 불러왔습니다!")

                # 2단계: 차트 그리기 (모듈 호출)
                fig = draw_candlestick_chart(df, ticker)

                # Streamlit에 Plotly 차트 띄우기
                st.plotly_chart(fig, use_container_width=True)

            else: 
                 st.error("❌ 해당 기간의 데이터를 찾을 수 없거나 종목 코드가 잘못되었습니다. 다시 확인해 주세요.") 

# 파이썬 스크립트 실행 진입점
if __name__ == "__main__":
    main() 
