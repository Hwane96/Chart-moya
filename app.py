import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta
import scipy.signal as signal
import numpy as np

# -----------------------------------------------------------------------------
# 1. 앱 기본 설정 (UI 초기화) 
# -----------------------------------------------------------------------------
st.set_page_config(page_title="차트 모야", page_icon="📈", layout="wide")

# -----------------------------------------------------------------------------
# 2. 모듈: 데이터 수집 엔진
# -----------------------------------------------------------------------------
def fetch_data(ticker, start_date, end_date, interval):
    """
    yfinance API를 통해 OHLCV 데이터를 수집하는 함수
    """
    try:
        end_date_yf = end_date + timedelta(days=1)
        df = yf.download(ticker, start=start_date, end=end_date_yf, interval=interval)
        return df
    except Exception as e:
        st.error(f"데이터를 불러오는 중 오류가 발생했습니다: {e}")
        return None

# -----------------------------------------------------------------------------
# 3. 모듈: 형태학적 극점(Peaks/Valleys) 탐색 엔진 (Phase 2 핵심)
# -----------------------------------------------------------------------------
def find_extrema(df, distance):
    """
    SciPy 라이브러리를 사용하여 캔들의 의미 있는 고점과 저점을 찾는 수학적 엔진
    """
    # 원본 데이터 훼손 방지를 위해 안전하게 처리 # 고점 찾기 (High 가격 기준) - distance만큼 떨어진 캔들 중 가장 높은 점 식별
    peaks, _ = signal.find_peaks(df['High'].values.flatten(), distance=distance)
    
    # 저점 찾기 (Low 가격 기준) - 최소값을 찾기 위해 배열에 마이너스(-)를 붙여서 find_peaks 적용
    valleys, _ = signal.find_peaks(-df['Low'].values.flatten(), distance=distance)

    # 데이터프레임에 고점/저점 여부를 기록할 새로운 컬럼 추가
    df['Is_Peak'] = False df['Is_Valley'] = False # 찾은 인덱스 위치에 True 값 부여
    df.iloc[peaks, df.columns.get_loc('Is_Peak')] = True df.iloc[valleys, df.columns.get_loc('Is_Valley')] = True

    return df

# -----------------------------------------------------------------------------
# 4. 모듈: 차트 시각화 엔진 (마커 추가)
# -----------------------------------------------------------------------------
def draw_candlestick_chart(df, ticker):
    """
    기존 캔들 차트 위에 찾아낸 고점과 저점 마커를 덧그리는 함수
    """
    fig = go.Figure()

    # 1. 기본 캔들스틱 차트 레이어
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'].values.flatten(),
        high=df['High'].values.flatten(),
        low=df['Low'].values.flatten(),
        close=df['Close'].values.flatten(),
        name=ticker,
        increasing_line_color='#ef4444', # 양봉 (빨강)
        decreasing_line_color='#3b82f6' # 음봉 (파랑)
     ))

     # 2. 고점(Peaks) 마커 레이어 추가 (초록색 역삼각형)
     peak_data = df[df['Is_Peak']]
     fig.add_trace(go.Scatter(
         x=peak_data.index,
         y=peak_data['High'].values.flatten(),
         mode='markers',
         name='스윙 고점 (Peak)',
         marker=dict(symbol='triangle-down', size=12, color='#22c55e', line=dict(width=1, color='white'))
     ))

     # 3. 저점(Valleys) 마커 레이어 추가 (보라색 삼각형)
     valley_data = df[df['Is_Valley']]
     fig.add_trace(go.Scatter(
         x=valley_data.index,
         y=valley_data['Low'].values.flatten(),
         mode='markers',
         name='스윙 저점 (Valley)',
         marker=dict(symbol='triangle-up', size=12, color='#a855f7', line=dict(width=1, color='white'))
     ))

     fig.update_layout(
         title=f"{ticker} 차트 및 극점 분석",
         yaxis_title="가격", xaxis_title="날짜",
         template="plotly_dark",
         xaxis_rangeslider_visible=False,
         margin=dict(l=20, r=20, t=50, b=20),
         legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
     )

     return fig

# -----------------------------------------------------------------------------
# 5. 메인 애플리케이션 로직
# -----------------------------------------------------------------------------
def main():
    st.title("📈 차트 모야 - 형태학적 패턴 분석기")
    st.markdown("수학적 알고리즘(`SciPy`)을 통해 캔들의 의미 있는 **고점과 저점(Local Extrema)**을 정밀하게 추출합니다.")
    st.divider()

    with st.sidebar:
        st.header("📊 데이터 설정")
        ticker = st.text_input("종목 코드 입력", value="BTC-USD")
        interval = st.selectbox("캔들 시간대", options=["1d", "1wk", "1mo", "1h", "15m", "5m"], index=0)
        start_date = st.date_input("시작일", datetime.today() - timedelta(days=180))
        end_date = st.date_input("종료일", datetime.today())

        st.divider()

        st.header("⚙️ 알고리즘 설정")
        # 민감도 슬라이더 추가: 거리가 멀수록 잔파동을 무시하고 큰 패턴만 찾음
        sensitivity = st.slider(
            "탐색 민감도 (캔들 간격)",
            min_value=3,
            max_value=30,
            value=7,
            help="값이 작을수록 자잘한 파동을 모두 잡고, 값이 클수록 굵직한 추세의 고점/저점만 잡습니다."
        )

        st.write("")
        analyze_button = st.button("🚀 분석 시작하기", type="primary", use_container_width=True)

    if analyze_button:
        if start_date >= end_date:
            st.warning("⚠️ 시작일은 종료일보다 이전이어야 합니다. 날짜를 다시 설정해 주세요.")
            return

        with st.spinner("데이터를 수집하고 알고리즘을 연산하는 중입니다..."):
       
            df = fetch_data(ticker, start_date, end_date, interval)

            if df is not None and not df.empty:
                # 2단계 핵심: 극점 탐색 알고리즘 연산
                df = find_extrema(df, distance=sensitivity)

                # 연산 결과 안내
                total_peaks = df['Is_Peak'].sum()
                total_valleys = df['Is_Valley'].sum()
                st.success(f"✅ 연산 완료! 해당 구간에서 총 **{total_peaks}개의 고점**과 **{total_valleys}개의 저점**을 발견했습니다.")

                # 차트 그리기 및 출력
                fig = draw_candlestick_chart(df, ticker)
                st.plotly_chart(fig, use_container_width=True)

            else: st.error("❌ 해당 기간의 데이터를 찾을 수 없거나 종목 코드가 잘못되었습니다.")

if __name__ == "__main__":
    main()