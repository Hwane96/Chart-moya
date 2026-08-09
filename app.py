import streamlit as st
import yfinance as yf
import plotly.graph_object as go
from detetime import datetime, timedelta

# -----------------------------------------------------------------------------
# 1. 앱 기본 설정 (UI 초기화)
# -----------------------------------------------------------------------------
st.set_page_config(page_title="차트모야", page_icon="📈", layout="wide")

# -----------------------------------------------------------------------------
# 2. 모듈: 데이터 수집 엔진
# -----------------------------------------------------------------------------
def fetch_data(ticker, start_date, end_date, interval):
    """
    yfinance API를 통해 OHLCV 데이터를 안전하게 수집하는 함수
    """
    try:
        # yfinance는 종료일 당일 데이터를 제외하는 경향이 있어 하루를 더해줍니다
        end_date_yf = end_date + timedelta(days=1)
        df = yf.download(ticker, start=start_date, end=end_date_yf, interval=interval)
        return df
    except Exception as e:
        st.error(f"데이터를 불러오는 중 오류가 발생하였습니다." {e}")
        return None

# -----------------------------------------------------------------------------
# 3. 모듈: 차트 시각화 엔진
# -----------------------------------------------------------------------------
def draw_candlestick_chart(df, ticker):
"""
Plotly를 사용하여 인터랙티브 캔들 차트를 렌더링하는 함수
"""
fig = go.Figure(data=[go.candlestick
