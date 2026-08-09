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
    df['Is_Peak'] = False
    df['Is_Valley'] = False # 찾은 인덱스 위치에 True 값 부여
    df.iloc[peaks, df.columns.get_loc('Is_Peak')] = True
    df.iloc[valleys, df.columns.get_loc('Is_Valley')] = True

    return df

# -----------------------------------------------------------------------------
# 모듈: 형태학적 패턴 인식 엔진 (Phase 3 추가)
# -----------------------------------------------------------------------------
def detect_patterns(df, peaks, valleys, tolerance):
    """
    고점과 저점 인덱스를 바탕으로 다양한 형태학적 패턴을 찾아내는 함수
    """
    patterns = {'double_bottom': [], 'double_top': [], 'hns': [], 'inv_hns': []}
    
    # 1. 쌍바닥 (W)
    for i in range(len(valleys) - 1):
        v1, v2 = valleys[i], valleys[i+1]
        if abs(df['Low'].iloc[v1] - df['Low'].iloc[v2]) / df['Low'].iloc[v1] * 100 <= tolerance:
            patterns['double_bottom'].append((v1, v2))

    # 2. 쌍봉 (M)
    for i in range(len(peaks) - 1):
        p1, p2 = peaks[i], peaks[i+1]
        if abs(df['High'].iloc[p1] - df['High'].iloc[p2]) / df['High'].iloc[p1] * 100 <= tolerance:
            patterns['double_top'].append((p1, p2))
            
    # 3. 헤드앤숄더 (고점 3개: 왼쪽어깨 - 머리 - 오른쪽어깨)
    for i in range(len(peaks) - 2):
        p1, p2, p3 = peaks[i], peaks[i+1], peaks[i+2]
        y1, y2, y3 = df['High'].iloc[p1], df['High'].iloc[p2], df['High'].iloc[p3]
        # 머리(y2)가 양 어깨보다 높고, 양 어깨(y1, y3)의 높이가 오차범위 내로 비슷할 때
        if y2 > y1 and y2 > y3 and abs(y1 - y3) / y1 * 100 <= tolerance:
            patterns['hns'].append((p1, p2, p3))

    # 4. 역 헤드앤숄더 (저점 3개: 왼쪽어깨 - 머리 - 오른쪽어깨)
    for i in range(len(valleys) - 2):
        v1, v2, v3 = valleys[i], valleys[i+1], valleys[i+2]
        y1, y2, y3 = df['Low'].iloc[v1], df['Low'].iloc[v2], df['Low'].iloc[v3]
        # 머리(y2)가 양 어깨보다 낮고, 양 어깨(y1, y3)의 높이가 오차범위 내로 비슷할 때
        if y2 < y1 and y2 < y3 and abs(y1 - y3) / y1 * 100 <= tolerance:
            patterns['inv_hns'].append((v1, v2, v3))
            
    return patterns

# -----------------------------------------------------------------------------
# 4. 모듈: 차트 시각화 엔진 (마커 추가)
# -----------------------------------------------------------------------------
def draw_candlestick_chart(df, ticker, peaks, valleys, patterns):
    fig = go.Figure(data=[go.Candlestick(
        x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
        name=ticker, increasing_line_color='red', decreasing_line_color='blue'
    )])

    # 고점/저점 마커
    fig.add_trace(go.Scatter(x=df.index[peaks], y=df['High'].iloc[peaks], mode='markers',
                             marker=dict(symbol='triangle-down', size=10, color='#00ff00'), name='고점'))
    fig.add_trace(go.Scatter(x=df.index[valleys], y=df['Low'].iloc[valleys], mode='markers',
                             marker=dict(symbol='triangle-up', size=10, color='#ff00ff'), name='저점'))

    # 쌍바닥 / 쌍봉 그리기
    for v1, v2 in patterns['double_bottom']:
        fig.add_trace(go.Scatter(x=[df.index[v1], df.index[v2]], y=[df['Low'].iloc[v1], df['Low'].iloc[v2]], mode='lines', line=dict(color='orange',          width=2, dash='dot'), name='쌍바닥'))
    for p1, p2 in patterns['double_top']:
        fig.add_trace(go.Scatter(x=[df.index[p1], df.index[p2]], y=[df['High'].iloc[p1], df['High'].iloc[p2]], mode='lines', line=dict(color='red', width=2, dash='dot'), name='쌍봉'))

    # 헤드앤숄더 그리기 (노란색 선)
    for p1, p2, p3 in patterns['hns']:
        fig.add_trace(go.Scatter(
            x=[df.index[p1], df.index[p2], df.index[p3]],
            y=[df['High'].iloc[p1], df['High'].iloc[p2], df['High'].iloc[p3]],
            mode='lines+markers', line=dict(color='yellow', width=3), name='헤드앤숄더'
        ))

    # 역 헤드앤숄더 그리기 (하늘색 선)
    for v1, v2, v3 in patterns['inv_hns']:
        fig.add_trace(go.Scatter(
            x=[df.index[v1], df.index[v2], df.index[v3]],
            y=[df['Low'].iloc[v1], df['Low'].iloc[v2], df['Low'].iloc[v3]],
            mode='lines+markers', line=dict(color='cyan', width=3), name='역헤드앤숄더'
        )) 

    fig.update_layout(title=f"{ticker} 패턴 분석", yaxis_title="가격", xaxis_title="날짜", template="plotly_dark", xaxis_rangeslider_visible=False,                 margin=dict(l=20, r=20, t=50, b=20)) 

    return fig 

# -----------------------------------------------------------------------------
# 5. 메인 애플리케이션 로직
# -----------------------------------------------------------------------------
def main():
    st.title("📈 차트 모야 - 형태학적 패턴 분석기")
    st.markdown("수학적 알고리즘(`SciPy`)을 통해 캔들의 의미 있는 **고점과 저점(Local Extrema)**을 정밀하게 추출합니다.")
    st.divider()

    # UI: 사이드바 패널 (입력 폼)
    with st.sidebar:
        st.header("📊 분석 설정")
        
        # 종목 코드 입력
        ticker = st.text_input("종목 코드 입력", value="BTC-USD")
        
        # 시간대 선택
        interval = st.selectbox(
            "캔들 시간대",
            options=["1d", "1wk", "1mo", "1h", "15m", "5m"],
            index=0
        )
        
        # 날짜 선택
        start_date = st.date_input("시작일", datetime.today() - timedelta(days=180))
        end_date = st.date_input("종료일", datetime.today())

        # (Phase 2 추가) 고점/저점 탐색 민감도 슬라이더
        prominence = st.slider(
            "탐색 민감도", 
            min_value=1.0, max_value=10.0, value=3.0, step=0.5,
            help="값이 작을수록 자잘한 굴곡도 모두 찾아내고, 클수록 큼직한 파동만 찾습니다."
        )

        # (Phase 3 추가) 패턴 오차 허용률 슬라이더
        tolerance = st.slider(
            "패턴 오차 허용률 (%)", 
            min_value=0.1, max_value=5.0, value=1.5, step=0.1,
            help="두 고점/저점의 가격 차이가 이 비율(%) 이내일 때 같은 위치로 간주합니다."
        )

        st.write("") # 버튼 위 여백
        
        # 분석 버튼
        analyze_button = st.button("🚀 분석 시작하기", type="primary", use_container_width=True)

    # 버튼 클릭 시 실행될 로직
    if analyze_button:
        # 날짜 유효성 검사
        if start_date >= end_date:
            st.warning("⚠️ 시작일은 종료일보다 이전이어야 합니다. 날짜를 다시 설정해 주세요.")
            return

        with st.spinner("데이터를 수집하고 패턴을 분석하는 중입니다..."):
            # 1단계: 데이터 가져오기 (모듈 호출)
            df = fetch_data(ticker, start_date, end_date, interval)

            # 데이터가 정상적으로 수집되었는지 확인
            if df is not None and not df.empty:
                st.success(f"✅ [{ticker}] 데이터를 성공적으로 불러왔습니다!")
                
                # 2단계: 고점과 저점 찾기 (Phase 2 추가 부분)
                from scipy.signal import find_peaks
                peaks, _ = find_peaks(df['High'], prominence=prominence)
                valleys, _ = find_peaks(-df['Low'], prominence=prominence)
                
                # 3단계: 패턴 찾기 (Phase 3 추가 부분)
                patterns = detect_patterns(df, peaks, valleys, tolerance)
                
                # 4단계: 차트 그리기 (패턴 데이터 추가 전달)
                fig = draw_candlestick_chart(df, ticker, peaks, valleys, patterns)
                
                # Streamlit에 차트 띄우기
                st.plotly_chart(fig, use_container_width=True)
                
                # 발견된 패턴 개수 화면에 출력 (Phase 4 업데이트)
                st.info(f"🔍 발견된 패턴: 쌍바닥 {len(patterns['double_bottom'])}개, 쌍봉 {len(patterns['double_top'])}개, "
                        f"헤드앤숄더 {len(patterns['hns'])}개, 역헤드앤숄더 {len(patterns['inv_hns'])}개")                
            
            else:
                st.error("❌ 해당 기간의 데이터를 찾을 수 없거나 종목 코드가 잘못되었습니다. 다시 확인해 주세요.")

# 파이썬 스크립트 실행 진입점
if __name__ == "__main__":
    main()