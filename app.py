import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta
from scipy.signal import find_peaks
import numpy as np
import io
import google.generativeai as genai
form PIL import Image

# -----------------------------------------------------------------------------
# 1. 앱 기본 설정 (UI 초기화)
# -----------------------------------------------------------------------------
st.set_page_config(page_title="차트 모야", page_icon="📈", layout="wide")

# -----------------------------------------------------------------------------
# 2. 모듈: 데이터 수집 엔진 (캐싱 및 1차 예외처리 적용)
# -----------------------------------------------------------------------------
@st.cache_data(ttl=900) # 15분(900초)마다 캐시 갱신
def fetch_data(ticker, start_date, end_date, interval):
    """
    yfinance API를 통해 OHLCV 데이터를 수집하는 함수
    """
    try:
        end_date_yf = end_date + timedelta(days=1)
        df = yf.download(ticker, start=start_date, end=end_date_yf, interval=interval)
        
        # 주말/휴장일 또는 상장폐지로 인해 빈 데이터가 오는 경우 방어
        if df is None or df.empty:
            return None
        return df
    except Exception:
        # 에러 발생 시 None을 반환하여 메인 로직에서 우아하게(Gracefully) 처리
        return None

# -----------------------------------------------------------------------------
# 3. 모듈: 형태학적 패턴 인식 및 보조지표 엔진
# -----------------------------------------------------------------------------
def get_zigzag_points(df, peaks, valleys):
    """
    독립적으로 찾아낸 고점과 저점을 시간(인덱스) 순으로 정렬하여 
    하나의 지그재그(ZigZag) 파동 배열로 병합하는 함수
    반환 형태: [(index, 'peak', price), (index, 'valley', price), ...]
    """
    zigzag = []
    for p in peaks:
        zigzag.append((p, 'peak', df['High'].iloc[p]))
    for v in valleys:
        zigzag.append((v, 'valley', df['Low'].iloc[v]))
        
    # 인덱스(시간) 기준으로 오름차순 정렬
    zigzag.sort(key=lambda x: x[0])
    
    # 노이즈 필터링: 연속된 고점이나 저점이 나오면 더 극단적인 값만 남김 (향후 19개 패턴 확장을 위한 필수 작업)
    filtered_zigzag = []
    for pt in zigzag:
        if not filtered_zigzag:
            filtered_zigzag.append(pt)
            continue
            
        last_pt = filtered_zigzag[-1]
        if last_pt[1] == pt[1]: # 같은 타입(연속된 고점 or 연속된 저점)일 경우
            if (pt[1] == 'peak' and pt[2] > last_pt[2]) or (pt[1] == 'valley' and pt[2] < last_pt[2]):
                filtered_zigzag[-1] = pt # 더 높은 고점이나 더 낮은 저점으로 교체
        else:
            filtered_zigzag.append(pt)
            
    return filtered_zigzag

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
        if y2 > y1 and y2 > y3 and abs(y1 - y3) / y1 * 100 <= tolerance:
            patterns['hns'].append((p1, p2, p3))
            
    # 4. 역 헤드앤숄더 (저점 3개: 왼쪽어깨 - 머리 - 오른쪽어깨)
    for i in range(len(valleys) - 2):
        v1, v2, v3 = valleys[i], valleys[i+1], valleys[i+2]
        y1, y2, y3 = df['Low'].iloc[v1], df['Low'].iloc[v2], df['Low'].iloc[v3]
        if y2 < y1 and y2 < y3 and abs(y1 - y3) / y1 * 100 <= tolerance:
            patterns['inv_hns'].append((v1, v2, v3))
            
    return patterns

def add_indicators(df):
    """
    차트 분석을 돕기 위한 5대 핵심 보조지표 계산 (SMA, EMA, RSI, 일목균형표, 거래량)
    """
    if df is None or df.empty:
        return df
        
    # 1. SMA (단순이동평균 7, 20, 50, 100일)
    df['SMA_7'] = df['Close'].rolling(window=7).mean()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_100'] = df['Close'].rolling(window=100).mean()
    
    # 2. EMA (지수이동평균 7, 20, 50, 100일)
    df['EMA_7'] = df['Close'].ewm(span=7, adjust=False).mean()
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
    df['EMA_100'] = df['Close'].ewm(span=100, adjust=False).mean()
    
    # 3. RSI (상대강도지수 14일)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # 4. 일목균형표 (Ichimoku Cloud) - 기본 설정값 9, 26, 52
    high_9 = df['High'].rolling(window=9).max()
    low_9 = df['Low'].rolling(window=9).min()
    df['Tenkan'] = (high_9 + low_9) / 2 # 전환선
    
    high_26 = df['High'].rolling(window=26).max()
    low_26 = df['Low'].rolling(window=26).min()
    df['Kijun'] = (high_26 + low_26) / 2 # 기준선
    
    df['Senkou_A'] = ((df['Tenkan'] + df['Kijun']) / 2).shift(26) # 선행스팬1
    
    high_52 = df['High'].rolling(window=52).max()
    low_52 = df['Low'].rolling(window=52).min()
    df['Senkou_B'] = ((high_52 + low_52) / 2).shift(26) # 선행스팬2
    
    return df

# -----------------------------------------------------------------------------
# 4. 모듈: 차트 시각화 엔진 (제미나이용 클린 차트 및 사용자용 메인 차트)
# -----------------------------------------------------------------------------
def generate_clean_chart_image(df):
    """
    제미나이 시각 분석을 위한 노이즈 없는 깔끔한 캔들 차트 캡처본 생성
    """
    fig = go.Figure(data=[go.Candlestick(
        x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
        increasing_line_color='red', decreasing_line_color='blue'
    )])
    
    # 축, 텍스트, 여백 등 시각적 노이즈 완전 제거
    fig.update_layout(
        xaxis=dict(visible=False, rangeslider=dict(visible=False)),
        yaxis=dict(visible=False),
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=0, r=0, t=0, b=0)
    )
    
    # 이미지를 바이트(Bytes)로 변환
    img_bytes = fig.to_image(format="png", width=800, height=600)
    return img_bytes

def draw_candlestick_chart(df, ticker, peaks, valleys, patterns, show_volume, show_sma, show_ema, show_ichimoku, show_rsi):
    fig = go.Figure()
    
    # 1. 거래량 (Volume) - 캔들 뒤(배경) 하단에 깔리게 가장 먼저 그리기
    if show_volume and 'Volume' in df.columns:
        fig.add_trace(go.Bar(
            x=df.index, y=df['Volume'], name='거래량',
            marker=dict(color='rgba(150, 150, 150, 0.3)'), 
            yaxis='y2'
        ))
        
    # 2. 캔들스틱 차트 (메인)
    fig.add_trace(go.Candlestick(
        x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
        name=ticker, increasing_line_color='red', decreasing_line_color='blue'
    ))
    
    # 3. 고점/저점 및 패턴 마커
    fig.add_trace(go.Scatter(x=df.index[peaks], y=df['High'].iloc[peaks], mode='markers', marker=dict(symbol='triangle-down', size=10, color='#00ff00'), name='고점'))
    fig.add_trace(go.Scatter(x=df.index[valleys], y=df['Low'].iloc[valleys], mode='markers', marker=dict(symbol='triangle-up', size=10, color='#ff00ff'), name='저점'))
    
    for v1, v2 in patterns['double_bottom']:
        fig.add_trace(go.Scatter(x=[df.index[v1], df.index[v2]], y=[df['Low'].iloc[v1], df['Low'].iloc[v2]], mode='lines', line=dict(color='orange', width=2, dash='dot'), name='쌍바닥'))
    for p1, p2 in patterns['double_top']:
        fig.add_trace(go.Scatter(x=[df.index[p1], df.index[p2]], y=[df['High'].iloc[p1], df['High'].iloc[p2]], mode='lines', line=dict(color='red', width=2, dash='dot'), name='쌍봉'))
    for p1, p2, p3 in patterns['hns']:
        fig.add_trace(go.Scatter(x=[df.index[p1], df.index[p2], df.index[p3]], y=[df['High'].iloc[p1], df['High'].iloc[p2], df['High'].iloc[p3]], mode='lines+markers', line=dict(color='yellow', width=3), name='헤드앤숄더'))
    for v1, v2, v3 in patterns['inv_hns']:
        fig.add_trace(go.Scatter(x=[df.index[v1], df.index[v2], df.index[v3]], y=[df['Low'].iloc[v1], df['Low'].iloc[v2], df['Low'].iloc[v3]], mode='lines+markers', line=dict(color='cyan', width=3), name='역헤드앤숄더'))
        
    # 4. 선택형 보조지표 레이어 추가
    if show_sma:
        if 'SMA_7' in df.columns: fig.add_trace(go.Scatter(x=df.index, y=df['SMA_7'], mode='lines', line=dict(color='#FFF59D', width=1), name='SMA(7)'))
        if 'SMA_20' in df.columns: fig.add_trace(go.Scatter(x=df.index, y=df['SMA_20'], mode='lines', line=dict(color='#FBC02D', width=1.5), name='SMA(20)'))
        if 'SMA_50' in df.columns: fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], mode='lines', line=dict(color='#F57F17', width=1.5), name='SMA(50)'))
        if 'SMA_100' in df.columns: fig.add_trace(go.Scatter(x=df.index, y=df['SMA_100'], mode='lines', line=dict(color='#E65100', width=2), name='SMA(100)'))
        
    if show_ema:
        if 'EMA_7' in df.columns: fig.add_trace(go.Scatter(x=df.index, y=df['EMA_7'], mode='lines', line=dict(color='#F48FB1', width=1), name='EMA(7)'))
        if 'EMA_20' in df.columns: fig.add_trace(go.Scatter(x=df.index, y=df['EMA_20'], mode='lines', line=dict(color='#E91E63', width=1.5), name='EMA(20)'))
        if 'EMA_50' in df.columns: fig.add_trace(go.Scatter(x=df.index, y=df['EMA_50'], mode='lines', line=dict(color='#AD1457', width=1.5), name='EMA(50)'))
        if 'EMA_100' in df.columns: fig.add_trace(go.Scatter(x=df.index, y=df['EMA_100'], mode='lines', line=dict(color='#880E4F', width=2), name='EMA(100)'))
        
    if show_ichimoku:
        if 'Tenkan' in df.columns: fig.add_trace(go.Scatter(x=df.index, y=df['Tenkan'], mode='lines', line=dict(color='#2962FF', width=1), name='전환선'))
        if 'Kijun' in df.columns: fig.add_trace(go.Scatter(x=df.index, y=df['Kijun'], mode='lines', line=dict(color='#B71C1C', width=1), name='기준선'))
        if 'Senkou_A' in df.columns and 'Senkou_B' in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df['Senkou_A'], mode='lines', line=dict(color='rgba(0,0,0,0)'), showlegend=False))
            fig.add_trace(go.Scatter(x=df.index, y=df['Senkou_B'], mode='lines', line=dict(color='rgba(0,0,0,0)'), fill='tonexty', fillcolor='rgba(38, 166, 154, 0.2)', name='구름대'))
            
    if show_rsi and 'RSI' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], mode='lines', line=dict(color='orange', width=1.5), name='RSI', yaxis='y3'))
        
    # 레이아웃: 다중 축을 이용해 겹치지 않고 보기 좋게 배치
    vol_max = df['Volume'].max() if 'Volume' in df.columns else 1
    fig.update_layout(
        title=f"{ticker} 패턴 분석", yaxis_title="가격", xaxis_title="날짜", template="plotly_dark", 
        xaxis_rangeslider_visible=False, margin=dict(l=20, r=20, t=50, b=20), hovermode='x unified',
        yaxis=dict(title="가격"),
        yaxis2=dict(overlaying='y', side='left', showgrid=False, visible=False, range=[0, vol_max * 4]),
        yaxis3=dict(title="RSI", overlaying='y', side='right', showgrid=False, range=[0, 100])
    )
    return fig

# --- 추가할 코드 (차트 시각화 엔진 아래, 메인 로직 위) ---
def analyze_chart_with_gemini(image_bytes, api_key):
    """
    제미나이 1.5 Flash 모델을 사용하여 차트 이미지를 시각적으로 분석하는 함수
    """
    if not api_key:
        return "⚠️ Google API Key가 입력되지 않았습니다."
        
    try:
        # API 키 설정 및 모델 초기화
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # 바이트 데이터를 PIL 이미지로 변환
        img = Image.open(io.BytesIO(image_bytes))
        
        # 프롬프트 설정 (PM님의 기획에 맞춰 수정 가능)
        prompt = """
        너는 20년 경력의 수석 트레이더야. 
        첨부된 차트 이미지는 시각적 노이즈를 제거한 캔들 차트야. 
        이 차트에서 보이는 캔들스틱 패턴(예: 쌍바닥, 헤드앤숄더, 깃발형, 삼각수렴 등)을 찾아줘.
        또한 현재의 지지선과 저항선을 파악하고, 향후 가격 이동 방향에 대한 단기적인 직관적 의견을 제시해 줘.
        답변은 마크다운 형식으로 보기 쉽게 정리해 줘.
        """
        
        # 제미나이 API 호출 (이미지와 텍스트 동시 전송)
        response = model.generate_content([prompt, img])
        return response.text
        
    except Exception as e:
        return f"❌ 제미나이 분석 중 오류가 발생했습니다: {e}"

# -----------------------------------------------------------------------------
# 5. 메인 애플리케이션 로직
# -----------------------------------------------------------------------------
def main():
    st.title("📈 차트 모야 - 형태학적 패턴 분석기")
    st.markdown("수학적 알고리즘(`SciPy`)을 통해 캔들의 의미 있는 **고점과 저점(Local Extrema)**을 정밀하게 추출합니다.")
    st.divider()
    
    # UI: 사이드바 패널 (입력 폼)
    with st.sidebar:
        st.header("🔑 AI 분석 설정")
        gemini_api_key = st.text_input("Google Gemini API Key", type="password", help="트랙 B(AI 직관 분석)를 위해 필요합니다.")
        st.divider()

        st.header("📊 분석 설정")
        ticker = st.text_input("종목 코드 입력", value="BTC-USD")
        interval = st.selectbox("캔들 시간대", options=["1d", "1wk", "1mo", "1h", "15m", "5m"], index=0)
        start_date = st.date_input("시작일", datetime.today() - timedelta(days=180))
        end_date = st.date_input("종료일", datetime.today())
        
        prominence = st.slider("탐색 민감도", min_value=1.0, max_value=10.0, value=3.0, step=0.5,
                               help="값이 작을수록 자잘한 굴곡도 모두 찾아내고, 클수록 큼직한 파동만 찾습니다.")
        tolerance = st.slider("패턴 오차 허용률 (%)", min_value=0.1, max_value=5.0, value=1.5, step=0.1,
                              help="두 고점/저점의 가격 차이가 이 비율(%) 이내일 때 같은 위치로 간주합니다.")
                              
        st.divider()
        st.subheader("🛠 보조지표 설정")
        show_volume = st.checkbox("거래량 (Volume)", value=True)
        show_sma = st.checkbox("SMA (7, 20, 50, 100일)", value=False)
        show_ema = st.checkbox("EMA (7, 20, 50, 100일)", value=False)
        show_ichimoku = st.checkbox("일목균형표 (Ichimoku Cloud)", value=False)
        show_rsi = st.checkbox("RSI (상대강도지수)", value=False)
        
        st.write("") # 버튼 위 여백
        analyze_button = st.button("🚀 분석 시작하기", type="primary", use_container_width=True)
        
    # 버튼 클릭 시 실행될 로직
    if analyze_button:
        if start_date >= end_date:
            st.warning("⚠️ 시작일은 종료일보다 이전이어야 합니다. 날짜를 다시 설정해 주세요.")
            return
            
        with st.spinner("데이터 수집 및 패턴 분석 중..."):
            # 1단계: 캐싱된 데이터 가져오기
            df = fetch_data(ticker, start_date, end_date, interval)
            
            # 철벽 방어: 데이터 검증
            if df is None or df.empty:
                st.error("❌ 데이터를 불러올 수 없습니다. 종목 코드(Ticker)가 올바른지, 혹은 해당 기간이 휴장일인지 확인해 주세요.")
            else:
                st.success(f"✅ [{ticker}] 데이터를 성공적으로 불러왔습니다!")
                
                # 지표 계산 연산
                df = add_indicators(df)
                
                # --- 백그라운드 클린 차트 생성 ---
                clean_image_bytes = generate_clean_chart_image(df)
                st.session_state['clean_chart'] = clean_image_bytes
                st.toast("✅ 제미나이 시각 분석용 캔들 캡처가 백그라운드에서 완료되었습니다!")
                
                # 2단계: 고점과 저점 찾기 (1차 추출)
                raw_peaks, _ = find_peaks(df['High'], prominence=prominence)
                raw_valleys, _ = find_peaks(-df['Low'], prominence=prominence)
                
                # 지그재그(ZigZag) 파동 배열 생성 및 노이즈 필터링 적용
                zigzag = get_zigzag_points(df, raw_peaks, raw_valleys)
                
                # 정제된 지그재그 배열에서 '진짜' 고점/저점 인덱스만 다시 추출
                peaks = [pt[0] for pt in zigzag if pt[1] == 'peak']
                valleys = [pt[0] for pt in zigzag if pt[1] == 'valley']
                
                # 3단계: 패턴 찾기 (노이즈가 제거된 핵심 파동 위주로 탐색)
                patterns = detect_patterns(df, peaks, valleys, tolerance)                
                
                # 4단계: 차트 그리기
                fig = draw_candlestick_chart(df, ticker, peaks, valleys, patterns,
                                             show_volume, show_sma, show_ema, show_ichimoku, show_rsi)
                
                # Streamlit에 차트 띄우기
                st.plotly_chart(fig, use_container_width=True)
                
                # 패턴 개수 출력
                st.info(f"🔍 발견된 패턴: 쌍바닥 {len(patterns['double_bottom'])}개, 쌍봉 {len(patterns['double_top'])}개, "
                        f"헤드앤숄더 {len(patterns['hns'])}개, 역헤드앤숄더 {len(patterns['inv_hns'])}개")
                
                # --- 여기서부터 새로 추가되는 투트랙 화면 분할 코드 ---
                st.divider()
                col1, col2 = st.columns(2)
                
                # 트랙 A (차가운 이성: 알고리즘)
                with col1:
                    st.subheader("🤖 트랙 A: 알고리즘 분석 (차가운 이성)")
                    st.write("📊 **추출된 주요 파동(ZigZag) 개수:**", len(zigzag), "개")
                    st.success("수학적 형태학 패턴 분석이 완료되었습니다. (현재 메이저 4개 패턴 감지 중, 추가 15개 패턴 업데이트 대기 중)")
                
                # 트랙 B (유연한 직관: Gemini)
                with col2:
                    st.subheader("👁️ 트랙 B: AI 비전 분석 (유연한 직관)")
                    if gemini_api_key:
                        with st.spinner("제미나이가 차트를 노려보는 중입니다... 🕵️‍♂️"):
                            gemini_result = analyze_chart_with_gemini(st.session_state['clean_chart'], gemini_api_key)
                            st.markdown(gemini_result)
                    else:
                        st.info("👈 사이드바에 Gemini API Key를 입력하시면 AI 직관 분석 결과를 볼 수 있습니다.")

# 파이썬 스크립트 실행 진입점
if __name__ == "__main__":
    main()