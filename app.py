# [수정/추가할 코드 범위: 1. 앱 기본 설정 직전, import 선언부 맨 아래]

import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta
from scipy.signal import find_peaks
import numpy as np
import io
import google.generativeai as genai
from PIL import Image
import base64
import time
import os
from dotenv import load_dotenv
import ccxt
import pandas as pd

# .env 파일 로드 (보안을 위한 환경 변수 세팅)
load_dotenv()

# 환경 변수에서 API 키 불러오기
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY")
BINANCE_SECRET_KEY = os.getenv("BINANCE_SECRET_KEY")
KOREA_INV_APP_KEY = os.getenv("KOREA_INV_APP_KEY")
KOREA_INV_APP_SECRET = os.getenv("KOREA_INV_APP_SECRET")
# --- [여기까지 새로 추가하는 코드입니다] ---

# -----------------------------------------------------------------------------
# 1. 앱 기본 설정 (UI 초기화)
# -----------------------------------------------------------------------------
st.set_page_config(page_title="차트 모야", page_icon="📈", layout="wide")

# -----------------------------------------------------------------------------
# 1.5. 모듈: 모바일/태블릿(탭 S7 등) UI 반응형 CSS 최적화
# -----------------------------------------------------------------------------
def inject_custom_css():
    st.markdown("""
        <style>
        /* 1. 전체 화면 여백 축소 (작은 화면에서 공간 낭비 방지) */
        .block-container {
            padding-top: 2rem !important;
            padding-bottom: 2rem !important;
            padding-left: 1rem !important;
            padding-right: 1rem !important;
        }
        
        /* 2. 태블릿(세로) 및 모바일 기기에서 컬럼을 강제로 위아래 100% 폭으로 배치 */
        @media (max-width: 992px) {
            [data-testid="column"] {
                width: 100% !important;
                flex: 1 1 100% !important;
                min-width: 100% !important;
                margin-bottom: 1rem;
            }
        }
        
        /* 3. 3줄 브리핑 결과 카드(Success Box)의 텍스트 가독성 및 자간 조정 */
        div[data-testid="stAlert"] {
            font-size: 1.05rem;
            line-height: 1.7;
        }
        </style>
    """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 1.6. 모듈: 커스텀 배경화면 설정
# -----------------------------------------------------------------------------
def set_bg_from_local(image_file):
    """
    로컬 이미지 파일을 읽어서 Streamlit 앱의 전체 배경으로 설정하는 함수
    """
    with open(image_file, "rb") as f:
        encoded_string = base64.b64encode(f.read()).decode()
    
    st.markdown(
        f"""
        <style>
        /* 전체 앱 배경화면 설정 */
        .stApp {{
            background-image: url(data:image/png;base64,{encoded_string});
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        
        /* 내용이 들어가는 메인 컨테이너에 반투명 배경을 깔아 글자 가독성 확보 */
        .block-container {{
            background-color: rgba(0, 0, 0, 0.7) !important;
            border-radius: 15px;
            padding-top: 3rem !important;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# -----------------------------------------------------------------------------
# 2. 모듈: 데이터 수집 엔진 (팩토리 패턴을 통한 다중 소스 분기)
# -----------------------------------------------------------------------------

def fetch_crypto_data_binance(ticker, start_date, end_date, interval):
    """바이낸스 API를 이용한 암호화폐 데이터 수집 (ccxt 활용)"""
    try:
        # yfinance 티커(BTC-USD)를 ccxt 바이낸스 티커(BTC/USDT)로 변환
        symbol = ticker.replace("-USD", "/USDT")
        binance = ccxt.binance()
        
        # 캔들 시간대 맵핑 (yfinance -> ccxt)
        tf_map = {"1d": "1d", "1wk": "1w", "1mo": "1M", "1h": "1h", "15m": "15m", "5m": "5m"}
        timeframe = tf_map.get(interval, "1d")
        
        # 시작일을 타임스탬프로 변환
        since = int(time.mktime(start_date.timetuple()) * 1000)
        
        # 데이터 수집 (API 딜레이 방지를 위해 limit 1000으로 설정)
        ohlcv = binance.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
        if not ohlcv:
            return None
            
        # 데이터를 Pandas DataFrame으로 변환 및 정제
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        return df
    except Exception as e:
        # 에러 발생 시 None을 반환
        return None

def fetch_korean_stock_kis(ticker, start_date, end_date, interval):
    """한국투자증권 API를 이용한 국내 주식 데이터 수집 (뼈대 구축)"""
    # TODO: 1.1에서 세팅한 KOREA_INV_APP_KEY를 활용해 실제 API 통신 로직 구현 예정
    # 현재는 구조만 잡아두고, 임시로 yfinance를 우회하여 데이터를 가져오도록 둡니다.
    try:
        return fetch_us_stock_yf(ticker + ".KS", start_date, end_date, interval)
    except Exception:
        return None

def fetch_us_stock_yf(ticker, start_date, end_date, interval):
    """야후 파이낸스를 이용한 미국 주식 데이터 수집 (기존 로직 유지)"""
    try:
        end_date_yf = end_date + timedelta(days=1)
        df = yf.download(ticker, start=start_date, end=end_date_yf, interval=interval)
        if df is None or df.empty:
            return None
        return df
    except Exception:
        return None

@st.cache_data(ttl=900) # 15분(900초)마다 캐시 갱신
def fetch_data(ticker, start_date, end_date, interval):
    """
    입력된 티커(Ticker)의 형태를 분석하여 적절한 거래소(데이터 소스)로 자동 라우팅하는 팩토리 함수
    """
    # 1. 코인: '-USD'로 끝나는 경우 바이낸스 엔진으로 분기
    if ticker.endswith("-USD") or ticker.endswith("/USDT"):
        return fetch_crypto_data_binance(ticker, start_date, end_date, interval)
        
    # 2. 국장: 6자리 숫자로만 이루어진 경우 (예: 005930) 한국투자증권 엔진으로 분기
    elif ticker.isdigit() and len(ticker) == 6:
        return fetch_korean_stock_kis(ticker, start_date, end_date, interval)
        
    # 3. 미장 등 기타: 야후 파이낸스 엔진(기존)으로 분기
    else:
        return fetch_us_stock_yf(ticker, start_date, end_date, interval)

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
    # 지속형 패턴 3종(박스권, 깃발형, 페넌트) 추가
    patterns = {
        'double_bottom': [], 'double_top': [], 'hns': [], 'inv_hns': [],
        'asc_triangle': [], 'desc_triangle': [], 'sym_triangle': [],
        'rectangle': [], 'flag': [], 'pennant': []
    }
    
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
            
    # 5. 어센딩 트라이앵글 (고점은 수평 저항, 저점은 상승 지지)
    for i in range(len(peaks) - 1):
        for j in range(len(valleys) - 1):
            p1, p2 = peaks[i], peaks[i+1]
            v1, v2 = valleys[j], valleys[j+1]
            # 파동이 시간상으로 겹치는지 대략 확인
            if p1 < v2 and v1 < p2: 
                # 고점은 오차율 이내로 비슷하고, 저점은 오차율 이상으로 확실히 상승할 때
                if abs(df['High'].iloc[p1] - df['High'].iloc[p2]) / df['High'].iloc[p1] * 100 <= tolerance:
                    if df['Low'].iloc[v2] > df['Low'].iloc[v1] * (1 + tolerance/100):
                        patterns['asc_triangle'].append((p1, p2, v1, v2))
                        
    # 6. 디센딩 트라이앵글 (고점은 하락 저항, 저점은 수평 지지)
    for i in range(len(peaks) - 1):
        for j in range(len(valleys) - 1):
            p1, p2 = peaks[i], peaks[i+1]
            v1, v2 = valleys[j], valleys[j+1]
            if p1 < v2 and v1 < p2:
                # 저점은 오차율 이내로 비슷하고, 고점은 확실히 하락할 때
                if abs(df['Low'].iloc[v1] - df['Low'].iloc[v2]) / df['Low'].iloc[v1] * 100 <= tolerance:
                    if df['High'].iloc[p2] < df['High'].iloc[p1] * (1 - tolerance/100):
                        patterns['desc_triangle'].append((p1, p2, v1, v2))
                        
    # 7. 대칭 삼각수렴 (고점은 하락 저항, 저점은 상승 지지)
    for i in range(len(peaks) - 1):
        for j in range(len(valleys) - 1):
            p1, p2 = peaks[i], peaks[i+1]
            v1, v2 = valleys[j], valleys[j+1]
            if p1 < v2 and v1 < p2:
                if df['High'].iloc[p2] < df['High'].iloc[p1] * (1 - tolerance/100):
                    if df['Low'].iloc[v2] > df['Low'].iloc[v1] * (1 + tolerance/100):
                        patterns['sym_triangle'].append((p1, p2, v1, v2))
                        
    # 8. 박스권 (직사각형, Rectangle) - 고점 수평, 저점 수평
    for i in range(len(peaks) - 1):
        for j in range(len(valleys) - 1):
            p1, p2 = peaks[i], peaks[i+1]
            v1, v2 = valleys[j], valleys[j+1]
            if p1 < v2 and v1 < p2:
                if abs(df['High'].iloc[p1] - df['High'].iloc[p2]) / df['High'].iloc[p1] * 100 <= tolerance:
                    if abs(df['Low'].iloc[v1] - df['Low'].iloc[v2]) / df['Low'].iloc[v1] * 100 <= tolerance:
                        patterns['rectangle'].append((p1, p2, v1, v2))
                        
    # 9. 깃발형 (Flag) - 고점과 저점이 같은 방향으로 하락(또는 상승)하는 채널
    for i in range(len(peaks) - 1):
        for j in range(len(valleys) - 1):
            p1, p2 = peaks[i], peaks[i+1]
            v1, v2 = valleys[j], valleys[j+1]
            if p1 < v2 and v1 < p2:
                # 하락 깃발 (우하향 채널 폼)
                if df['High'].iloc[p2] < df['High'].iloc[p1] * (1 - tolerance/100) and df['Low'].iloc[v2] < df['Low'].iloc[v1] * (1 - tolerance/100):
                    patterns['flag'].append((p1, p2, v1, v2))
                    
    # 10. 페넌트 (Pennant) - 좁은 구간에서 빠르게 발생하는 대칭 삼각수렴
    for i in range(len(peaks) - 1):
        for j in range(len(valleys) - 1):
            p1, p2 = peaks[i], peaks[i+1]
            v1, v2 = valleys[j], valleys[j+1]
            if p1 < v2 and v1 < p2:
                if df['High'].iloc[p2] < df['High'].iloc[p1] * (1 - tolerance/100) and df['Low'].iloc[v2] > df['Low'].iloc[v1] * (1 + tolerance/100):
                    # 캔들 간격(기간)이 10봉 이내로 짧은 경우를 페넌트로 분류
                    if abs(p2 - p1) <= 10 and abs(v2 - v1) <= 10:
                        patterns['pennant'].append((p1, p2, v1, v2))
                        
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

def calculate_volume_profile(df, bins=50):
    """
    가격을 n개의 구간(bins)으로 나누어 매물대(Volume Profile)를 계산하는 함수.
    향후 거미줄 매매(Grid trading) 셋업을 위한 주요 지지/저항선 탐색에 활용됩니다.
    """
    if df is None or df.empty or 'Volume' not in df.columns:
        return None
        
    min_price = df['Low'].min()
    max_price = df['High'].max()
    
    # 가격 구간(bin) 생성
    price_bins = np.linspace(min_price, max_price, bins)
    volume_profile = np.zeros(bins - 1)
    
    # 각 캔들을 순회하며 해당 가격 구간에 거래량 분배
    for _, row in df.iterrows():
        # 캔들의 평균 가격(Typical Price)을 기준으로 처리
        typical_price = (row['High'] + row['Low'] + row['Close']) / 3
        bin_idx = np.digitize(typical_price, price_bins) - 1 
        
        # 배열 인덱스 초과 방지
        bin_idx = max(0, min(bin_idx, len(volume_profile) - 1))
        volume_profile[bin_idx] += row['Volume']
        
    # 차트 매핑을 위한 각 구간의 중간 가격 계산
    bin_centers = (price_bins[:-1] + price_bins[1:]) / 2
    
    return pd.DataFrame({
        'Price': bin_centers,
        'Volume': volume_profile
    })

def calculate_grid_targets(df, vp_df):
    """
    매물대(Volume Profile) 데이터를 바탕으로 현재 가격 기준 주요 지지선(매수 타점)과 
    저항선(매도 타점)을 3차까지 계산하여 반환하는 함수
    """
    if df is None or df.empty or vp_df is None or vp_df.empty:
        return None, None
        
    current_price = df['Close'].iloc[-1]
    
    # 거래량이 많은 순으로 정렬하여 핵심 매물대(High Volume Nodes) 추출
    sorted_vp = vp_df.sort_values(by='Volume', ascending=False)
    
    # 현재가보다 낮은 가격대 -> 지지선 (Support)
    supports = sorted_vp[sorted_vp['Price'] < current_price]['Price'].head(3).tolist()
    # 현재가보다 높은 가격대 -> 저항선 (Resistance)
    resistances = sorted_vp[sorted_vp['Price'] > current_price]['Price'].head(3).tolist()
    
    # 데이터가 부족해 타점이 3개가 안 될 경우, 현재가 대비 2% 간격으로 임시 계산하여 채워넣음
    for i in range(3 - len(supports)):
        supports.append(current_price * (1 - 0.02 * (i + 1)))
    for i in range(3 - len(resistances)):
        resistances.append(current_price * (1 + 0.02 * (i + 1)))
        
    # 가격 순으로 정렬 (지지선은 현재가에서 가까운 순서대로 / 저항선도 현재가에서 가까운 순서대로)
    supports.sort(reverse=True)
    resistances.sort()
    
    return supports, resistances

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

def draw_candlestick_chart(df, ticker, peaks, valleys, patterns, show_volume, show_sma, show_ema, show_ichimoku, show_rsi, show_vp=False, vp_df=None):
    fig = go.Figure()
    
    # 1. 거래량 (Volume)
    if show_volume and 'Volume' in df.columns:
        fig.add_trace(go.Bar(
            x=df.index, y=df['Volume'], name='거래량',
            marker=dict(color='rgba(150, 150, 150, 0.3)'), 
            yaxis='y2'
        ))
        
    # 2. 캔들스틱 차트
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
        
    for p1, p2, v1, v2 in patterns['asc_triangle'] + patterns['desc_triangle'] + patterns['sym_triangle']:
        fig.add_trace(go.Scatter(x=[df.index[p1], df.index[p2]], y=[df['High'].iloc[p1], df['High'].iloc[p2]], mode='lines', line=dict(color='#E040FB', width=2), name='상단 추세선'))
        fig.add_trace(go.Scatter(x=[df.index[v1], df.index[v2]], y=[df['Low'].iloc[v1], df['Low'].iloc[v2]], mode='lines', line=dict(color='#E040FB', width=2), name='하단 추세선'))
        
    for p1, p2, v1, v2 in patterns['rectangle'] + patterns['flag'] + patterns['pennant']:
        fig.add_trace(go.Scatter(x=[df.index[p1], df.index[p2]], y=[df['High'].iloc[p1], df['High'].iloc[p2]], mode='lines', line=dict(color='#69F0AE', width=2, dash='dash'), name='채널 상단'))
        fig.add_trace(go.Scatter(x=[df.index[v1], df.index[v2]], y=[df['Low'].iloc[v1], df['Low'].iloc[v2]], mode='lines', line=dict(color='#69F0AE', width=2, dash='dash'), name='채널 하단'))
        
    # 4. 선택형 보조지표 레이어
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

    # 5. 매물대(Volume Profile) 레이어 추가
    if show_vp and vp_df is not None:
        fig.add_trace(go.Bar(
            x=vp_df['Volume'], 
            y=vp_df['Price'],
            orientation='h',
            name='매물대',
            marker=dict(color='rgba(0, 150, 255, 0.3)', line=dict(color='rgba(0, 150, 255, 0.6)', width=1)),
            xaxis='x2',
            yaxis='y'
        ))
        
    vol_max = df['Volume'].max() if 'Volume' in df.columns else 1
    vp_max = vp_df['Volume'].max() if (show_vp and vp_df is not None) else 1
    
    fig.update_layout(
        title=f"{ticker} 패턴 및 매물대 분석", yaxis_title="가격", xaxis_title="날짜", template="plotly_dark", 
        xaxis_rangeslider_visible=False, margin=dict(l=20, r=20, t=50, b=20), hovermode='x unified',
        yaxis=dict(title="가격"),
        yaxis2=dict(overlaying='y', side='left', showgrid=False, visible=False, range=[0, vol_max * 4]),
        yaxis3=dict(title="RSI", overlaying='y', side='right', showgrid=False, range=[0, 100]),
        # 매물대용 보조 x축 추가
        xaxis2=dict(overlaying='x', side='top', showgrid=False, visible=False, range=[0, vp_max * 4])
    )
    return fig

# --- 추가할 코드 (차트 시각화 엔진 아래, 메인 로직 위) ---
def analyze_chart_with_gemini(image_bytes, api_key, model_name="gemini-1.5-flash"):
    """
    제미나이 모델을 사용하여 차트 이미지를 시각적으로 분석하는 함수
    """
    if not api_key:
        return "⚠️ Google API Key가 입력되지 않았습니다."
    try:
        # API 키 설정 및 모델 초기화
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)
        
        # 바이트 데이터를 PIL 이미지로 변환
        img = Image.open(io.BytesIO(image_bytes))
        
        # 프롬프트 고도화 (입체적 분석 및 실전 트레이딩 셋업 고려)
        prompt = """
        너는 20년 경력의 금융자산(암호화폐, 주식, 선물, 외환) 전문 트레이더야. 
        첨부된 차트 이미지는 시각적 노이즈를 제거한 캔들 차트야. 
        이 차트에서 보이는 캔들스틱 패턴(특히 헤드앤숄더, 쌍바닥, 쌍봉 등)을 형태학적으로 분석해줘.
        또한, 차트의 흐름을 바탕으로 RSI, EMA, SMA, 거래량 같은 주요 모멘텀 지표들이 
        현재 어떤 상태일지 유추해서 함께 설명해 줘. 
        마지막으로 이 차트 상황에서 분할 매매 셋업을 한다면 
        어느 구간에 분할 구간(매수/매도 벽)을 치는 것이 유리할지 주요 지지선과 저항선을 기반으로 단기적인 직관적 의견을 제시해 줘.
        답변은 마크다운 형식으로 가독성 좋게 정리해 줘.
        """
        
        # 제미나이 API 호출 (이미지와 텍스트 동시 전송)
        response = model.generate_content([prompt, img])
        return response.text
        
    # API Rate Limit 또는 기타 인증 오류 방어 로직 강화
    except Exception as e:
        error_msg = str(e)
        if "429" in error_msg or "quota" in error_msg.lower():
            return "⚠️ **API 요청 한도 초과(Rate Limit):** 너무 많은 요청이 발생했습니다. 잠시 후 다시 시도해 주세요."
        elif "API_KEY_INVALID" in error_msg or "key" in error_msg.lower():
            return "❌ **잘못된 API Key:** Google Gemini API Key를 다시 확인해 주세요."
        else:
            return f"❌ **제미나이 분석 중 알 수 없는 오류 발생:**\n`{error_msg}`"

# -----------------------------------------------------------------------------
# 4.5. 모듈: 교차 검증 및 브리핑 생성 엔진 (Phase 3)
# -----------------------------------------------------------------------------
def generate_mobile_briefing(api_key, math_patterns_found, gemini_has_pattern, confidence, model_name="gemini-1.5-flash"):
    """
    팩트 데이터(수학+시각 교차 검증 결과)를 바탕으로 3줄 요약 브리핑을 생성하는 함수
    """
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)
        
        prompt = f"""
        너는 모바일 기기(태블릿/스마트폰) 환경에 맞춰 주식/암호화폐 차트 분석 결과를 '딱 3줄'로 브리핑해주는 수석 트레이더야.
        
        [종합된 팩트 데이터]
        - 알고리즘 탐지 패턴 (트랙 A): {', '.join(math_patterns_found) if math_patterns_found else '없음'}
        - AI 비전 패턴 인식 여부 (트랙 B): {'인식됨' if gemini_has_pattern else '인식 안 됨'}
        - 시스템 교차 검증 신뢰도: {confidence}
        
        위 팩트 데이터를 바탕으로 다음 조건에 맞춰 3줄짜리 한국어 브리핑을 작성해.
        1. 첫 번째 줄: 현재 차트에서 두드러지는 핵심 패턴(헤드앤숄더, 쌍바닥 등)과 전체적인 추세 방향 요약
        2. 두 번째 줄: 차트 흐름을 통해 유추해 본 5대 핵심 보조지표(RSI, 일목균형표, EMA, SMA, 거래량)의 현재 상태 추정
        3. 세 번째 줄: 주요 지지선과 저항선을 기반으로, 3분할 매매를 진행할 경우 가장 유리한 1차, 2차, 3차 진입 및 청산 예상 가격대 제시
        
        인사말이나 부연 설명 절대 없이, 딱 1. 2. 3. 넘버링만 해서 간결하고 엣지있게 출력해.
        """
     
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"⚠️ 브리핑 생성 중 오류 발생: {str(e)}"

# -----------------------------------------------------------------------------
# 4.8. 모듈: 수익화(BM) 및 광고 배너 엔진 (Phase 4)
# -----------------------------------------------------------------------------
def show_ad_banner():
    """
    무료 유저에게 노출되는 제휴 배너 (바이비트, 트레이딩뷰 등)
    """
    st.markdown("""
        <div style="text-align: center; margin-top: 30px; margin-bottom: 20px; padding: 15px; border: 1px solid #444; border-radius: 8px; background-color: #1e1e1e;">
            <p style="color: #888; font-size: 0.8em; margin-bottom: 5px;">Advertisement</p>
            <a href="https://www.bybit.com/" target="_blank" style="text-decoration: none;">
                <!-- 실제 배너 이미지 URL로 교체 가능 -->
                <img src="https://via.placeholder.com/728x90.png?text=Bybit+Referral+Banner+-+Trade+Crypto" alt="Bybit Ad" style="max-width: 100%; border-radius: 5px;">
            </a>
            <p style="margin-top: 10px; font-size: 0.9em;">
                🚀 <a href="https://www.tradingview.com/" target="_blank" style="color: #4CAF50; text-decoration: none; font-weight: bold;">트레이딩뷰 프리미엄 가입하고 혜택받기</a>
            </p>
        </div>
    """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 4.9. 모듈: 상태(Session) 초기화 콜백
# -----------------------------------------------------------------------------
def reset_analysis_state():
    """
    사용자가 종목 코드, 멤버십 등급, 캔들 시간대 등을 변경했을 때
    기존에 남아있던 AI 분석 잔상과 캐시를 깔끔하게 지워주는 콜백 함수
    """
    keys_to_clear = ['gemini_result_cache', 'briefing_cache', 'clean_chart', 'last_chart_img', 'last_model']
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]

# -----------------------------------------------------------------------------
# 5. 메인 애플리케이션 로직
# -----------------------------------------------------------------------------
def main():
    inject_custom_css() # <-- 기존 코드
    
    # --- [수정/추가할 부분: 배경화면 적용 함수 호출] ---
    try:
        set_bg_from_local("background.png") # 실제 배경 이미지 파일명으로 변경해 주세요!
    except Exception:
        pass # 이미지가 없어도 앱이 멈추지 않도록 예외 처리
        
    st.title("📈 차트 모야 - 형태학적 패턴 분석기") # <-- 기존 코드
    st.markdown("수학적 알고리즘(`SciPy`)을 통해 캔들의 의미 있는 **고점과 저점(Local Extrema)**을 정밀하게 추출합니다.") # <-- 기존 코드
    st.divider()
    
   # UI: 사이드바 패널 (입력 폼)
    with st.sidebar:
        st.header("🔑 AI 분석 설정")
        gemini_api_key = st.text_input("Google Gemini API Key", type="password", help="트랙 B(AI 직관 분석)를 위해 필요합니다.")
        
        # --- [Phase 4: 유저 등급 선택 UI 추가] ---
        st.divider()
        st.header("💎 멤버십 설정")
        user_tier = st.radio("유저 등급 선택 (가상)", ["Free (광고O, Flash모델)", "Premium (광고X, Pro모델)"], on_change=reset_analysis_state)
        # 등급에 따른 모델 동적 할당
        selected_model = "gemini-1.5-pro" if "Premium" in user_tier else "gemini-1.5-flash"
        
        st.divider()
        st.header("📊 분석 설정")
        ticker = st.text_input("종목 코드 입력", value="BTC-USD", on_change=reset_analysis_state)
        interval = st.selectbox("캔들 시간대", options=["1d", "1wk", "1mo", "1h", "15m", "5m"], index=0, on_change=reset_analysis_state)
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
        show_vp = st.checkbox("매물대 (Volume Profile)", value=True)
        
        st.write("") # 버튼 위 여백
        analyze_button = st.button("🚀 분석 시작하기", type="primary", use_container_width=True)
        
    # 버튼 클릭 시 실행될 로직
    if analyze_button:
        if start_date >= end_date:
            st.warning("⚠️ 시작일은 종료일보다 이전이어야 합니다. 날짜를 다시 설정해 주세요.")
            return
            
        with st.spinner("데이터 수집 및 패턴 분석 중... 📊"):
            try:
                # 1단계: 캐싱된 데이터 가져오기
                df = fetch_data(ticker, start_date, end_date, interval)
                
                # 철벽 방어 1: 데이터가 없거나 서버 오류일 때
                if df is None or df.empty:
                    st.error("❌ 데이터를 불러올 수 없습니다. 종목 코드(Ticker)가 올바른지, 혹은 거래소 서버 상태를 확인해 주세요.")
                # 철벽 방어 2: 데이터가 너무 적어 알고리즘이 터지는 현상 방지 (최소 50봉 요구)
                elif len(df) < 50:
                    st.warning(f"⚠️ 분석에 필요한 캔들 데이터가 부족합니다. (현재 {len(df)}개 / 최소 50개 필요)\n\n조회 기간(시작일~종료일)을 더 넓게 설정해 주세요.")
                else:
                    st.success(f"✅ [{ticker}] 데이터를 성공적으로 불러왔습니다! (총 {len(df)}봉)")
                
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
                vp_df = calculate_volume_profile(df) if show_vp else None
                
                # 4단계: 차트 그리기
                fig = draw_candlestick_chart(df, ticker, peaks, valleys, patterns,
                                             show_volume, show_sma, show_ema, show_ichimoku, show_rsi, show_vp, vp_df)
                
                # Streamlit에 차트 띄우기
                st.plotly_chart(fig, use_container_width=True)
                
                # 패턴 개수 출력 (수렴형 및 지속형 패턴 추가)
                st.info(f"🔍 발견된 반전 패턴: 쌍바닥 {len(patterns['double_bottom'])}개, 쌍봉 {len(patterns['double_top'])}개, "
                        f"헤드앤숄더 {len(patterns['hns'])}개, 역헤드앤숄더 {len(patterns['inv_hns'])}개\n\n"
                        f"📐 발견된 수렴 패턴: 어센딩 {len(patterns['asc_triangle'])}개, 디센딩 {len(patterns['desc_triangle'])}개, "
                        f"대칭삼각 {len(patterns['sym_triangle'])}개\n\n"
                        f"🛤️ 발견된 지속 패턴: 박스권 {len(patterns['rectangle'])}개, 깃발형 {len(patterns['flag'])}개, "
                        f"페넌트 {len(patterns['pennant'])}개")
                
                # --- 여기서부터 새로 추가되는 투트랙 화면 분할 코드 ---
                st.divider()
                col1, col2 = st.columns(2)
                
                # 트랙 A (차가운 이성: 알고리즘)
                with col1:
                    st.subheader("🤖 트랙 A: 알고리즘 분석 (차가운 이성)")
                    st.write("📊 **추출된 주요 파동(ZigZag) 개수:**", len(zigzag), "개")
                    # 안내 문구의 숫자(4개 -> 10개, 15개 -> 9개)를 업데이트
                    st.success("수학적 형태학 패턴 분석이 완료되었습니다. (현재 메이저 10개 패턴 감지 중, 추가 9개 패턴 업데이트 대기 중)")
                
               # 트랙 B (유연한 직관: Gemini 또는 패턴 이미지)
                with col2:
                    st.subheader("👁️ 트랙 B: AI 비전 분석 (유연한 직관)")
                    
                    # --- [무료 회원(Free) 로직: 제미나이 대신 로컬 예시 이미지 제공] ---
                    if "Free" in user_tier:
                        st.info("💡 **무료 버전 제공:** 현재 차트에서 감지된 주요 패턴의 표준 예시입니다.")
                        
                        # 감지된 패턴 우선순위에 따라 1개의 대표 이미지를 출력
                        try:
                            if len(patterns['hns']) > 0:
                                st.image("assets/hns.png", caption="📖 감지된 패턴: 헤드앤숄더 (하락 반전)", use_column_width=True)
                            elif len(patterns['inv_hns']) > 0:
                                st.image("assets/inv_hns.png", caption="📖 감지된 패턴: 역헤드앤숄더 (상승 반전)", use_column_width=True)
                            elif len(patterns['double_top']) > 0:
                                st.image("assets/double_top.png", caption="📖 감지된 패턴: 쌍봉 (하락 반전)", use_column_width=True)
                            elif len(patterns['double_bottom']) > 0:
                                st.image("assets/double_bottom.png", caption="📖 감지된 패턴: 쌍바닥 (상승 반전)", use_column_width=True)
                            else:
                                st.warning("현재 명확한 넥라인을 가진 반전 패턴이 감지되지 않았습니다.")
                        except FileNotFoundError:
                            st.error("⚠️ 패턴 예시 이미지가 아직 업로드되지 않았습니다. (추후 assets 폴더에 추가 예정)")
                            
                        st.success("💎 **Premium 멤버십 혜택:** 제미나이 Pro AI의 정밀 차트 분석 리포트와 분할 매매(거미줄 매매) 타점 추천을 받아보세요!")
                        gemini_result = None # 무료 회원은 Phase 3 브리핑 생성을 위해 None으로 처리
                        
                    # --- [유료 회원(Premium) 로직: 기존 제미나이 AI 시각 분석 수행] ---
                    else:
                        if gemini_api_key:
                            is_chart_changed = ('last_chart_img' not in st.session_state) or (st.session_state['last_chart_img'] != st.session_state['clean_chart'])
                            is_model_changed = ('last_model' not in st.session_state) or (st.session_state['last_model'] != selected_model)
                            
                            if is_chart_changed or is_model_changed:
                                with st.spinner(f"제미나이({selected_model})가 차트를 노려보는 중입니다... 🕵️‍♂️"):
                                    gemini_result = analyze_chart_with_gemini(st.session_state['clean_chart'], gemini_api_key, selected_model)
                                    st.session_state['gemini_result_cache'] = gemini_result
                                    st.session_state['last_chart_img'] = st.session_state['clean_chart']
                                    st.session_state['last_model'] = selected_model 
                                    st.session_state['briefing_cache'] = None 
                            else:
                                gemini_result = st.session_state['gemini_result_cache']
                                st.toast("⚡ 기존 AI 분석 결과를 즉시 불러왔습니다! (API 호출 스킵)")
                            st.markdown(gemini_result)
                        else:
                            st.info("👈 사이드바에 Gemini API Key를 입력하시면 AI 직관 분석 결과를 볼 수 있습니다.")
                            gemini_result = None
                        
                # --- Phase 3: 교차 검증 및 브리핑 로직 ---
                if gemini_api_key and gemini_result:
                    st.divider()
                    st.header("🤝 Phase 3: 수학/AI 교차 검증 및 최종 브리핑")
                    with st.container():
                        # 1. 파이썬 수학적 탐지 결과 확인
                        math_patterns_found = [k for k, v in patterns.items() if len(v) > 0]
                        math_has_pattern = len(math_patterns_found) > 0
                        
                        # 2. 제미나이 시각적 탐지 결과 확인
                        pattern_keywords = ['쌍바닥', '쌍봉', '헤드앤숄더', '어센딩', '디센딩', '삼각', '박스권', '깃발', '페넌트']
                        gemini_has_pattern = any(keyword in gemini_result for keyword in pattern_keywords)
                        
                        # 3. 교차 검증 신뢰도 판별
                        if math_has_pattern and gemini_has_pattern:
                            confidence = "최상 (수학/AI 교차 검증 일치)"
                            icon = "🟢"
                        elif math_has_pattern or gemini_has_pattern:
                            confidence = "보통 (수학/AI 중 하나만 패턴 감지)"
                            icon = "🟡"
                        else:
                            confidence = "주의 (뚜렷한 패턴 미감지, 횡보/관망 권장)"
                            icon = "🔴"
                            
                        st.subheader(f"{icon} 시스템 신뢰도: {confidence}")
                        
                        # --- [3줄 브리핑 전용 API 중복 호출 방지] ---
                        if ('briefing_cache' not in st.session_state) or (st.session_state['briefing_cache'] is None):
                            with st.spinner("모바일 3줄 브리핑 요약본을 생성하는 중입니다... 📱"):
                                briefing_text = generate_mobile_briefing(
                                    api_key=gemini_api_key, 
                                    math_patterns_found=math_patterns_found, 
                                    gemini_has_pattern=gemini_has_pattern, 
                                    confidence=confidence,
                                    model_name=selected_model # 모델 변경 파라미터 적용
                                )
                                st.session_state['briefing_cache'] = briefing_text
                        else:
                            briefing_text = st.session_state['briefing_cache']
                            
                        # 4. 최종 출력 (중복 버그 제거됨)
                        st.success(f"**[전문가 3줄 브리핑]**\n\n{briefing_text}")
                        
                        # --- [여기서부터 추가되는 Premium 유저 전용 로직: 분할 매매(거미줄) 타점 추천] ---
                        if "Premium" in user_tier:
                            st.divider()
                            st.subheader("🎯 Premium 전용: 매물대 기반 거미줄 타점 셋업")
                            st.markdown("가장 두터운 매물대(Volume Profile)를 기준으로 산출된 1~3차 진입 및 청산 목표가입니다.")
                            
                            supports, resistances = calculate_grid_targets(df, vp_df if 'vp_df' in locals() else None)
                            
                            if supports and resistances:
                                current_p = df['Close'].iloc[-1]
                                st.metric("현재가 (Current Price)", f"{current_p:,.2f}")
                                
                                target_col1, target_col2 = st.columns(2)
                                with target_col1:
                                    st.info(f"🔽 **분할 매수 (지지선)**\n\n"
                                            f"**1차 진입:** {supports[0]:,.2f}\n\n"
                                            f"**2차 진입:** {supports[1]:,.2f}\n\n"
                                            f"**3차 진입:** {supports[2]:,.2f}")
                                with target_col2:
                                    st.error(f"🔼 **분할 매도 (저항선)**\n\n"
                                             f"**1차 목표:** {resistances[0]:,.2f}\n\n"
                                             f"**2차 목표:** {resistances[1]:,.2f}\n\n"
                                             f"**3차 목표:** {resistances[2]:,.2f}")
                        # --- [여기까지 Premium 전용 로직] ---

                # 👇 [수정된 부분]: 들여쓰기를 왼쪽으로 쫙 뺐어!
                # --- [Phase 4: 수익화(BM) 로직 - 무료 유저만 광고 노출] ---
                if "Free" in user_tier:
                    st.divider()
                    show_ad_banner()
                            
            except Exception as e:
                # 철벽 방어 3: 앱 크래시 방지 및 우아한 에러 메시지 출력
                st.error("🚨 분석 처리 중 예상치 못한 오류가 발생했습니다. 잠시 후 다시 시도해 주세요.")

# 파이썬 스크립트 실행 진입점
if __name__ == "__main__":
    # 앱이 처음 실행되어 'splash_shown' 상태가 없을 때만 스플래시 동작
    if 'splash_shown' not in st.session_state:
        # 스플래시 화면용 빈 컨테이너 생성
        splash_placeholder = st.empty()
        
        with splash_placeholder.container():
            # 스플래시가 뜨는 동안 사이드바와 상단 헤더를 숨겨서 앱처럼 보이게 하는 CSS
            st.markdown(
                """
                <style>
                [data-testid="stSidebar"] {display: none !important;}
                header {visibility: hidden !important;}
                </style>
                """,
                unsafe_allow_html=True
            )
            
            # 이미지를 중앙에 적절한 크기로 배치하기 위한 레이아웃 분할
            st.write("") # 상단 여백
            st.write("")
            col1, col2, col3 = st.columns([1, 1.5, 1])
            with col2:
                # PM님이 전달해준 파일명 그대로 사용! 
                st.image("차트 모야 스플래시.jpg", use_column_width=True)
        
        # 2.5초 동안 이미지 노출 대기 (필요시 숫자 수정 가능)
        time.sleep(2.5)
        
        # 스플래시 시청 완료 처리 후 스트림릿 화면 새로고침
        st.session_state['splash_shown'] = True
        st.rerun()
    else:
        # 스플래시 시청이 완료된 상태라면 메인 앱 로직 실행
        main()
