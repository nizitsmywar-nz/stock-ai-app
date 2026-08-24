import os
import json
import warnings
from datetime import datetime, timedelta

import streamlit as st
import pandas as pd
import pandas_datareader.data as web
import ta
import yfinance as yf
from dotenv import load_dotenv

from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

warnings.filterwarnings("ignore")
load_dotenv()

# --- 페이지 설정 ---
st.set_page_config(
    page_title="AI 주식 분석 대시보드 Pro",
    page_icon="📈",
    layout="wide"
)

# -------------------------------------------------------------
# 1. RAG 데이터 수집 모듈 (확장 지표 + 뉴스 + 애널리스트 의견)
# -------------------------------------------------------------
def fetch_stock_technical_data(ticker: str):
    stock = yf.Ticker(ticker)
    df = stock.history(period="6mo")
    if df.empty:
        return {}, "N/A"
    
    # 이동평균선
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_60'] = df['Close'].rolling(window=60).mean()
    df['SMA_120'] = df['Close'].rolling(window=120).mean()
    
    # 보조지표 (RSI, MACD, MFI, 볼린저밴드)
    df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
    macd = ta.trend.MACD(df['Close'])
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    df['MACD_Hist'] = macd.macd_diff()
    
    # 자금흐름지수 MFI (Money Flow Index)
    try:
        df['MFI'] = ta.volume.money_flow_index(df['High'], df['Low'], df['Close'], df['Volume'], window=14)
    except Exception:
        df['MFI'] = None

    bb = ta.volatility.BollingerBands(df['Close'], window=20, window_dev=2)
    df['BB_High'] = bb.bollinger_hband()
    df['BB_Low'] = bb.bollinger_lband()
    
    latest = df.iloc[-1]
    last_date = df.index[-1].strftime("%Y-%m-%d")
    
    data = {
        "ticker": ticker,
        "data_date": last_date,
        "current_price": round(float(latest['Close']), 2),
        "sma_20": round(float(latest['SMA_20']), 2) if pd.notnull(latest['SMA_20']) else "N/A",
        "sma_60": round(float(latest['SMA_60']), 2) if pd.notnull(latest['SMA_60']) else "N/A",
        "sma_120": round(float(latest['SMA_120']), 2) if pd.notnull(latest['SMA_120']) else "N/A",
        "rsi_14": round(float(latest['RSI']), 2) if pd.notnull(latest['RSI']) else "N/A",
        "mfi_14": round(float(latest['MFI']), 2) if pd.notnull(latest['MFI']) else "N/A",
        "macd": round(float(latest['MACD']), 2) if pd.notnull(latest['MACD']) else "N/A",
        "macd_signal": round(float(latest['MACD_Signal']), 2) if pd.notnull(latest['MACD_Signal']) else "N/A",
        "macd_hist": round(float(latest['MACD_Hist']), 2) if pd.notnull(latest['MACD_Hist']) else "N/A",
        "bb_upper": round(float(latest['BB_High']), 2) if pd.notnull(latest['BB_High']) else "N/A",
        "bb_lower": round(float(latest['BB_Low']), 2) if pd.notnull(latest['BB_Low']) else "N/A",
        "recent_volume_trend": "상승" if latest['Volume'] > df['Volume'].tail(5).mean() else "하락"
    }
    return data, last_date

def fetch_macro_indicators():
    macro_data = {}
    
    # 1. FRED 국채 금리
    try:
        end = datetime.now()
        start = end - timedelta(days=30)
        fred_res = web.DataReader('DGS10', 'fred', start, end).dropna()
        dgs10 = fred_res.iloc[-1, 0]
        macro_data["us_10y_yield"] = {
            "value": f"{round(float(dgs10), 2)}%",
            "date": fred_res.index[-1].strftime("%Y-%m-%d")
        }
    except Exception:
        macro_data["us_10y_yield"] = {"value": "N/A", "date": "N/A"}
        
    # 2. VIX, 유가, 달러
    for name, ticker in [("vix", "^VIX"), ("wti_oil", "CL=F"), ("dollar_index", "DX-Y.NYB")]:
        try:
            hist = yf.Ticker(ticker).history(period="5d")
            if not hist.empty:
                macro_data[name] = {
                    "value": round(float(hist['Close'].iloc[-1]), 2),
                    "date": hist.index[-1].strftime("%Y-%m-%d")
                }
            else:
                macro_data[name] = {"value": "N/A", "date": "N/A"}
        except Exception:
            macro_data[name] = {"value": "N/A", "date": "N/A"}
    return macro_data

def fetch_fundamentals(ticker: str):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        return {
            "trailing_pe": info.get("trailingPE", "N/A"),
            "forward_pe": info.get("forwardPE", "N/A"),
            "price_to_book": info.get("priceToBook", "N/A"),
            "roe": info.get("returnOnEquity", "N/A"),
            "operating_margins": info.get("operatingMargins", "N/A"),
            "target_mean_price": info.get("targetMeanPrice", "N/A"),
            "recommendation_key": info.get("recommendationKey", "N/A")
        }
    except Exception:
        return {}

def fetch_sector_performance():
    sector_etfs = ["XLK", "XLF", "XLE", "XLV", "XLI"]
    summary = {}
    for etf in sector_etfs:
        try:
            hist = yf.Ticker(etf).history(period="5d")
            if len(hist) >= 2:
                pct = (hist['Close'].iloc[-1] - hist['Close'].iloc[0]) / hist['Close'].iloc[0] * 100
                summary[etf] = {
                    "change_5d": f"{pct:+.2f}%",
                    "date": hist.index[-1].strftime("%Y-%m-%d")
                }
            else:
                summary[etf] = {"change_5d": "N/A", "date": "N/A"}
        except Exception:
            summary[etf] = {"change_5d": "N/A", "date": "N/A"}
    return summary

def fetch_news(ticker: str, limit: int = 5):
    """최신 종목 관련 주요 뉴스 기사 및 링크 수집"""
    try:
        stock = yf.Ticker(ticker)
        news_list = stock.news
        articles = []
        for n in news_list[:limit]:
            title = n.get("title", "")
            publisher = n.get("publisher", "")
            link = n.get("link", "")
            pub_time = n.get("providerPublishTime", None)
            pub_date = datetime.fromtimestamp(pub_time).strftime("%Y-%m-%d %H:%M") if pub_time else "최근"
            
            articles.append({
                "title": title,
                "publisher": publisher,
                "date": pub_date,
                "link": link
            })
        return articles
    except Exception:
        return []

def fetch_recent_upgrades_downgrades(ticker: str, months: int = 2):
    """최근 2개월간 증권사(IB) 투자의견 및 목표주가 변동 히스토리"""
    try:
        stock = yf.Ticker(ticker)
        upgrades = stock.upgrades_downgrades
        if upgrades is None or upgrades.empty:
            return []
        
        # 날짜 필터링 (최근 60일)
        cutoff_date = datetime.now() - timedelta(days=months * 30)
        
        # 인덱스가 DatetimeIndex인 경우 처리
        if isinstance(upgrades.index, pd.DatetimeIndex):
            filtered = upgrades[upgrades.index >= cutoff_date.strftime("%Y-%m-%d")]
        elif 'Date' in upgrades.columns:
            upgrades['Date'] = pd.to_datetime(upgrades['Date'])
            filtered = upgrades[upgrades['Date'] >= cutoff_date]
        else:
            filtered = upgrades.head(5)
            
        records = []
        for idx, row in filtered.head(7).iterrows():
            date_str = idx.strftime("%Y-%m-%d") if isinstance(idx, pd.Timestamp) else str(idx)[:10]
            records.append({
                "date": date_str,
                "firm": row.get("Firm", "N/A"),
                "to_grade": row.get("ToGrade", "N/A"),
                "from_grade": row.get("FromGrade", "N/A"),
                "action": row.get("Action", "N/A")
            })
        return records
    except Exception:
        return []

def extract_clean_text(content):
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        return "\n".join([p["text"] if isinstance(p, dict) and "text" in p else str(p) for p in content])
    return str(content)

# -------------------------------------------------------------
# 2. UI 및 메인 실행부
# -------------------------------------------------------------
st.title("📈 AI 주식 종합 분석 대시보드")
st.caption("실시간 주가·기술적 지표·거시경제·뉴스·증권가 컨센서스 기반 Gemini 분석 보고서")

with st.sidebar:
    st.header("⚙️ 종목 검색")
    ticker_input = st.text_input("종목 티커 입력 (예: GOOGL, NVDA, TSLA, AAPL)", value="GOOGL").upper()
    analyze_btn = st.button("분석 실행", type="primary", use_container_width=True)

if analyze_btn:
    with st.spinner(f"[{ticker_input}] 실시간 시장 지표 수집 및 분석 중..."):
        tech_data, stock_date = fetch_stock_technical_data(ticker_input)
        macro_data = fetch_macro_indicators()
        fund_data = fetch_fundamentals(ticker_input)
        sector_data = fetch_sector_performance()
        news_data = fetch_news(ticker_input, limit=5)
        analyst_data = fetch_recent_upgrades_downgrades(ticker_input, months=2)
        
        # 1. 상단 핵심 지표 메트릭 (MFI 및 MACD 반영)
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("현재가", f"${tech_data.get('current_price', 'N/A')}")
        c2.metric("RSI(14)", tech_data.get('rsi_14', 'N/A'))
        c3.metric("MFI 수급(14)", tech_data.get('mfi_14', 'N/A'))
        c4.metric("미 10년물 금리", macro_data.get('us_10y_yield', {}).get('value', 'N/A'))
        c5.metric("VIX 변동성", macro_data.get('vix', {}).get('value', 'N/A'))
        
        st.divider()
        
        # 2. 데이터 기준일 정보 표시
        with st.expander("🕒 분석 데이터 수집 기준일자 (최신 여부 검증)", expanded=False):
            d_c1, d_c2, d_c3 = st.columns(3)
            d_c1.write(f"- **주가/기술지표 기준일:** {stock_date}")
            d_c2.write(f"- **미 국채 10년물 기준일:** {macro_data.get('us_10y_yield', {}).get('date', 'N/A')}")
            d_c3.write(f"- **VIX / 유가 기준일:** {macro_data.get('vix', {}).get('date', 'N/A')}")
        
        # 3. Gemini AI 분석 리포트 생성
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            try:
                api_key = st.secrets["GEMINI_API_KEY"]
            except Exception:
                api_key = None
                
        if not api_key:
            st.error("GEMINI_API_KEY가 설정되지 않았습니다. Streamlit Secrets에 등록하세요.")
        else:
            template = """
[RAG 주입 데이터]
1. 기술적/수급 데이터 ({ticker}) (기준일: {stock_date}):
{tech_json}

2. 매크로/시장 지표:
{macro_json}

3. 주요 섹터 5일 등락률:
{sector_json}

4. 펀더멘털/밸류에이션:
{fund_json}

5. 최신 주요 기사 헤드라인:
{news_json}

6. 최근 2개월 증권가 투자의견 변동:
{analyst_json}

---

[지시사항]
위 [RAG 주입 데이터]를 기반으로 아래 항목을 사실 위주로 날카롭게 분석할 것:

1. 거시환경 및 시장 국면
- 경기 국면 (회복 / 활황 / 둔화 / 침체 판정)
- 단기 변동성 촉발 요인
- 권장 자산 배분 비중 (주식 : 채권 : 현금)

2. 섹터 전망 및 순환매
- 상대적 강세 섹터 및 약세 섹터 요약
- 자금 순환매(Rotation) 방향

3. 종목 종합 평가 ({ticker})
- 기술적 분석: 이평선 배열, MACD 모멘텀(히스토그램), MFI 자금 유입 상태, 지지/저항선, 과매수/과매도 여부
- 최근 증권가 컨센서스 평가: 상향/하향 추세 및 시장의 기대치 요약
- 스코어카드 (각 10점 만점): 성장성, 수익성, 밸류에이션, 해자, 리스크
- 종합 평점 및 최종 투자 의견 (적극매수 / 분할매수 / 관망 / 비중축소)
- 매매 시나리오: 분할 매수 밴드, 목표가/익절 라인, 손절(Stop-loss) 기준선
"""
            prompt = PromptTemplate(
                input_variables=["ticker", "stock_date", "tech_json", "macro_json", "sector_json", "fund_json", "news_json", "analyst_json"],
                template=template
            )
            llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash", google_api_key=api_key)
            chain = prompt | llm
            
            response = chain.invoke({
                "ticker": ticker_input,
                "stock_date": stock_date,
                "tech_json": json.dumps(tech_data, indent=2, ensure_ascii=False),
                "macro_json": json.dumps(macro_data, indent=2, ensure_ascii=False),
                "sector_json": json.dumps(sector_data, indent=2, ensure_ascii=False),
                "fund_json": json.dumps(fund_data, indent=2, ensure_ascii=False),
                "news_json": json.dumps([n['title'] for n in news_data], indent=2, ensure_ascii=False),
                "analyst_json": json.dumps(analyst_data, indent=2, ensure_ascii=False)
            })
            
            st.markdown(extract_clean_text(response.content))
            
        st.divider()

        # 4. 하단 부가 섹션 (최신 뉴스 링크 & 증권가 의견 표)
        col_left, col_right = st.columns(2)
        
        with col_left:
            st.subheader("📰 최신 주요 뉴스 및 링크")
            if news_data:
                for item in news_data:
                    st.markdown(f"- **[{item['title']}]({item['link']})**  \n  *{item['publisher']} ({item['date']})*")
            else:
                st.info("수집된 최신 뉴스가 없습니다.")
                
        with col_right:
            st.subheader("🏛️ 최근 2개월 증권사 투자의견 변동")
            if analyst_data:
                df_analyst = pd.DataFrame(analyst_data)
                df_analyst.columns = ["일자", "증권사", "투자의견", "이전의견", "액션"]
                st.dataframe(df_analyst, use_container_width=True, hide_index=True)
            else:
                st.info("최근 2개월간 등록된 투자의견 변동 내역이 없습니다.")