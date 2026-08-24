import os
import json
import math
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

st.set_page_config(
    page_title="AI Stock Valuation Dashboard Pro",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 커스텀 CSS ---
st.markdown("""
<style>
    @import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard/dist/web/static/pretendard.css');
    * { font-family: 'Pretendard', -apple-system, sans-serif !important; }
    
    .block-container { padding-top: 1.8rem; padding-bottom: 3rem; }
    
    /* 상단 지표 카드 */
    .metric-card {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 12px 14px;
        margin-bottom: 8px;
    }
    .metric-title {
        font-size: 0.78rem;
        color: #94a3b8;
        font-weight: 500;
        margin-bottom: 3px;
    }
    .metric-value {
        font-size: 1.25rem;
        font-weight: 700;
        color: #f8fafc;
    }
    
    /* 밸류에이션 카드 - 전통 가치 */
    .fair-card-value {
        background: linear-gradient(145deg, rgba(30, 41, 59, 0.4), rgba(15, 23, 42, 0.6));
        border: 1px solid rgba(148, 163, 184, 0.2);
        border-radius: 12px;
        padding: 14px 16px;
        text-align: left;
    }
    
    /* 밸류에이션 카드 - 성장주 가치 */
    .fair-card-growth {
        background: linear-gradient(145deg, rgba(49, 46, 129, 0.25), rgba(15, 23, 42, 0.6));
        border: 1px solid rgba(129, 140, 248, 0.35);
        border-radius: 12px;
        padding: 14px 16px;
        text-align: left;
    }
    
    .fair-formula {
        font-size: 0.73rem;
        color: #64748b;
        margin-top: 4px;
    }
    .diff-badge-up {
        background: rgba(34, 197, 94, 0.15);
        color: #4ade80;
        padding: 2px 7px;
        border-radius: 6px;
        font-size: 0.78rem;
        font-weight: 600;
        float: right;
    }
    .diff-badge-down {
        background: rgba(239, 68, 68, 0.15);
        color: #f87171;
        padding: 2px 7px;
        border-radius: 6px;
        font-size: 0.78rem;
        font-weight: 600;
        float: right;
    }
    
    /* 뉴스 박스 */
    .news-box {
        background: rgba(255, 255, 255, 0.03);
        border-left: 3px solid #6366f1;
        padding: 12px 16px;
        margin-bottom: 10px;
        border-radius: 0 10px 10px 0;
    }
    .news-summary {
        font-size: 0.88rem;
        color: #f1f5f9;
        font-weight: 500;
        line-height: 1.4;
        margin-bottom: 5px;
    }
    .news-title-link {
        text-decoration: none;
        color: #818cf8;
        font-weight: 600;
        font-size: 0.82rem;
    }
    .news-meta {
        font-size: 0.74rem;
        color: #94a3b8;
    }
    
    .report-container {
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid rgba(99, 102, 241, 0.25);
        border-radius: 14px;
        padding: 22px;
        margin-top: 12px;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# 1. RAG 데이터 수집 및 밸류에이션 계산
# -------------------------------------------------------------
def fetch_stock_technical_data(ticker: str):
    stock = yf.Ticker(ticker)
    df = stock.history(period="6mo")
    if df.empty:
        return {}, "N/A"
    
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_60'] = df['Close'].rolling(window=60).mean()
    df['SMA_120'] = df['Close'].rolling(window=120).mean()
    df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
    macd = ta.trend.MACD(df['Close'])
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    df['MACD_Hist'] = macd.macd_diff()
    
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

def format_market_cap(market_cap):
    if not market_cap or market_cap == "N/A":
        return "N/A"
    if market_cap >= 1e12:
        return f"${market_cap / 1e12:.2f}T"
    elif market_cap >= 1e9:
        return f"${market_cap / 1e9:.2f}B"
    elif market_cap >= 1e6:
        return f"${market_cap / 1e6:.2f}M"
    return f"${market_cap:,.0f}"

def fetch_fundamentals_and_valuation(ticker: str):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        market_cap = info.get("marketCap", "N/A")
        trailing_pe = info.get("trailingPE", "N/A")
        forward_pe = info.get("forwardPE", "N/A")
        pbr = info.get("priceToBook", "N/A")
        ps_ratio = info.get("priceToSalesTrailing12Months", "N/A")
        
        roe_raw = info.get("returnOnEquity", None)
        roe_pct = round(roe_raw * 100, 2) if roe_raw is not None else "N/A"
        eps = info.get("trailingEps", None)
        forward_eps = info.get("forwardEps", None)
        bps = info.get("bookValue", None)
        revenue_per_share = info.get("revenuePerShare", None)
        target_mean_price = info.get("targetMeanPrice", "N/A")
        
        # 성장률 추정 (earningsGrowth 또는 pegRatio 기반 추산, 기본 15~25%)
        earnings_growth = info.get("earningsGrowth", None)
        if earnings_growth and earnings_growth > 0:
            est_growth = min(earnings_growth * 100, 35.0)
        else:
            est_growth = 20.0  # 기본 성장률 가정
            
        # ==========================================
        # 1. 전통 3대 가치투자 모델 (Value Models)
        # ==========================================
        value_models = {}
        # 그레이엄 청산가치
        if eps and bps and eps > 0 and bps > 0:
            value_models["graham"] = round(math.sqrt(22.5 * eps * bps), 2)
        else:
            value_models["graham"] = "산출불가"
            
        # 피터 린치 가치모델
        if eps and eps > 0 and roe_raw and roe_raw > 0:
            value_models["peter_lynch"] = round(eps * min(roe_raw * 100, 25.0), 2)
        else:
            value_models["peter_lynch"] = "산출불가"

        # ROE-PBR 모델
        if bps and bps > 0 and roe_raw and roe_raw > 0:
            value_models["roe_pbr"] = round(bps * (roe_raw / 0.10), 2)
        else:
            value_models["roe_pbr"] = "산출불가"

        # ==========================================
        # 2. 성장주 3대 밸류에이션 모델 (Growth Models)
        # ==========================================
        growth_models = {}
        
        # (1) Forward PEG 1.5 적정가 모델: Forward EPS * (Growth * 1.5)
        f_eps = forward_eps if forward_eps and forward_eps > 0 else eps
        if f_eps and f_eps > 0:
            peg_fair = f_eps * (est_growth * 1.5)
            growth_models["forward_peg"] = round(peg_fair, 2)
        else:
            growth_models["forward_peg"] = "산출불가"

        # (2) PSR 타깃 매출가치 모델: Revenue Per Share * 타깃 PSR(빅테크 평균 8.5배)
        if revenue_per_share and revenue_per_share > 0:
            target_psr = 8.5
            growth_models["psr_target"] = round(revenue_per_share * target_psr, 2)
        else:
            growth_models["psr_target"] = "산출불가"

        # (3) 간이 2단계 DCF 성장가치 모델
        # WACC 9.0%, 영구성장률 2.5%, 5개년 고성장 적용
        if f_eps and f_eps > 0:
            wacc = 0.09
            g_long = 0.025
            pv_sum = 0
            cur_cf = f_eps
            for y in range(1, 6):
                cur_cf *= (1 + est_growth / 100)
                pv_sum += cur_cf / ((1 + wacc) ** y)
            terminal_val = (cur_cf * (1 + g_long)) / (wacc - g_long)
            pv_terminal = terminal_val / ((1 + wacc) ** 5)
            growth_models["dcf_growth"] = round(pv_sum + pv_terminal, 2)
        else:
            growth_models["dcf_growth"] = "산출불가"

        return {
            "market_cap_fmt": format_market_cap(market_cap),
            "trailing_pe": round(trailing_pe, 2) if isinstance(trailing_pe, (int, float)) else trailing_pe,
            "forward_pe": round(forward_pe, 2) if isinstance(forward_pe, (int, float)) else forward_pe,
            "pbr": round(pbr, 2) if isinstance(pbr, (int, float)) else pbr,
            "ps_ratio": round(ps_ratio, 2) if isinstance(ps_ratio, (int, float)) else ps_ratio,
            "roe": f"{roe_pct}%" if roe_pct != "N/A" else "N/A",
            "target_mean_price": target_mean_price,
            "value_models": value_models,
            "growth_models": growth_models
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
    try:
        stock = yf.Ticker(ticker)
        raw_news = stock.news
        if not raw_news:
            return []
            
        articles = []
        for n in raw_news[:limit]:
            content = n.get("content", {})
            if isinstance(content, dict) and content:
                title = content.get("title", "")
                summary = content.get("summary", "")
                publisher = content.get("provider", {}).get("displayName", "Yahoo Finance")
                click_url = content.get("clickThroughUrl", {})
                link = click_url.get("url", "") if isinstance(click_url, dict) else click_url
                if not link:
                    link = content.get("canonicalUrl", {}).get("url", "")
                pub_date = content.get("pubDate", "최근")
                if "T" in str(pub_date):
                    pub_date = str(pub_date).split("T")[0]
            else:
                title = n.get("title", "")
                summary = ""
                publisher = n.get("publisher", "Yahoo Finance")
                link = n.get("link", "")
                pub_time = n.get("providerPublishTime", None)
                pub_date = datetime.fromtimestamp(pub_time).strftime("%Y-%m-%d") if pub_time else "최근"
            
            if title:
                articles.append({
                    "title": title,
                    "raw_summary": summary,
                    "publisher": publisher,
                    "date": pub_date,
                    "link": link or f"https://finance.yahoo.com/quote/{ticker}"
                })
        return articles
    except Exception:
        return []

def summarize_news_with_gemini(news_list, api_key):
    if not news_list or not api_key:
        return news_list
        
    try:
        titles_and_summaries = [
            f"[{i+1}] 제목: {n['title']} / 영문요약: {n.get('raw_summary', '')}"
            for i, n in enumerate(news_list)
        ]
        prompt_text = (
            "아래 영문 주식 기사 목록을 읽고 핵심 내용을 투자자가 바로 이해할 수 있도록 한국어로 1~2문장으로 요약해 줘. "
            "반드시 JSON 리스트 형식(예: [\"요약1\", \"요약2\", ...])으로만 응답해.\n\n"
            + "\n".join(titles_and_summaries)
        )
        
        llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash", google_api_key=api_key)
        res = llm.invoke(prompt_text)
        
        cleaned = extract_clean_text(res.content).strip()
        if "```json" in cleaned:
            cleaned = cleaned.split("```json")[1].split("```")[0].strip()
        elif "```" in cleaned:
            cleaned = cleaned.split("