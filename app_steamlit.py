import os
import json
import math
import time
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

# -------------------------------------------------------------
# 1. RAG 데이터 수집 모듈
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
        
    asset_tickers = [
        ("vix", "^VIX"),
        ("dollar_index", "DX-Y.NYB"),
        ("wti_oil", "CL=F"),
        ("gold", "GC=F"),
        ("bitcoin", "BTC-USD")
    ]
    for name, ticker in asset_tickers:
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
        
        earnings_growth = info.get("earningsGrowth", None)
        if earnings_growth and earnings_growth > 0:
            est_growth = min(earnings_growth * 100, 35.0)
        else:
            est_growth = 20.0
            
        value_models = {}
        if eps and bps and eps > 0 and bps > 0:
            value_models["graham"] = round(math.sqrt(22.5 * eps * bps), 2)
        else:
            value_models["graham"] = "산출불가"
            
        if eps and eps > 0 and roe_raw and roe_raw > 0:
            value_models["peter_lynch"] = round(eps * min(roe_raw * 100, 25.0), 2)
        else:
            value_models["peter_lynch"] = "산출불가"

        if bps and bps > 0 and roe_raw and roe_raw > 0:
            value_models["roe_pbr"] = round(bps * (roe_raw / 0.10), 2)
        else:
            value_models["roe_pbr"] = "산출불가"

        growth_models = {}
        f_eps = forward_eps if forward_eps and forward_eps > 0 else eps
        if f_eps and f_eps > 0:
            growth_models["forward_peg"] = round(f_eps * (est_growth * 1.5), 2)
        else:
            growth_models["forward_peg"] = "산출불가"

        if revenue_per_share and revenue_per_share > 0:
            growth_models["psr_target"] = round(revenue_per_share * 8.5, 2)
        else:
            growth_models["psr_target"] = "산출불가"

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

def extract_clean_text(content):
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        return "\n".join([p["text"] if isinstance(p, dict) and "text" in p else str(p) for p in content])
    return str(content)

def fetch_recent_upgrades_downgrades(ticker: str, months: int = 2):
    try:
        stock = yf.Ticker(ticker)
        upgrades = stock.upgrades_downgrades
        if upgrades is None or upgrades.empty:
            return []
        
        cutoff_date = datetime.now() - timedelta(days=months * 30)
        
        if isinstance(upgrades.index, pd.DatetimeIndex):
            filtered = upgrades[upgrades.index >= cutoff_date.strftime("%Y-%m-%d")]
        elif 'Date' in upgrades.columns:
            upgrades['Date'] = pd.to_datetime(upgrades['Date'])
            filtered = upgrades[upgrades['Date'] >= cutoff_date]
        else:
            filtered = upgrades.head(7)
            
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

# -------------------------------------------------------------
# 2. UI 레이아웃
# -------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚡ **AI Stock Analyst**")
    ticker_input = st.text_input("종목 티커 (Ticker)", value="TSLA").upper()
    analyze_btn = st.button("🚀 분석 실행", type="primary", use_container_width=True)
    st.markdown("---")
    st.caption("• **가치 모델:** 그레이엄, 린치, ROE-PBR\n• **성장주 모델:** PEG 1.5, 타깃 PSR, DCF\n• **자산군 전망:** 현금·채권·주식·코인·금·원유\n• **추론 엔진:** Gemini 3.6 Flash")

st.header(f"📊 {ticker_input} 종합 밸류에이션 & 투자전략 리포트")

if analyze_btn:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        try:
            api_key = st.secrets["GEMINI_API_KEY"]
        except Exception:
            api_key = None
            
    if not api_key:
        st.error("GEMINI_API_KEY가 설정되지 않았습니다. Streamlit Secrets에 등록하세요.")
    else:
        with st.spinner(f"🔍 [{ticker_input}] 실시간 매크로/자산군 지표 수집 및 Gemini 분석 중..."):
            tech_data, stock_date = fetch_stock_technical_data(ticker_input)
            macro_data = fetch_macro_indicators()
            fund_data = fetch_fundamentals_and_valuation(ticker_input)
            sector_data = fetch_sector_performance()
            news_data = fetch_news(ticker_input, limit=5)
            analyst_data = fetch_recent_upgrades_downgrades(ticker_input, months=2)
            
            curr_p = tech_data.get('current_price', 0)
            
            # 1. 상단 핵심 메트릭
            with st.container(border=True):
                st.markdown("**🏢 핵심 시장 및 재무 지표**")
                r1_c1, r1_c2, r1_c3, r1_c4 = st.columns(4)
                r1_c1.metric("현재 주가", f"${curr_p}")
                r1_c2.metric("시가총액", str(fund_data.get('market_cap_fmt', 'N/A')))
                r1_c3.metric("PER (선행/후행)", f"{fund_data.get('forward_pe', 'N/A')} / {fund_data.get('trailing_pe', 'N/A')}")
                r1_c4.metric("PBR / PSR", f"{fund_data.get('pbr', 'N/A')} / {fund_data.get('ps_ratio', 'N/A')}")
                
                st.divider()
                
                st.markdown("**🌐 글로벌 매크로 & 6대 유동성 자산 실시간 현황**")
                r2_c1, r2_c2, r2_c3, r2_c4 = st.columns(4)
                r2_c1.metric(
                    "MACD (Signal)", 
                    f"{tech_data.get('macd', 'N/A')} ({tech_data.get('macd_signal', 'N/A')})", 
                    f"Hist: {tech_data.get('macd_hist', 'N/A'):+}" if isinstance(tech_data.get('macd_hist'), (int, float)) else None
                )
                r2_c2.metric("RSI(14) / MFI 수급", f"{tech_data.get('rsi_14', 'N/A')} / {tech_data.get('mfi_14', 'N/A')}")
                r2_c3.metric("미 10년물 금리 / 달러", f"{macro_data.get('us_10y_yield', {}).get('value', 'N/A')} / {macro_data.get('dollar_index', {}).get('value', 'N/A')}")
                r2_c4.metric("금($/oz) / 비트코인($)", f"${macro_data.get('gold', {}).get('value', 'N/A')} / ${macro_data.get('bitcoin', {}).get('value', 'N/A'):,}" if isinstance(macro_data.get('bitcoin', {}).get('value'), (int, float)) else f"${macro_data.get('gold', {}).get('value', 'N/A')} / N/A")

            # 2. 성장주 3대 밸류에이션 모델
            with st.container(border=True):
                st.markdown("##### 🚀 **성장주/빅테크 맞춤형 밸류에이션 모델 (Growth Models)**")
                g_models = fund_data.get('growth_models', {})
                g1, g2, g3, g4 = st.columns(4)
                
                target_p = fund_data.get('target_mean_price', 'N/A')
                diff_t = round(((target_p - curr_p) / curr_p) * 100, 1) if isinstance(target_p, (int, float)) and curr_p else None
                g1.metric("IB 컨센서스 목표가", f"${target_p}" if isinstance(target_p, (int, float)) else str(target_p), f"{diff_t:+.1f}%" if diff_t is not None else None)
                
                peg_p = g_models.get('forward_peg', 'N/A')
                diff_peg = round(((peg_p - curr_p) / curr_p) * 100, 1) if isinstance(peg_p, (int, float)) and curr_p else None
                g2.metric("Forward PEG 1.5 모델", f"${peg_p}" if isinstance(peg_p, (int, float)) else str(peg_p), f"{diff_peg:+.1f}%" if diff_peg is not None else None)
                
                psr_p = g_models.get('psr_target', 'N/A')
                diff_psr = round(((psr_p - curr_p) / curr_p) * 100, 1) if isinstance(psr_p, (int, float)) and curr_p else None
                g3.metric("PSR 타깃 매출가치 (8.5배)", f"${psr_p}" if isinstance(psr_p, (int, float)) else str(psr_p), f"{diff_psr:+.1f}%" if diff_psr is not None else None)
                
                dcf_p = g_models.get('dcf_growth', 'N/A')
                diff_dcf = round(((dcf_p - curr_p) / curr_p) * 100, 1) if isinstance(dcf_p, (int, float)) and curr_p else None
                g4.metric("2단계 DCF 현금흐름 모델", f"${dcf_p}" if isinstance(dcf_p, (int, float)) else str(dcf_p), f"{diff_dcf:+.1f}%" if diff_dcf is not None else None)

            # 3. 전통 가치투자 모델
            with st.container(border=True):
                st.markdown("##### 🏛️ **전통 제조업/자산가치 기반 3대 모델 (Value Models - 청산/장부가치 기준)**")
                v_models = fund_data.get('value_models', {})
                v1, v2, v3 = st.columns(3)
                
                g_val = v_models.get('graham', 'N/A')
                diff_g = round(((g_val - curr_p) / curr_p) * 100, 1) if isinstance(g_val, (int, float)) and curr_p else None
                v1.metric("그레이엄 청산가치", f"${g_val}" if isinstance(g_val, (int, float)) else str(g_val), f"{diff_g:+.1f}%" if diff_g is not None else None)
                
                l_val = v_models.get('peter_lynch', 'N/A')
                diff_l = round(((l_val - curr_p) / curr_p) * 100, 1) if isinstance(l_val, (int, float)) and curr_p else None
                v2.metric("피터 린치 가치모델", f"${l_val}" if isinstance(l_val, (int, float)) else str(l_val), f"{diff_l:+.1f}%" if diff_l is not None else None)
                
                r_val = v_models.get('roe_pbr', 'N/A')
                diff_r = round(((r_val - curr_p) / curr_p) * 100, 1) if isinstance(r_val, (int, float)) and curr_p else None
                v3.metric("ROE-PBR 자본가치", f"${r_val}" if isinstance(r_val, (int, float)) else str(r_val), f"{diff_r:+.1f}%" if diff_r is not None else None)

            st.caption(f"🕒 기준일자: 주가/재무제표 ({stock_date}) | FRED 국채금리 ({macro_data.get('us_10y_yield', {}).get('date', 'N/A')})")

            # 4. Gemini AI 분석 (단일 호출 + 재시도 핸들링)
            template = """
[RAG 주입 데이터]
1. 기술적/수급 데이터 ({ticker}) (기준일: {stock_date}):
{tech_json}

2. 매크로/6대 자산 지표 (국채금리, 달러, VIX, 금, 비트코인, 원유):
{macro_json}

3. 주요 섹터 5일 등락률:
{sector_json}

4. 펀더멘털 및 6대 밸류에이션:
{fund_json}

5. 최신 주요 기사:
{news_json}

6. 최근 2개월 증권가 투자의견 변동:
{analyst_json}

---

[지시사항]
위 데이터를 바탕으로 객관적이고 예리한 분석을 수행할 것:

1. 거시환경 및 시장 국면
- 경기 국면 (회복 / 활황 / 둔화 / 침체 판정)
- 단기 변동성 촉발 요인
- 최신 뉴스와 매크로 지표 기반 [6대 유동성 자산 변동 예측]:
  * 현금 (달러): 전망 (상승/중립/하락) 및 사유
  * 채권 (미 국채 등): 금리 경로에 따른 가격 전망 및 사유
  * 주식 (위험자산): 시장 유동성 및 실적 장세 기반 전망
  * 코인 (가상자산): 비트코인 등 위험선호 심리 및 유동성 민감도 전망
  * 금 (원자재/안전자산): 실질금리 및 지정학 리스크 기반 전망
  * 원유 (에너지): 공급망 및 경기 수요 기반 가격 전망
- 권장 자산 배분 비중 (주식 : 채권 : 대체자산/금·코인 : 현금)

2. 섹터 전망 및 순환매
- 상대적 강세 섹터 및 약세 섹터 요약
- 자금 순환매(Rotation) 방향

3. 밸류에이션 및 적정주가 종합 평가 ({ticker})
- 전통 가치모델과 성장주 모델 간의 괴리 원인 분석
- 해당 종목의 비즈니스 특성에 비추어 볼 때 가장 유효한 적정주가 밴드 제시
- PER/PBR/PSR/ROE 관점에서의 고평가/저평가 종합 판정

4. 종목 종합 평가 ({ticker})
- 기술적 분석: 이평선 배열, MACD 모멘텀(히스토그램), MFI 수급 상태, 지지/저항선
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
            
            response_content = None
            try:
                response = chain.invoke({
                    "ticker": ticker_input,
                    "stock_date": stock_date,
                    "tech_json": json.dumps(tech_data, indent=2, ensure_ascii=False),
                    "macro_json": json.dumps(macro_data, indent=2, ensure_ascii=False),
                    "sector_json": json.dumps(sector_data, indent=2, ensure_ascii=False),
                    "fund_json": json.dumps(fund_data, indent=2, ensure_ascii=False),
                    "news_json": json.dumps([{"title": n["title"], "summary": n.get("raw_summary", "")} for n in news_data], indent=2, ensure_ascii=False),
                    "analyst_json": json.dumps(analyst_data, indent=2, ensure_ascii=False)
                })
                response_content = extract_clean_text(response.content)
            except Exception as e:
                # Rate Limit 발생 시 4초 대기 후 1회 자동 재시도
                time.sleep(4)
                try:
                    response = chain.invoke({
                        "ticker": ticker_input,
                        "stock_date": stock_date,
                        "tech_json": json.dumps(tech_data, indent=2, ensure_ascii=False),
                        "macro_json": json.dumps(macro_data, indent=2, ensure_ascii=False),
                        "sector_json": json.dumps(sector_data, indent=2, ensure_ascii=False),
                        "fund_json": json.dumps(fund_data, indent=2, ensure_ascii=False),
                        "news_json": json.dumps([{"title": n["title"], "summary": n.get("raw_summary", "")} for n in news_data], indent=2, ensure_ascii=False),
                        "analyst_json": json.dumps(analyst_data, indent=2, ensure_ascii=False)
                    })
                    response_content = extract_clean_text(response.content)
                except Exception:
                    response_content = "⚠️ Gemini API 분당 요청 한도(Rate Limit)에 도달했습니다. 10~20초 후 다시 [분석 실행] 버튼을 눌러주세요."

            with st.container(border=True):
                st.markdown("### 📝 **AI 종합 분석 브리핑**")
                st.markdown(response_content)
            
            st.write("")

            # 5. 하단 뉴스 및 증권가 투자의견
            col_left, col_right = st.columns([1.1, 0.9])
            
            with col_left:
                with st.container(border=True):
                    st.markdown("##### 📰 **최신 주요 뉴스 및 기사 링크**")
                    if news_data:
                        for item in news_data:
                            st.markdown(f"**[{item['title']}]({item['link']})**")
                            if item.get("raw_summary"):
                                st.caption(f"요약: {item['raw_summary'][:150]}...")
                            st.caption(f"출처: {item['publisher']} | {item['date']}")
                            st.divider()
                    else:
                        st.info("수집된 최신 뉴스가 없습니다.")
                        
            with col_right:
                with st.container(border=True):
                    st.markdown("##### 🏛️ **최근 2개월 증권가 투자의견 변동**")
                    if analyst_data:
                        df_analyst = pd.DataFrame(analyst_data)
                        df_analyst.columns = ["일자", "증권사", "투자의견", "이전의견", "액션"]
                        st.dataframe(df_analyst, use_container_width=True, hide_index=True)
                    else:
                        st.info("최근 2개월간 등록된 투자의견 변동 내역이 없습니다.")