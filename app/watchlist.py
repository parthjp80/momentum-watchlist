#!/usr/bin/env python3
"""
Momentum Stock Watchlist Builder — AUTO-DISCOVERY EDITION
==========================================================
Zero manual ticker input required. The program:

  1. DISCOVERS candidates automatically from:
       - S&P 500 constituents   (Wikipedia)
       - NASDAQ-100 constituents (Wikipedia)
       - Day's top % gainers    (Yahoo Finance screener)
       - Day's high-volume movers (Yahoo Finance screener)
       - ETF top holdings: QQQ, XLK, ARKK, IJR

  2. PRE-FILTERS on minimum liquidity & price thresholds so penny stocks
     and illiquid names are excluded before heavy work starts.

  3. SCORES every candidate across 7 momentum criteria using real market data.

  4. ENRICHES the top N picks with Claude AI + web_search for live catalysts,
     entry notes, key risks, and active signals.

  5. EMAILS a rich HTML report to configured recipients.

No WATCHLIST env var needed. Everything is discovered fresh each day.
"""

import os
import re
import json
import time
import logging
import smtplib
import requests
import anthropic
import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from jinja2 import Environment, FileSystemLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
log = logging.getLogger(__name__)

# ── Config from env ──────────────────────────────────────────────────────────
ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]
EMAIL_FROM        = os.environ["EMAIL_FROM"]
EMAIL_TO          = os.environ["EMAIL_TO"]
SMTP_HOST         = os.environ.get("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT         = int(os.environ.get("SMTP_PORT", "587"))
SMTP_USER         = os.environ["SMTP_USER"]
SMTP_PASS         = os.environ["SMTP_PASS"]

TOP_N          = int(os.environ.get("TOP_N", "10"))
MIN_PRICE      = float(os.environ.get("MIN_PRICE", "5"))
MIN_AVG_VOL    = float(os.environ.get("MIN_AVG_VOLUME", "500000"))
MAX_CANDIDATES = int(os.environ.get("MAX_CANDIDATES", "200"))

WEIGHTS = {
    "sharpe":   float(os.environ.get("W_SHARPE",   "20")),
    "volume":   float(os.environ.get("W_VOLUME",   "18")),
    "breakout": float(os.environ.get("W_BREAKOUT", "15")),
    "ema":      float(os.environ.get("W_EMA",      "15")),
    "atr":      float(os.environ.get("W_ATR",      "12")),
    "tape":     float(os.environ.get("W_TAPE",     "10")),
    "pattern":  float(os.environ.get("W_PATTERN",  "10")),
}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0 Safari/537.36"
    )
}


# ════════════════════════════════════════════════════════════════════════════
# STAGE 1 — DISCOVER CANDIDATES
# ════════════════════════════════════════════════════════════════════════════

# NASDAQ-100 components (as of Q2 2026 rebalance) — fallback if live fetch fails
_NDX100_FALLBACK = [
    "AAPL","MSFT","NVDA","AMZN","META","TSLA","GOOGL","GOOG","AVGO","COST",
    "NFLX","TMUS","AMD","PEP","LIN","CSCO","ADBE","QCOM","TXN","AMGN",
    "INTU","ISRG","CMCSA","BKNG","MU","HON","AMAT","VRTX","ADP","PANW",
    "SBUX","ADI","GILD","LRCX","MELI","MDLZ","INTC","REGN","KLAC","SNPS",
    "CDNS","CEG","CTAS","PYPL","CSX","ORLY","MRNA","NXPI","MRVL","PCAR",
    "ABNB","FTNT","CRWD","MNST","KDP","ODFL","ROST","IDXX","DXCM","FAST",
    "AZN","CTSH","EA","WBD","BIIB","FANG","GEHC","ON","EXC","XEL",
    "TEAM","ZS","ANSS","VRSK","DLTR","DDOG","CSGP","GFS","TTD","TTWO",
    "SIRI","ILMN","WBA","ALGN","LCID","ENPH","ZM","RIVN","DASH","EBAY",
    "MCHP","CPRT","PAYX","CHTR","PDD","ASML","ARM","CCEP","CDW","SMCI",
]

def get_sp500() -> list:
    # GitHub-hosted S&P 500 constituents CSV (reliable, no bot detection)
    url = ("https://raw.githubusercontent.com/datasets/s-and-p-500-companies"
           "/main/data/constituents.csv")
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        import io
        df = pd.read_csv(io.StringIO(resp.text))
        tickers = [str(t).replace(".", "-") for t in df["Symbol"].dropna()
                   if re.match(r"^[A-Z]{1,5}$", str(t).replace(".", ""))]
        log.info(f"  S&P 500 GitHub: {len(tickers)} tickers")
        return tickers
    except Exception as e:
        log.warning(f"SP500 (GitHub) fetch failed: {e}")
        return []


def get_nasdaq100() -> list:
    # Wikipedia NASDAQ-100 with retries + fallback to hardcoded list
    try:
        tables = pd.read_html(
            "https://en.wikipedia.org/wiki/Nasdaq-100",
            attrs={"id": "constituents"},
            storage_options={"User-Agent": HEADERS["User-Agent"]},
        )
        tickers = tables[0]["Ticker"].dropna().tolist()
        if len(tickers) > 50:
            log.info(f"  NASDAQ-100 Wikipedia: {len(tickers)} tickers")
            return tickers
    except Exception:
        pass
    log.warning("NASDAQ-100 Wikipedia failed — using hardcoded fallback list")
    return _NDX100_FALLBACK


def get_yahoo_movers(screener: str, count: int = 25) -> list:
    """Pull tickers from Yahoo Finance screener page via regex on embedded JSON."""
    url = (f"https://finance.yahoo.com/screener/predefined/{screener}"
           f"?offset=0&count={count}")
    try:
        resp = requests.get(url, headers=HEADERS, timeout=12)
        matches = re.findall(r'"symbol":"([A-Z]{1,5})"', resp.text)
        return list(dict.fromkeys(matches))[:count]
    except Exception as e:
        log.warning(f"Yahoo screener ({screener}) failed: {e}")
        return []


def get_etf_holdings(etf: str, max_h: int = 25) -> list:
    try:
        obj = yf.Ticker(etf)
        data = obj.funds_data
        if data is not None:
            holdings = data.top_holdings
            if holdings is not None and not holdings.empty:
                return holdings.index.tolist()[:max_h]
    except Exception:
        pass
    return []


def get_yf_screen_movers() -> list:
    """Use yfinance built-in screen for most-actives as a fallback source."""
    try:
        result = yf.screen("most_actives", size=50)
        if result and "quotes" in result:
            return [q["symbol"] for q in result["quotes"] if q.get("symbol")]
    except Exception:
        pass
    return []


def clean_ticker(t: str) -> str | None:
    t = str(t).upper().strip()
    if not t or len(t) > 6:
        return None
    if not re.match(r'^[A-Z][A-Z0-9.\-]{0,5}$', t):
        return None
    return t


def discover_candidates() -> list:
    log.info("=== STAGE 1: Discovering candidates ===")
    all_tickers = []

    # Priority 1 — fresh momentum: today's movers
    for src, fn in [
        ("Yahoo day-gainers",   lambda: get_yahoo_movers("day_gainers", 30)),
        ("Yahoo most-actives",  lambda: get_yahoo_movers("most_actives", 30)),
        ("yfinance screen",     get_yf_screen_movers),
    ]:
        log.info(f"  Fetching {src}...")
        tickers = fn()
        log.info(f"  → {len(tickers)} tickers")
        all_tickers.extend(tickers)

    # Priority 2 — broad index coverage
    for src, fn in [
        ("S&P 500",    get_sp500),
        ("NASDAQ-100", get_nasdaq100),
    ]:
        log.info(f"  Fetching {src}...")
        tickers = fn()
        log.info(f"  → {len(tickers)} tickers")
        all_tickers.extend(tickers)

    # Priority 3 — thematic / growth ETFs
    for etf in ["QQQ", "XLK", "ARKK", "IJR"]:
        log.info(f"  Fetching {etf} holdings...")
        tickers = get_etf_holdings(etf)
        log.info(f"  → {len(tickers)} tickers")
        all_tickers.extend(tickers)

    # Deduplicate preserving order (movers stay at front)
    seen, unique = set(), []
    for raw in all_tickers:
        t = clean_ticker(raw)
        if t and t not in seen:
            seen.add(t)
            unique.append(t)

    log.info(f"Total unique candidates: {len(unique)}")
    return unique[: MAX_CANDIDATES * 2]


# ════════════════════════════════════════════════════════════════════════════
# STAGE 2 — LIQUIDITY PRE-FILTER
# ════════════════════════════════════════════════════════════════════════════

def passes_liquidity(ticker: str) -> bool:
    """Quick check using fast_info — avoids downloading full history."""
    try:
        info = yf.Ticker(ticker).fast_info
        price = getattr(info, "last_price", None) or getattr(info, "regularMarketPrice", None)
        vol   = getattr(info, "three_month_average_volume", None)
        if price is None or vol is None:
            return False
        return float(price) >= MIN_PRICE and float(vol) >= MIN_AVG_VOL
    except Exception:
        return False


# ════════════════════════════════════════════════════════════════════════════
# STAGE 3 — FETCH INDICATORS
# ════════════════════════════════════════════════════════════════════════════

def fetch_ticker_data(ticker: str) -> dict | None:
    try:
        df = yf.download(ticker, period="60d", interval="1d",
                         auto_adjust=True, progress=False, actions=False)
        if df.empty or len(df) < 15:
            return None

        closes  = df["Close"].squeeze().dropna()
        volumes = df["Volume"].squeeze().dropna()
        highs   = df["High"].squeeze().dropna()
        lows    = df["Low"].squeeze().dropna()

        if len(closes) < 10:
            return None

        price = float(closes.iloc[-1])
        if price < MIN_PRICE:
            return None

        # Actual liquidity guard on historical data
        avg_vol_20 = float(volumes.iloc[-21:-1].mean()) if len(volumes) > 21 else float(volumes.mean())
        if avg_vol_20 < MIN_AVG_VOL:
            return None

        # Modified Sharpe (no risk-free rate)
        returns = closes.pct_change().dropna()
        mod_sharpe = (float(returns.mean() / returns.std() * np.sqrt(252))
                      if returns.std() > 0 else 0.0)

        # Volume ratio vs 10-day avg
        vol_10avg = float(volumes.iloc[-11:-1].mean())
        vol_ratio = float(volumes.iloc[-1] / vol_10avg) if vol_10avg > 0 else 1.0

        # 9-period EMA
        ema9       = closes.ewm(span=9, adjust=False).mean()
        ema9_val   = float(ema9.iloc[-1])
        ema9_prev  = float(ema9.iloc[-5]) if len(ema9) >= 5 else ema9_val
        ema9_slope = (ema9_val - ema9_prev) / ema9_prev * 100 if ema9_prev else 0.0

        # ATR (14-period) + ATR momentum ratio vs 30-day avg
        tr = pd.concat([
            highs - lows,
            (highs - closes.shift(1)).abs(),
            (lows  - closes.shift(1)).abs()
        ], axis=1).max(axis=1)
        atr14     = float(tr.rolling(14).mean().iloc[-1])
        atr_avg30 = float(tr.rolling(min(30, len(tr))).mean().iloc[-1])
        atr_pct   = atr14 / price * 100
        atr_ratio = atr14 / atr_avg30 if atr_avg30 > 0 else 1.0

        # Breakout vs 20-day high
        high_20      = float(highs.iloc[-21:-1].max()) if len(highs) > 21 else float(highs.max())
        high_52w     = float(highs.max())
        low_52w      = float(lows.min())
        breakout_pct = (price - high_20) / high_20 * 100
        range_pct    = ((price - low_52w) / (high_52w - low_52w) * 100
                        if high_52w > low_52w else 50.0)

        prev_close       = float(closes.iloc[-2]) if len(closes) > 1 else price
        price_change_pct = (price - prev_close) / prev_close * 100

        return {
            "ticker":           ticker,
            "price":            round(price, 2),
            "price_change_pct": round(price_change_pct, 2),
            "mod_sharpe":       round(mod_sharpe, 3),
            "vol_ratio":        round(vol_ratio, 2),
            "avg_vol_20":       int(avg_vol_20),
            "ema9":             round(ema9_val, 2),
            "ema9_slope":       round(ema9_slope, 3),
            "above_ema9":       price > ema9_val,
            "atr_pct":          round(atr_pct, 2),
            "atr_ratio":        round(atr_ratio, 2),
            "breakout_pct":     round(breakout_pct, 2),
            "range_pct":        round(range_pct, 1),
            "high_20":          round(high_20, 2),
            "high_52w":         round(high_52w, 2),
            "low_52w":          round(low_52w, 2),
            "recent_returns":   [round(float(r), 5) for r in returns.tail(20).tolist()],
        }
    except Exception as e:
        log.debug(f"{ticker}: {e}")
        return None


# ════════════════════════════════════════════════════════════════════════════
# STAGE 4 — SCORE
# ════════════════════════════════════════════════════════════════════════════

def score_stock(d: dict) -> dict:
    sharpe_score  = int(np.clip((d["mod_sharpe"] + 1) / 2 * 100, 0, 100))
    volume_score  = int(np.clip(20 + 26.7 * (d["vol_ratio"] - 1), 0, 100))
    bp            = d["breakout_pct"]
    breakout_score = int(np.clip(70 + bp * 5, 0, 100)) if bp > 0 \
                     else int(np.clip(70 + bp * 3, 0, 100))
    ema_base      = 75 if d["above_ema9"] else 30
    ema_score     = int(np.clip(ema_base + np.clip(d["ema9_slope"] * 15, -20, 25), 0, 100))
    atr_score     = int(np.clip(30 + 35 * (d["atr_ratio"] - 1), 0, 100))
    tape_score    = int(np.clip(
        40
        + (20 if d["above_ema9"]            else 0)
        + (20 if d["price_change_pct"] > 0.5 else 0)
        + (20 if d["vol_ratio"] > 1.5       else 0),
        0, 100))
    range_score   = int(np.clip(d["range_pct"], 0, 100))
    pos_days      = sum(1 for r in d["recent_returns"] if r > 0)
    consistency   = pos_days / len(d["recent_returns"]) * 100 if d["recent_returns"] else 50
    pattern_score = int((range_score + consistency) / 2)

    scores = {
        "sharpe": sharpe_score, "volume": volume_score,
        "breakout": breakout_score, "ema": ema_score,
        "atr": atr_score, "tape": tape_score, "pattern": pattern_score,
    }

    total_w = sum(WEIGHTS.values())
    overall = int(round(sum(scores[k] * WEIGHTS[k] for k in scores) / total_w))
    signal  = ("strong" if overall >= 75 else "good" if overall >= 60
               else "watch" if overall >= 45 else "weak")

    return {**d, "scores": scores, "overall_score": overall, "signal": signal}


# ════════════════════════════════════════════════════════════════════════════
# STAGE 5 — CLAUDE AI ENRICHMENT WITH LIVE WEB SEARCH
# ════════════════════════════════════════════════════════════════════════════

def enrich_with_claude(top_stocks: list) -> list:
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    today  = datetime.now().strftime("%A, %B %d, %Y")

    summaries = [{
        "ticker": s["ticker"], "price": s["price"],
        "price_change": s["price_change_pct"], "overall_score": s["overall_score"],
        "signal": s["signal"], "vol_ratio": s["vol_ratio"],
        "above_ema9": s["above_ema9"], "ema9_slope": s["ema9_slope"],
        "atr_pct": s["atr_pct"], "breakout_pct": s["breakout_pct"],
        "scores": s["scores"],
    } for s in top_stocks]

    prompt = f"""Today is {today}. You are a quantitative momentum analyst with live web access.

These stocks were algorithmically selected as today's top momentum picks using
Modified Sharpe ratio, volume analysis, EMA trend, ATR momentum, and breakout signals:

{json.dumps(summaries, indent=2)}

Use web search to find the latest news and catalysts for each ticker, then respond
ONLY with a JSON array (no markdown, no preamble, no extra text):
[
  {{
    "ticker": "AAPL",
    "companyName": "Apple Inc.",
    "sector": "Technology",
    "catalyst": "one sentence: current news catalyst or market driver based on today's news",
    "entryNote": "specific actionable entry note for a trader today (price levels, conditions)",
    "keyRisk": "one sentence key risk or reason for caution",
    "activeSignals": ["signal 1 with specifics", "signal 2 with specifics", "signal 3 with specifics"]
  }}
]

Be specific with price levels and realistic. Use web_search for each ticker before responding."""

    try:
        messages_hist = [{"role": "user", "content": prompt}]
        while True:
            response = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=3000,
                tools=[{"type": "web_search_20250305", "name": "web_search"}],
                messages=messages_hist,
            )
            messages_hist.append({"role": "assistant", "content": response.content})
            if response.stop_reason in ("end_turn", None):
                break
            if response.stop_reason != "tool_use":
                break
        raw = "".join(b.text for b in response.content if hasattr(b, "text"))
    except Exception as e:
        log.warning(f"Claude enrichment failed: {e}")
        raw = "[]"

    clean = raw.replace("```json", "").replace("```", "").strip()
    try:
        enrichments = json.loads(clean[clean.index("["):clean.rindex("]")+1])
    except Exception:
        enrichments = []

    emap = {e["ticker"]: e for e in enrichments}
    for s in top_stocks:
        e = emap.get(s["ticker"], {})
        s["companyName"]   = e.get("companyName", s["ticker"])
        s["sector"]        = e.get("sector", "")
        s["catalyst"]      = e.get("catalyst", "No catalyst data")
        s["entryNote"]     = e.get("entryNote", "Monitor for volume confirmation entry")
        s["keyRisk"]       = e.get("keyRisk", "")
        s["activeSignals"] = e.get("activeSignals", [])
    return top_stocks


# ════════════════════════════════════════════════════════════════════════════
# STAGE 5b — TRADE PLAN: ENTRY, STOP, TARGETS, VOLUME CONFIRMATION
# ════════════════════════════════════════════════════════════════════════════

def generate_trade_plans(top_stocks: list) -> list:
    """
    For each of the top stocks, generate a detailed trade plan including:
      - Volume confirmation entry condition (specific volume threshold)
      - Entry trigger price (exact level to enter)
      - Stop loss (hard stop + trailing stop logic)
      - Price targets: T1 (conservative), T2 (base), T3 (extended)
      - Risk/reward ratio
      - Position sizing guidance (% of portfolio, max risk)
      - Time horizon
      - Invalidation condition (when the setup is no longer valid)
    Uses web_search to check current intraday levels and news.
    """
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    today  = datetime.now().strftime("%A, %B %d, %Y")

    trade_inputs = [{
        "ticker":        s["ticker"],
        "companyName":   s.get("companyName", s["ticker"]),
        "price":         s["price"],
        "price_change":  s["price_change_pct"],
        "high_20":       s["high_20"],
        "high_52w":      s["high_52w"],
        "low_52w":       s["low_52w"],
        "ema9":          s["ema9"],
        "ema9_slope":    s["ema9_slope"],
        "atr_pct":       s["atr_pct"],
        "vol_ratio":     s["vol_ratio"],
        "avg_vol_20":    s["avg_vol_20"],
        "above_ema9":    s["above_ema9"],
        "breakout_pct":  s["breakout_pct"],
        "overall_score": s["overall_score"],
        "signal":        s["signal"],
        "catalyst":      s.get("catalyst", ""),
        "activeSignals": s.get("activeSignals", []),
    } for s in top_stocks]

    prompt = f"""Today is {today}. You are an expert momentum trader and risk manager.

You have been given the top {len(top_stocks)} momentum stocks selected algorithmically today.
Use web search to check current price levels and recent chart structure for each, then generate
a precise trade plan for each stock.

Stock data:
{json.dumps(trade_inputs, indent=2)}

CRITICAL INSTRUCTIONS:
- Volume confirmation entry: specify the EXACT volume threshold (e.g. "Enter only if 5-min bar volume > 1.5x its 10-bar avg") 
- Entry price: give a specific dollar level or condition (e.g. "$213.50 breakout above morning high with volume")
- Stop loss: give a specific dollar level based on ATR or key support (NOT a generic percentage)
- Targets T1/T2/T3: use ATR multiples, prior resistance levels, or round numbers — be specific with dollar prices
- Risk/reward: calculate actual R:R ratio for each target
- Position size: give a concrete % of portfolio for a $100k account example
- Time horizon: intraday, swing (2-5 days), or positional (1-3 weeks)
- Invalidation: the exact price or condition that kills the setup

Respond ONLY with a JSON array (no markdown, no preamble):
[
  {{
    "ticker": "AAPL",
    "setupType": "Momentum breakout / Flag continuation / EMA bounce / Volume surge",
    "timeHorizon": "Swing (2-5 days)",
    "volumeConfirmation": {{
      "minimumVolume": "Entry bar must show > 2x average 10-day volume (> 120M shares)",
      "volumePattern": "Look for accelerating volume on 5-min bars as price approaches entry",
      "tapeSigns": "Watch for large bid stacking at entry level, minimal ask pressure"
    }},
    "entry": {{
      "triggerPrice": 214.50,
      "triggerCondition": "Break and close above $214.50 (20-day high) on 5-min chart with volume > 1.5x avg",
      "entryType": "Breakout",
      "alternateEntry": "Pullback to 9 EMA (~$211.20) on declining volume for lower-risk entry"
    }},
    "stopLoss": {{
      "hardStop": 209.80,
      "hardStopRationale": "Below 2x ATR ($2.10) from entry and under 9 EMA — momentum invalidated",
      "trailingStop": "Trail stop to breakeven once T1 hit; trail by 1 ATR below each new daily high thereafter",
      "maxRiskDollars": 470,
      "maxRiskPct": 2.2
    }},
    "targets": [
      {{
        "label": "T1",
        "price": 218.50,
        "rationale": "1.5x ATR extension from entry, prior intraday resistance",
        "riskReward": "1.9:1",
        "action": "Take 40% off position, move stop to breakeven"
      }},
      {{
        "label": "T2",
        "price": 223.00,
        "rationale": "3x ATR extension, measured move from flag base",
        "riskReward": "4.0:1",
        "action": "Take another 40% off position, trail remaining"
      }},
      {{
        "label": "T3",
        "price": 230.00,
        "rationale": "Extended target at prior all-time high / round number resistance",
        "riskReward": "7.2:1",
        "action": "Final 20% — let it run with trailing stop"
      }}
    ],
    "positionSizing": {{
      "portfolioRiskPct": 1.0,
      "sharesFor100kAccount": 21,
      "dollarExposure": 4505,
      "portfolioPct": 4.5
    }},
    "invalidation": "Setup fails if price closes below $209.80 or volume dries up below 0.8x average on any breakout attempt",
    "additionalNotes": "Any extra context on timing, news events to watch, or sector tailwinds"
  }}
]

Use real ATR values and actual price levels from the data provided. Be precise — traders will use these numbers."""

    try:
        messages_hist = [{"role": "user", "content": prompt}]
        while True:
            response = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=6000,
                tools=[{"type": "web_search_20250305", "name": "web_search"}],
                messages=messages_hist,
            )
            messages_hist.append({"role": "assistant", "content": response.content})
            if response.stop_reason in ("end_turn", None):
                break
            if response.stop_reason != "tool_use":
                break
        raw = "".join(b.text for b in response.content if hasattr(b, "text"))
    except Exception as e:
        log.warning(f"Trade plan generation failed: {e}")
        raw = "[]"

    clean = raw.replace("```json", "").replace("```", "").strip()
    try:
        plans = json.loads(clean[clean.index("["):clean.rindex("]")+1])
    except Exception as e:
        log.warning(f"Trade plan JSON parse failed: {e}")
        plans = []

    plan_map = {p["ticker"]: p for p in plans}
    for s in top_stocks:
        s["tradePlan"] = plan_map.get(s["ticker"], None)

    log.info(f"Trade plans generated for {len(plans)}/{len(top_stocks)} stocks")
    return top_stocks


# ════════════════════════════════════════════════════════════════════════════
# STAGE 6 — EMAIL
# ════════════════════════════════════════════════════════════════════════════

def send_email(top_stocks: list, scan_stats: dict):
    today_str = datetime.now().strftime("%A, %B %d, %Y")
    env       = Environment(loader=FileSystemLoader("/app/templates"))
    html_body = env.get_template("email.html").render(
        stocks=top_stocks, date=today_str,
        weights=WEIGHTS, top_n=TOP_N, scan_stats=scan_stats
    )

    lines = [
        f"TOP {TOP_N} MOMENTUM STOCKS — {today_str}",
        f"Universe: {scan_stats['universe']} candidates → "
        f"{scan_stats['liquid']} liquid → {scan_stats['scanned']} scored",
        "=" * 60,
    ]
    for i, s in enumerate(top_stocks, 1):
        chg = s["price_change_pct"]
        lines.append(
            f"{i:2}. {s['ticker']:6} | {s['overall_score']:3}/100 | "
            f"{s['signal'].upper():6} | ${s['price']} "
            f"({'+' if chg >= 0 else ''}{chg}%) | Vol {s['vol_ratio']}x"
        )
        for icon, key in [("⚡", "catalyst"), ("📍", "entryNote"), ("⚠️", "keyRisk")]:
            if s.get(key):
                lines.append(f"    {icon} {s[key]}")
        tp = s.get("tradePlan")
        if tp:
            entry = tp.get("entry", {})
            sl    = tp.get("stopLoss", {})
            tgts  = tp.get("targets", [])
            lines.append(f"    📊 Setup: {tp.get('setupType','')} | {tp.get('timeHorizon','')}")
            lines.append(f"    🔔 Vol confirm: {tp.get('volumeConfirmation',{}).get('minimumVolume','')}")
            lines.append(f"    ▶  Entry:  ${entry.get('triggerPrice','')} — {entry.get('triggerCondition','')}")
            lines.append(f"    🛑 Stop:   ${sl.get('hardStop','')} — {sl.get('hardStopRationale','')}")
            for t in tgts:
                lines.append(f"    🎯 {t['label']}: ${t['price']} ({t['riskReward']} R:R) — {t['rationale']}")
            if tp.get("invalidation"):
                lines.append(f"    ❌ Invalid: {tp['invalidation']}")
        lines.append("")

    msg            = MIMEMultipart("alternative")
    msg["Subject"] = f"📈 Top {TOP_N} Momentum Stocks — {today_str}"
    msg["From"]    = EMAIL_FROM
    msg["To"]      = EMAIL_TO
    msg.attach(MIMEText("\n".join(lines), "plain"))
    msg.attach(MIMEText(html_body, "html"))

    with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as srv:
        srv.ehlo(); srv.starttls()
        srv.login(SMTP_USER, SMTP_PASS)
        srv.sendmail(EMAIL_FROM, [r.strip() for r in EMAIL_TO.split(",")], msg.as_string())
    log.info(f"Email sent to {EMAIL_TO}")


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

def run():
    log.info("╔══════════════════════════════════════════════════╗")
    log.info("║  Momentum Watchlist — Full Auto-Discovery Mode   ║")
    log.info("╚══════════════════════════════════════════════════╝")

    # 1. Discover
    candidates = discover_candidates()

    # 2. Liquidity filter
    log.info(f"\n=== STAGE 2: Liquidity filter ({len(candidates)} candidates) ===")
    liquid = []
    for i, t in enumerate(candidates):
        if len(liquid) >= MAX_CANDIDATES:
            break
        if passes_liquidity(t):
            liquid.append(t)
        if (i + 1) % 25 == 0:
            log.info(f"  Checked {i+1}/{len(candidates)} → {len(liquid)} passed")
        time.sleep(0.1)
    log.info(f"  → {len(liquid)} liquid candidates")

    # 3+4. Score
    log.info(f"\n=== STAGE 3+4: Scoring {len(liquid)} candidates ===")
    results = []
    for i, ticker in enumerate(liquid):
        data = fetch_ticker_data(ticker)
        if data:
            results.append(score_stock(data))
        if (i + 1) % 20 == 0:
            log.info(f"  [{i+1}/{len(liquid)}] scored: {len(results)}")
        time.sleep(0.2)

    if not results:
        log.error("No stocks scored — check network connectivity and API access")
        return

    results.sort(key=lambda x: x["overall_score"], reverse=True)
    top = results[:TOP_N]
    log.info(f"\nTop {TOP_N}: {[(s['ticker'], s['overall_score']) for s in top]}")

    scan_stats = {
        "universe": len(candidates),
        "liquid":   len(liquid),
        "scanned":  len(results),
        "sources":  "S&P 500 + NASDAQ-100 + QQQ/XLK/ARKK/IJR holdings + day movers",
    }

    # 5. Enrich
    log.info("\n=== STAGE 5a: Claude AI enrichment with web search ===")
    top = enrich_with_claude(top)

    # 5b. Trade plans
    log.info("\n=== STAGE 5b: Generating trade plans (entry/stop/targets/volume) ===")
    top = generate_trade_plans(top)

    # 6. Email
    log.info("\n=== STAGE 6: Sending email ===")
    send_email(top, scan_stats)
    log.info("\n✓ Complete.")


if __name__ == "__main__":
    run()
