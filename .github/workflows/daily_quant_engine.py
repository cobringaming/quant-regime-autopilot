import os
import json
import time
import datetime
import warnings
import smtplib
from email.message import EmailMessage
import requests
import xml.etree.ElementTree as ET
from urllib.parse import quote
import numpy as np
import pandas as pd
import yfinance as yf
from google import genai
from google.genai import types

warnings.filterwarnings('ignore')

# =====================================================================
# 1. CONFIGURATION & SECRETS (Set Here for Colab, or via GitHub Secrets)
# =====================================================================
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "YOUR_GEMINI_API_KEY")
FINNHUB_API_KEY = os.environ.get("FINNHUB_API_KEY", "YOUR_FINNHUB_API_KEY")

# PHONE PUSH NOTIFICATION TOPIC (Download 'ntfy' app on iOS/Android & subscribe to this exact topic)
NTFY_TOPIC = os.environ.get("NTFY_TOPIC", "reid-quant-trades-8492")

# OPTIONAL: Direct SMS via Gmail App Password + Carrier Gateway (e.g. 8035551234@vtext.com)
SMTP_EMAIL = os.environ.get("SMTP_EMAIL", "")          # Your Gmail address (optional)
SMTP_APP_PASSWORD = os.environ.get("SMTP_PASSWORD", "") # 16-char Google App Password (optional)
SMS_GATEWAY = os.environ.get("SMS_GATEWAY", "")         # e.g., "1234567890@vtext.com"

ACCOUNT_EQUITY = 100000.0
MAX_RISK_PER_TRADE = 0.01
FUNDAMENTALS_FILE = "market_fundamentals.csv"
FUNDAMENTALS_CACHE_DAYS = 7

SYSTEM_EV_BENCHMARKS = {
    'SYS_1_DAY3_SHORT':     {'win_prob_pct': 52.85, 'avg_win_pct': 4.05, 'avg_loss_pct': 2.07, 'ev_pct': 1.16, 'payoff_ratio': 1.96},
    'SYS_2A_DARVAS_BOX':    {'win_prob_pct': 48.50, 'avg_win_pct': 6.80, 'avg_loss_pct': 2.40, 'ev_pct': 2.06, 'payoff_ratio': 2.83},
    'SYS_2B_TTM_SQUEEZE':   {'win_prob_pct': 54.20, 'avg_win_pct': 5.55, 'avg_loss_pct': 2.50, 'ev_pct': 1.86, 'payoff_ratio': 2.22},
    'SYS_3_PEAD_DRIFT':     {'win_prob_pct': 64.00, 'avg_win_pct': 5.20, 'avg_loss_pct': 2.30, 'ev_pct': 2.50, 'payoff_ratio': 2.26},
    'SYS_4_MACRO_ROTATION': {'win_prob_pct': 62.00, 'avg_win_pct': 4.50, 'avg_loss_pct': 2.10, 'ev_pct': 1.99, 'payoff_ratio': 2.14},
    'SYS_5B_FILTERED_ORB':  {'win_prob_pct': 43.00, 'avg_win_pct': 3.80, 'avg_loss_pct': 1.05, 'ev_pct': 1.04, 'payoff_ratio': 3.62},
    'SYS_6_VRP_OPTIONS':    {'win_prob_pct': 75.00, 'avg_win_pct': 5.50, 'avg_loss_pct': 5.90, 'ev_pct': 2.65, 'payoff_ratio': 3.10},
}

# =====================================================================
# 2. FREE AUTONOMOUS GOOGLE NEWS + YAHOO FINANCE RESEARCH ENGINE
# =====================================================================
def fetch_google_news_rss(query, max_items=4):
    headlines = []
    try:
        url = f"https://news.google.com/rss/search?q={quote(query)}&hl=en-US&gl=US&ceid=US:en"
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        resp = requests.get(url, headers=headers, timeout=6)
        if resp.status_code == 200:
            root = ET.fromstring(resp.content)
            for item in root.findall('.//item')[:max_items]:
                title = item.find('title')
                if title is not None and title.text:
                    headlines.append(title.text.strip())
    except Exception:
        pass
    return headlines

def fetch_stock_catalyst_news(ticker):
    combined = []
    try:
        yf_news = yf.Ticker(ticker).news
        if yf_news:
            for art in yf_news[:3]:
                t_str = art.get('title') or art.get('content', {}).get('title', '')
                if t_str:
                    combined.append(t_str.strip())
    except Exception:
        pass

    combined.extend(fetch_google_news_rss(f"{ticker} stock news when:3d", max_items=3))

    if FINNHUB_API_KEY and FINNHUB_API_KEY != "YOUR_FINNHUB_API_KEY":
        try:
            end_s = datetime.date.today().strftime('%Y-%m-%d')
            start_s = (datetime.date.today() - datetime.timedelta(days=4)).strftime('%Y-%m-%d')
            u = f"https://finnhub.io/api/v1/company-news?symbol={ticker}&from={start_s}&to={end_s}&token={FINNHUB_API_KEY}"
            r = requests.get(u, timeout=5)
            if r.status_code == 200:
                for a in r.json()[:3]:
                    if a.get('headline'):
                        combined.append(a['headline'].strip())
            time.sleep(0.25)
        except Exception:
            pass

    unique = list(dict.fromkeys([h for h in combined if h]))
    return " | ".join(unique[:5]) if unique else "Technical setup; no major corporate press release in last 72h."

def pre_classify_catalyst(headlines_text, system_id):
    text_low = headlines_text.lower()
    if any(k in text_low for k in ['merger', 'acquire', 'acquisition', 'buyout', 'tender offer', 'special dividend', 'takeover']):
        return "CLASS_A_HARD_FLOOR"
    if any(k in text_low for k in ['public offering', 'secondary offering', 'convertible notes', 'reverse stock split', 'dilution']):
        return "CLASS_B_DILUTION"
    if any(k in text_low for k in ['q1', 'q2', 'q3', 'q4', 'earnings', 'eps', 'revenue beat', 'tops estimates', 'misses estimates']):
        return "CLASS_C_EARNINGS_SURPRISE"
    if system_id == "SYS_1_DAY3_SHORT" or any(k in text_low for k in ['guidance', 'outlook', 'restructuring', 'clinical', 'trial', 'fda', 'analyst', 'price target', 'upgrade']):
        return "CLASS_D_SOFT_CATALYST" if system_id == "SYS_1_DAY3_SHORT" else "TECHNICAL_COMPRESSION"
    return "TECHNICAL_COMPRESSION"

# =====================================================================
# 3. FETCH S&P 500 UNIVERSE & FUNDAMENTALS
# =====================================================================
def get_sp500_tickers():
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    html_content = requests.get(url, headers=headers).text
    sp500 = pd.read_html(html_content)[0]['Symbol'].tolist()
    return [ticker.replace('.', '-') for ticker in sp500]

def get_fundamentals(tickers):
    if os.path.exists(FUNDAMENTALS_FILE):
        age_days = (time.time() - os.path.getmtime(FUNDAMENTALS_FILE)) / 86400
        if age_days <= FUNDAMENTALS_CACHE_DAYS:
            print(f"[1/6] Loaded cached '{FUNDAMENTALS_FILE}' ({age_days:.1f} days old).")
            return pd.read_csv(FUNDAMENTALS_FILE)

    print(f"[1/6] Building '{FUNDAMENTALS_FILE}' for {len(tickers)} S&P 500 companies...")
    records = []
    for t in tickers:
        try:
            info = yf.Ticker(t).info
            records.append({
                'Ticker': t,
                'Sector': info.get('sector', 'Unknown'),
                'Industry': info.get('industry', 'Unknown'),
                'Market Cap': info.get('marketCap', None),
                'Beta (Volatility Risk)': info.get('beta', None),
                'Trailing P/E': info.get('trailingPE', None),
                'Forward P/E': info.get('forwardPE', None),
                'Profit Margin': info.get('profitMargins', None),
                'Short Ratio': info.get('shortRatio', None)
            })
        except Exception:
            pass
    df_f = pd.DataFrame(records)
    df_f.to_csv(FUNDAMENTALS_FILE, index=False)
    return df_f

# =====================================================================
# 4. SYSTEM 4 & 6: MACRO REGIME & VOLATILITY CONTROLLER
# =====================================================================
def evaluate_macro_regime():
    print("[2/6] Evaluating Macro Regime (SPY, QQQ, RSP, TLT, BIL, GLD, ^VIX, ^VIX3M) & Sector ETFs...")
    macro_tickers = ['SPY', 'QQQ', 'RSP', 'TLT', 'BIL', 'GLD', '^VIX', '^VIX3M']
    sector_tickers = ['XLB', 'XLE', 'XLF', 'XLI', 'XLK', 'XLP', 'XLU', 'XLV', 'XLY']

    raw = yf.download(macro_tickers + sector_tickers, period="2y", interval="1d", auto_adjust=True, progress=False)
    data = raw['Close'] if isinstance(raw.columns, pd.MultiIndex) else raw
    data = data.ffill().dropna(how='all')

    spy_close = float(data['SPY'].iloc[-1])
    spy_sma200 = float(data['SPY'].rolling(200).mean().iloc[-1])
    spy_bull = bool(spy_close >= spy_sma200)

    def ret_12m(sym):
        if sym not in data.columns:
            return 0.0
        s = data[sym].dropna()
        return float((s.iloc[-1] / s.iloc[max(-252, -len(s))]) - 1.0)

    r12 = {sym: round(ret_12m(sym) * 100, 2) for sym in ['SPY', 'QQQ', 'RSP', 'TLT', 'BIL', 'GLD']}
    rets = data[['SPY', 'TLT']].pct_change()
    spy_tlt_corr = float(rets['SPY'].tail(90).corr(rets['TLT'].tail(90)))

    vix = float(data['^VIX'].dropna().iloc[-1])
    vix_sma10 = float(data['^VIX'].dropna().rolling(10).mean().iloc[-1])
    vix3m = float(data['^VIX3M'].dropna().iloc[-1]) if '^VIX3M' in data.columns and len(data['^VIX3M'].dropna()) > 0 else vix * 1.05
    vix_ratio = vix / vix3m
    term_structure = "CONTANGO" if vix_ratio < 1.0 else "BACKWARDATION"

    if max(r12['SPY'], r12['RSP']) > r12['BIL'] and spy_bull:
        leader = max(['SPY', 'QQQ', 'RSP'], key=lambda s: r12[s])
        core_alloc = f"100% Equities ({leader} Momentum Leader at +{r12[leader]}% 12M)"
        core_ticker = leader
    elif spy_tlt_corr < -0.40 and r12['TLT'] > r12['BIL']:
        core_alloc = "100% Long-Duration Treasury Bonds (TLT Deflation Hedge)"
        core_ticker = "TLT"
    else:
        core_alloc = "50% T-Bills (BIL) + 50% Gold (GLD) (Inflation / Positive SPY-TLT Correlation Shield)"
        core_ticker = "BIL / GLD"

    if vix > 25.0:
        vrp_play = "Mode 6B (Volatility Crush): Sell 30-45 DTE SPY Put Credit Spreads 5%-7% OTM; close at 50% profit or 21 DTE."
    elif vix_ratio < 1.0:
        vrp_play = "Mode 6A (Contango Harvesting): Sell 30-45 DTE SPY/QQQ Put Credit Spreads or Iron Condors (25-35 delta, credit >= 1/3 width); close at 50% profit or 21 DTE."
    else:
        vrp_play = "Mode 6C (Backwardation Alert): Pause standard short puts; use VXX puts or wait for VIX/VIX3M < 1.0."

    sec_rets = {s: round(ret_12m(s) * 100, 2) for s in sector_tickers if s in data.columns}
    sorted_secs = sorted(sec_rets.items(), key=lambda x: x[1], reverse=True)
    top_3_sectors = [f"{s} ({r}%)" for s, r in sorted_secs[:3]]
    bot_3_sectors = [f"{s} ({r}%)" for s, r in sorted_secs[-3:]]

    vix_kelly_scalar = float(np.clip(1.0 - ((vix - vix_sma10) / vix_sma10), 0.50, 1.25))

    macro_headlines = []
    macro_headlines.extend(fetch_google_news_rss("Federal Reserve FOMC interest rates inflation when:3d", 3))
    macro_headlines.extend(fetch_google_news_rss("S&P 500 stock market outlook economy when:2d", 3))

    return {
        'scan_date': data.index[-1].strftime('%Y-%m-%d'),
        'spy_close': round(spy_close, 2),
        'spy_sma200': round(spy_sma200, 2),
        'spy_bull_trend': spy_bull,
        'returns_12m_pct': r12,
        'spy_tlt_corr_90d': round(spy_tlt_corr, 3),
        'vix_spot': round(vix, 2),
        'vix3m_spot': round(vix3m, 2),
        'vix_term_ratio': round(vix_ratio, 3),
        'vix_term_structure': term_structure,
        'system_4_core_allocation': core_alloc,
        'system_4_core_ticker': core_ticker,
        'system_6_vrp_options_action': vrp_play,
        'top_3_momentum_sectors': top_3_sectors,
        'bottom_3_laggard_sectors': bot_3_sectors,
        'vix_kelly_scalar': round(vix_kelly_scalar, 3),
        'live_macro_news_headlines': " | ".join(macro_headlines[:6])
    }

# =====================================================================
# 5. MULTI-SYSTEM SCANNER
# =====================================================================
def wilder_atr(h, l, c, period=14):
    prev_c = c.shift(1)
    tr = pd.concat([h - l, (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False).mean()

def safe_float(val, default=0.0):
    try:
        if pd.isna(val) or np.isnan(val) or np.isinf(val):
            return default
        return float(val)
    except Exception:
        return default

def run_equity_scanner(tickers, df_fund, macro):
    print(f"[3/6] Scanning {len(tickers)} S&P 500 stocks across all 6 Quantitative Systems...")
    raw = yf.download(tickers, period="6mo", interval="1d", auto_adjust=True, progress=False)
    opens, highs, lows, closes, vols = raw['Open'], raw['High'], raw['Low'], raw['Close'], raw['Volume']

    df_fund = df_fund.copy()
    df_fund['EAR_PE'] = pd.to_numeric(df_fund['Trailing P/E'], errors='coerce') / pd.to_numeric(df_fund['Forward P/E'], errors='coerce').replace(0, np.nan)
    f_map = df_fund.set_index('Ticker').to_dict(orient='index')

    strict_candidates = []
    watchlist_backups = []
    excluded_squeeze_risk = []
    vix_scalar = macro['vix_kelly_scalar']

    spy_p = macro['spy_close']
    strict_candidates.append({
        'ticker': macro['system_4_core_ticker'],
        'sector': 'Macro / Multi-Asset ETF',
        'system_id': 'SYS_4_MACRO_ROTATION',
        'direction': 'LONG',
        'catalyst_class': 'TECHNICAL_COMPRESSION',
        'entry_trigger_price': spy_p,
        'atr_14': round(spy_p * 0.012, 2),
        'intraday_catastrophe_stop': round(macro['spy_sma200'], 2),
        'eod_355pm_stop_rule': f"Rebalance monthly; hold {macro['system_4_core_allocation']} while SPY >= 200-SMA.",
        'profit_target_or_time_exit': f"Top Sectors: {', '.join(macro['top_3_momentum_sectors'])}",
        'quarter_kelly_allocation_pct': 25.0,
        'recommended_shares_or_contracts': max(1, int((ACCOUNT_EQUITY * 0.25) / spy_p)),
        'quant_telemetry': f"12M SPY: {macro['returns_12m_pct']['SPY']}%, RSP: {macro['returns_12m_pct']['RSP']}%, 90d SPY-TLT Corr: {macro['spy_tlt_corr_90d']}",
        'recent_news_headlines': macro['live_macro_news_headlines']
    })

    put_short_strike = round(spy_p * 0.94, 0)
    put_long_strike = put_short_strike - 5.0
    strict_candidates.append({
        'ticker': 'SPY (30-45 DTE Options)',
        'sector': 'Index Options (VRP)',
        'system_id': 'SYS_6_VRP_OPTIONS',
        'direction': 'CREDIT_SPREAD',
        'catalyst_class': 'TECHNICAL_COMPRESSION',
        'entry_trigger_price': put_short_strike,
        'atr_14': round(spy_p * 0.015, 2),
        'intraday_catastrophe_stop': put_long_strike,
        'eod_355pm_stop_rule': f"Sell ${put_short_strike}/${put_long_strike} Put Credit Spread (collect >= $1.65 credit).",
        'profit_target_or_time_exit': "Mechanical exit at 50% max profit or 21 DTE.",
        'quarter_kelly_allocation_pct': 7.61,
        'recommended_shares_or_contracts': max(1, int((ACCOUNT_EQUITY * 0.01) / 335.0)),
        'quant_telemetry': f"Spot VIX: {macro['vix_spot']} ({macro['vix_term_structure']}, VIX/VIX3M={macro['vix_term_ratio']})",
        'recent_news_headlines': macro['system_6_vrp_options_action']
    })

    for t in tickers:
        if t not in closes.columns:
            continue
        c = closes[t].dropna()
        if len(c) < 55:
            continue
        o, h, l, v = opens[t].reindex(c.index), highs[t].reindex(c.index), lows[t].reindex(c.index), vols[t].reindex(c.index)

        f = f_map.get(t, {})
        beta = safe_float(f.get('Beta (Volatility Risk)'), 1.1)
        dtc = safe_float(f.get('Short Ratio'), 2.5)
        ear_pe = safe_float(f.get('EAR_PE'), 1.15)
        sector = str(f.get('Sector', 'Unknown'))

        atr14 = safe_float(wilder_atr(h, l, c, 14).iloc[-1], 0.0)
        last_c = safe_float(c.iloc[-1], 0.0)
        vol_sma20 = safe_float(v.rolling(20).mean().iloc[-1], 0.0)
        vol_sma30 = safe_float(v.rolling(30).mean().iloc[-1], 0.0)

        # Enforce Non-Penny Stock Gate (>= $5.00)
        if atr14 <= 0 or last_c < 5.0 or vol_sma20 <= 0:
            continue

        # SYSTEM 1: Day-3 Exhaustion Short
        day1_gap = safe_float((o.iloc[-2] - c.iloc[-3]) / c.iloc[-3])
        day1_ret = safe_float((c.iloc[-2] - c.iloc[-3]) / c.iloc[-3])
        day1_shock = max(day1_gap, day1_ret)

        if day1_shock >= 0.035:
            if dtc >= 3.5:
                excluded_squeeze_risk.append({
                    'ticker': t,
                    'exclusion_reason': f"Day-1 surged +{day1_shock*100:.1f}%, but Short Ratio (DTC={dtc:.2f}d) >= 3.5d squeeze limit."
                })
            elif beta >= 1.05:
                v_d1, v_d2 = safe_float(v.iloc[-2], 1.0), safe_float(v.iloc[-1], 1.0)
                wick = (safe_float(h.iloc[-1]) - last_c) / max(safe_float(h.iloc[-1]) - safe_float(l.iloc[-1]), 1e-4)
                v_score = (v_d1 / max(vol_sma30, 1.0)) * max(0.1, (1.0 - (v_d2 / max(v_d1, 1.0)))) * max(0.2, wick)

                w_alloc = 0.0719 * vix_scalar
                stop_dist = max(1.8 * atr14, 0.05 * last_c)
                rec_sh = max(1, min(int((ACCOUNT_EQUITY * w_alloc) / last_c), int((ACCOUNT_EQUITY * MAX_RISK_PER_TRADE) / stop_dist)))

                ticket = {
                    'ticker': t,
                    'sector': sector,
                    'system_id': 'SYS_1_DAY3_SHORT',
                    'direction': 'SHORT',
                    'entry_trigger_price': round(last_c, 2),
                    'atr_14': round(atr14, 2),
                    'intraday_catastrophe_stop': round(last_c + stop_dist, 2),
                    'eod_355pm_stop_rule': f"Cover at 3:55 PM ET if > ${last_c*1.025:.2f} (+2.5%); if green, lock in 70% & trail 30%.",
                    'profit_target_or_time_exit': f"Target -4.05% mean-reversion drop ( ~${last_c*0.96:.2f} )",
                    'quarter_kelly_allocation_pct': round(w_alloc * 100, 2),
                    'recommended_shares_or_contracts': rec_sh,
                    'quant_telemetry': f"Day-1 Surge: +{day1_shock*100:.1f}%, V_score: {v_score:.2f}, Beta: {beta:.2f}, DTC: {dtc:.2f}d",
                    '_rank_score': day1_shock * 100 + v_score
                }
                if day1_shock >= 0.045 and v_score >= 0.15:
                    strict_candidates.append(ticket)
                else:
                    watchlist_backups.append(ticket)

        # SYSTEM 2A: Darvas Box Breakout
        sma50 = safe_float(c.rolling(50).mean().iloc[-1], last_c)
        box_top = safe_float(h.iloc[-11:-1].max(), last_c)
        box_bot = safe_float(l.iloc[-11:-1].min(), last_c * 0.95)
        box_rng = (box_top - box_bot) / max(box_bot, 1e-4)
        vol_ratio = safe_float(v.iloc[-1]) / vol_sma20

        if last_c > sma50 and 0.015 <= box_rng <= 0.12:
            w_alloc = 0.0758 * vix_scalar
            stop_dist = max(abs(last_c - box_bot), 1.5 * atr14)
            rec_sh = max(1, min(int((ACCOUNT_EQUITY * w_alloc) / last_c), int((ACCOUNT_EQUITY * MAX_RISK_PER_TRADE) / stop_dist)))

            if last_c > box_top and vol_ratio >= 1.25 and ear_pe >= 1.15:
                strict_candidates.append({
                    'ticker': t,
                    'sector': sector,
                    'system_id': 'SYS_2A_DARVAS_BOX',
                    'direction': 'LONG',
                    'entry_trigger_price': round(last_c, 2),
                    'atr_14': round(atr14, 2),
                    'intraday_catastrophe_stop': round(box_bot, 2),
                    'eod_355pm_stop_rule': f"Ratchet stop to each new confirmed Darvas Box Bottom (initial ${box_bot:.2f}).",
                    'profit_target_or_time_exit': f"Breakout above ${box_top:.2f}; trail via Darvas state machine",
                    'quarter_kelly_allocation_pct': round(w_alloc * 100, 2),
                    'recommended_shares_or_contracts': rec_sh,
                    'quant_telemetry': f"Darvas Breakout [${box_bot:.2f}-${box_top:.2f}], Vol: {vol_ratio:.2f}x avg, EAR_PE: {ear_pe:.2f}",
                    '_rank_score': vol_ratio * ear_pe
                })
            elif last_c >= 0.985 * box_top and ear_pe >= 1.35:
                watchlist_backups.append({
                    'ticker': t,
                    'sector': sector,
                    'system_id': 'SYS_2A_DARVAS_BOX',
                    'direction': 'LONG',
                    'entry_trigger_price': round(box_top + 0.05, 2),
                    'atr_14': round(atr14, 2),
                    'intraday_catastrophe_stop': round(box_bot, 2),
                    'eod_355pm_stop_rule': f"Pre-Breakout Darvas Coil: Trigger Buy-Stop above ${box_top:.2f} on >=1.5x volume.",
                    'profit_target_or_time_exit': f"Box Range {(box_rng*100):.1f}% compression coil",
                    'quarter_kelly_allocation_pct': round(w_alloc * 100, 2),
                    'recommended_shares_or_contracts': rec_sh,
                    'quant_telemetry': f"Coiling 1% below Box Top ${box_top:.2f} (Bot ${box_bot:.2f}), EAR_PE: {ear_pe:.2f}",
                    '_rank_score': ear_pe
                })

        # SYSTEM 2B: TTM Volatility Squeeze
        basis = c.rolling(20).mean()
        sd20 = c.rolling(20).std()
        atr20 = wilder_atr(h, l, c, 20)
        bb_up, bb_dn = basis + 2.0 * sd20, basis - 2.0 * sd20
        kc_up, kc_dn = basis + 1.5 * atr20, basis - 1.5 * atr20

        sqz_prev = bool((bb_up.iloc[-2] < kc_up.iloc[-2]) and (bb_dn.iloc[-2] > kc_dn.iloc[-2]))
        sqz_now = bool((bb_up.iloc[-1] < kc_up.iloc[-1]) and (bb_dn.iloc[-1] > kc_dn.iloc[-1]))
        sqz_fired = sqz_prev and not sqz_now

        mid = (((h.rolling(20).max() + l.rolling(20).min()) / 2.0) + basis) / 2.0
        mom_now = safe_float((c - mid).iloc[-1])
        mom_prev = safe_float((c - mid).iloc[-2])

        if (sqz_fired or sqz_now) and mom_now > 0 and mom_now > mom_prev and last_c > sma50:
            w_alloc = 0.0839 * vix_scalar
            stop_dist = 2.0 * atr14
            rec_sh = max(1, min(int((ACCOUNT_EQUITY * w_alloc) / last_c), int((ACCOUNT_EQUITY * MAX_RISK_PER_TRADE) / stop_dist)))
            t_obj = {
                'ticker': t,
                'sector': sector,
                'system_id': 'SYS_2B_TTM_SQUEEZE',
                'direction': 'LONG',
                'entry_trigger_price': round(last_c, 2),
                'atr_14': round(atr14, 2),
                'intraday_catastrophe_stop': round(last_c - stop_dist, 2),
                'eod_355pm_stop_rule': "Trail with 22-period 3.0x ATR Chandelier Exit.",
                'profit_target_or_time_exit': "Squeeze Expansion Target (+2.5x to +3.0x ATR)",
                'quarter_kelly_allocation_pct': round(w_alloc * 100, 2),
                'recommended_shares_or_contracts': rec_sh,
                'quant_telemetry': f"TTM Squeeze {'FIRED' if sqz_fired else 'COILING'}, Vol: {vol_ratio:.2f}x, EAR_PE: {ear_pe:.2f}",
                '_rank_score': vol_ratio + (2.0 if sqz_fired else 0.5)
            }
            if sqz_fired and vol_ratio >= 1.2:
                strict_candidates.append(t_obj)
            else:
                watchlist_backups.append(t_obj)

        # SYSTEM 3: PEAD / Fresh Catalyst Surge
        today_ret = safe_float((last_c - c.iloc[-2]) / c.iloc[-2])
        if today_ret >= 0.04 and vol_ratio >= 1.75:
            w_alloc = 0.10 * vix_scalar
            stop_dist = 1.5 * atr14
            rec_sh = max(1, min(int((ACCOUNT_EQUITY * w_alloc) / last_c), int((ACCOUNT_EQUITY * MAX_RISK_PER_TRADE) / stop_dist)))
            strict_candidates.append({
                'ticker': t,
                'sector': sector,
                'system_id': 'SYS_3_PEAD_DRIFT',
                'direction': 'LONG',
                'entry_trigger_price': round(last_c, 2),
                'atr_14': round(atr14, 2),
                'intraday_catastrophe_stop': round(last_c - stop_dist, 2),
                'eod_355pm_stop_rule': "Exit Day 5 if cumulative alpha vs SPY <= 0; hard time exit Trading Day +7.",
                'profit_target_or_time_exit': "Hold Trading Days +1 to +7 (PEAD front-loaded drift window)",
                'quarter_kelly_allocation_pct': round(w_alloc * 100, 2),
                'recommended_shares_or_contracts': rec_sh,
                'quant_telemetry': f"Today Surge: +{today_ret*100:.1f}%, Vol: {vol_ratio:.2f}x avg, DTC: {dtc:.2f}d",
                '_rank_score': today_ret * 100
            })

    watchlist_backups.sort(key=lambda x: x.get('_rank_score', 0), reverse=True)
    final_candidates = strict_candidates[:18]
    seen_tickers = {item['ticker'] for item in final_candidates}

    for b in watchlist_backups:
        if len(final_candidates) >= 20:
            break
        if b['ticker'] not in seen_tickers:
            final_candidates.append(b)
            seen_tickers.add(b['ticker'])

    print(f"[4/6] Scraping Free Live News (Google News RSS + Yahoo Finance) for {len(final_candidates)} candidates...")
    for item in final_candidates:
        item.pop('_rank_score', None)
        if 'recent_news_headlines' not in item:
            headlines = fetch_stock_catalyst_news(item['ticker'])
            item['recent_news_headlines'] = headlines
            item['catalyst_class'] = pre_classify_catalyst(headlines, item['system_id'])
        item['causal_rationale'] = f"{item['quant_telemetry']} | News: {item['recent_news_headlines'][:140]}"

    return final_candidates, excluded_squeeze_risk

# =====================================================================
# 6. SINGLE-CALL AI REASONING & PHONE ALERT DISPATCHER
# =====================================================================
def generate_ai_mobile_briefing(macro, candidates):
    print("[5/6] Running Single-Batch AI Reasoning for Mobile Alert...")
    top_lines = [
        f"Regime: {macro['system_4_core_allocation']}",
        f"VIX: {macro['vix_spot']} ({macro['vix_term_structure']}) | SPY-TLT Corr: {macro['spy_tlt_corr_90d']}"
    ]
    for c in candidates[:5]:
        if c.get('catalyst_class') != 'CLASS_A_HARD_FLOOR' or c.get('direction') != 'SHORT':
            top_lines.append(
                f"• {c['ticker']} ({c['system_id']}): {c['direction']} @ ${c['entry_trigger_price']} | "
                f"EV: +${c.get('expected_profit_dollars', 0):.0f} ({c.get('win_probability_pct', 50)}% Win) | "
                f"Stop: ${c['intraday_catastrophe_stop']} | Size: {c['recommended_shares_or_contracts']} sh"
            )
    fallback_summary = "\n".join(top_lines)

    if not GEMINI_API_KEY or GEMINI_API_KEY == "YOUR_GEMINI_API_KEY":
        return fallback_summary

    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        prompt = f"""
        You are an institutional Quantitative Portfolio Manager. Analyze today's scan ({macro['scan_date']}) and write a concise, high-signal Mobile Trade Alert (under 1,200 characters) ready for my phone lock screen.
        Rules:
        1. Strictly REJECT any System 1 Short that has Class A Hard Floor news (merger, buyout, dividend).
        2. State the Macro Regime Allocation & System 6 Options Play in 2 bullet points.
        3. List the Top 4 Highest-Expected-Value (EV $) Stock Trades with exact Ticker, System, Direction, Entry Price, Expected Return ($ and %), Win Probability %, Stop Price, 3:55 PM ET rule, Share Count, and 1-sentence news catalyst verification.

        Macro Telemetry: {json.dumps(macro)}
        Candidates: {json.dumps(candidates[:12])}
        """
        resp = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.1)
        )
        return resp.text.strip() if resp.text else fallback_summary
    except Exception as e:
        print(f"AI Summary fallback triggered ({e})")
        return fallback_summary

def send_mobile_alerts(scan_date, mobile_briefing):
    print(f"[6/6] Sending Instant Mobile Push Notification to ntfy topic '{NTFY_TOPIC}'...")
    if NTFY_TOPIC:
        try:
            requests.post(
                f"https://ntfy.sh/{NTFY_TOPIC}",
                data=mobile_briefing.encode('utf-8'),
                headers={
                    "Title": f"QuantRegime Trade Decisions ({scan_date})",
                    "Priority": "high",
                    "Tags": "chart_with_upwards_trend,moneybag"
                },
                timeout=8
            )
            print(f"-> Successfully pushed instant alert to https://ntfy.sh/{NTFY_TOPIC}")
        except Exception as e:
            print(f"ntfy push error: {e}")

    if SMTP_EMAIL and SMTP_APP_PASSWORD and SMS_GATEWAY:
        try:
            msg = EmailMessage()
            msg.set_content(mobile_briefing[:450])
            msg['Subject'] = f"Quant Trades {scan_date}"
            msg['From'] = SMTP_EMAIL
            msg['To'] = SMS_GATEWAY
            with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
                server.login(SMTP_EMAIL, SMTP_APP_PASSWORD)
                server.send_message(msg)
            print("-> Successfully sent SMS text via carrier gateway!")
        except Exception as e:
            print(f"SMS gateway error: {e}")

# =====================================================================
# MAIN EXECUTION (PROPERLY INDENTED)
# =====================================================================
if __name__ == "__main__":
    sp500 = get_sp500_tickers()
    df_fund = get_fundamentals(sp500)
    macro_telemetry = evaluate_macro_regime()
    candidates_list, excluded_list = run_equity_scanner(sp500, df_fund, macro_telemetry)

    # -----------------------------------------------------------------
    # ENRICH CANDIDATES WITH SIMPLE MODE EV METRICS & RESEARCH PANEL FEED
    # -----------------------------------------------------------------
    print("[4.5/6] Calculating Simple Mode Expected Returns (EV) & Building Research Panel Feed...")
    research_panel_articles = []

    for m_headline in macro_telemetry.get('live_macro_news_headlines', '').split(' | '):
        if m_headline.strip():
            research_panel_articles.append({
                'date': macro_telemetry['scan_date'],
                'ticker': 'MACRO / SPY',
                'headline': m_headline.strip(),
                'catalyst_class': 'MACRO_REGIME',
                'system_impact': f"Supports {macro_telemetry['system_4_core_allocation']}"
            })

    for item in candidates_list:
        sys_key = item.get('system_id', 'SYS_2A_DARVAS_BOX')
        bench = SYSTEM_EV_BENCHMARKS.get(sys_key, SYSTEM_EV_BENCHMARKS['SYS_2A_DARVAS_BOX'])
        notional = round(item['entry_trigger_price'] * item['recommended_shares_or_contracts'], 2)
        ev_dollars = round(notional * (bench['ev_pct'] / 100.0), 2)
        target_win_dollars = round(notional * (bench['avg_win_pct'] / 100.0), 2)
        max_risk_dollars = round(abs(item['entry_trigger_price'] - item['intraday_catastrophe_stop']) * item['recommended_shares_or_contracts'], 2)

        # Attach Simple Mode metrics directly to each candidate ticket
        item['notional_dollars'] = notional
        item['win_probability_pct'] = bench['win_prob_pct']
        item['expected_return_ev_pct'] = bench['ev_pct']
        item['expected_profit_dollars'] = ev_dollars
        item['target_win_dollars'] = target_win_dollars
        item['max_stop_risk_dollars'] = max_risk_dollars
        item['payoff_ratio'] = bench['payoff_ratio']
        item['plain_english_action'] = (
            f"{item['direction']} {item['recommended_shares_or_contracts']} sh of {item['ticker']} "
            f"@ ${item['entry_trigger_price']:.2f} (Stop: ${item['intraday_catastrophe_stop']:.2f})"
        )
        item['detailed_reasoning_dropdown'] = {
            'causal_thesis': f"Catalyst Class: {item.get('catalyst_class', 'TECHNICAL_COMPRESSION')}. {item.get('recent_news_headlines', '')[:180]}",
            'equation_telemetry': item.get('quant_telemetry', ''),
            'execution_plan': f"{item.get('eod_355pm_stop_rule', '')} | {item.get('profit_target_or_time_exit', '')}"
        }

        # Split headlines into structured cards for the Research Panel tab
        for headline_str in item.get('recent_news_headlines', '').split(' | '):
            if headline_str.strip():
                research_panel_articles.append({
                    'date': macro_telemetry['scan_date'],
                    'ticker': item['ticker'],
                    'headline': headline_str.strip(),
                    'catalyst_class': item.get('catalyst_class', 'TECHNICAL_COMPRESSION'),
                    'system_impact': (
                        f"Routes {item['ticker']} to {item['system_id']} ({item['direction']}) | "
                        f"Win Prob: {bench['win_prob_pct']}% | EV: +${ev_dollars:.2f} (+{bench['ev_pct']}%)"
                    )
                })

    # Sort candidates by Expected Dollar Profit (highest first) for Simple Mode
    candidates_list.sort(key=lambda x: x.get('expected_profit_dollars', 0), reverse=True)
    for idx, item in enumerate(candidates_list, start=1):
        item['simple_mode_rank'] = idx

    ai_mobile_briefing = generate_ai_mobile_briefing(macro_telemetry, candidates_list)

    payload = {
        "analysis_date": macro_telemetry['scan_date'],
        "account_equity": ACCOUNT_EQUITY,
        "macro_regime_telemetry": macro_telemetry,
        "ai_executive_briefing": ai_mobile_briefing,
        "candidates": candidates_list,
        "candidate_trade_tickets": candidates_list,
        "simple_mode_leaderboard": candidates_list,
        "research_panel_news": research_panel_articles,
        "excluded_tickers": excluded_list[:20]
    }

    with open("daily_quant_scan.json", "w") as f:
        json.dump(payload, f, indent=2, allow_nan=False)

    pd.DataFrame(candidates_list).to_csv("daily_quant_scan.csv", index=False)

    send_mobile_alerts(macro_telemetry['scan_date'], ai_mobile_briefing)
    print("\n=== FULL AUTOPILOT RUN COMPLETE ===")
    print(ai_mobile_briefing)
