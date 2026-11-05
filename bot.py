import os
import time
import requests
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
import pytz

# Reads your hidden Telegram credentials from GitHub Secrets
BOT_TOKEN = os.environ.get("BOT_TOKEN", "").strip()
CHAT_ID   = os.environ.get("CHAT_ID", "").strip()
EVENT_NAME = os.environ.get("GITHUB_EVENT_NAME", "")

SYMBOL = "GC=F"  # Gold Futures (GC=F)
CHECK_INTERVAL_SECONDS = 30
PRICE_OFFSET = -10.0  # Adjusts futures price to match OANDA spot
# Run for 5 hours and 45 minutes per session before handing off to the next run
MAX_RUNTIME_SECONDS = 5 * 3600 + 45 * 60


def send_telegram_Message(text):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    try:
        resp = requests.post(
            url,
            data={"chat_id": CHAT_ID, "text": text, "parse_mode": "Markdown"},
            timeout=10
        )
        if not resp.ok:
            print(f"Telegram API Error: {resp.text}")
    except Exception as e:
        print(f"Failed to send Telegram alert: {e}")


def send_telegram_alert(strategy_name, timeframe, action, direction, price, details=""):
    emoji = "🟢" if direction == "LONG" else "🔴"
    action_icon = "🚀" if action == "ENTRY" else "🏁"
    msg = (
        f"{action_icon} *[{strategy_name}] ({timeframe})*\n"
        f"{emoji} *{action} {direction}*\n"
        f"*Symbol:* `XAUUSD`\n"
        f"*Price:* `{price:.2f}`\n"
    )
    if details:
        msg += f"*Details:* {details}\n"
    msg += f"*Time (UTC):* `{datetime.now(pytz.utc).strftime('%Y-%m-%d %H:%M:%S')}`"
    print(msg)
    send_telegram_Message(msg)


def fetch_ohlc(symbol, interval, period):
    df = yf.Ticker(symbol).history(period=period, interval=interval)
    if df.empty:
        return df
    df = df[['Open', 'High', 'Low', 'Close']].dropna()
    
    # Apply price offset to match OANDA spot chart
    for col in ['Open', 'High', 'Low', 'Close']:
        df[col] = df[col] + PRICE_OFFSET
        
    df.index = df.index.tz_localize('UTC') if df.index.tz is None else df.index.tz_convert('UTC')
    return df.iloc[:-1].copy()


def fetch_2h_ohlc(symbol):
    df_1h = yf.Ticker(symbol).history(period="60d", interval="1h")
    if df_1h.empty:
        return df_1h
    df_1h = df_1h[['Open', 'High', 'Low', 'Close']].dropna()
    
    # Apply price offset to match OANDA spot chart
    for col in ['Open', 'High', 'Low', 'Close']:
        df_1h[col] = df_1h[col] + PRICE_OFFSET
        
    df_1h.index = df_1h.index.tz_localize('UTC') if df_1h.index.tz is None else df_1h.index.tz_convert('UTC')
    df_2h = df_1h.resample('2h', closed='left', label='left').agg({
        'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'
    }).dropna()
    return df_2h.iloc[:-1].copy()


# ----------------- STRATEGY 1: ICT (5m) -----------------
class ICTMS:
    def __init__(self):
        self.lastPrice = self.midPrice = self.prevPrice = None
        self.lastIndex = self.midIndex = self.prevIndex = None
        self.isCrossed = False

    def query_patterns(self, is_swing_high):
        if None in (self.prevPrice, self.midPrice, self.lastPrice):
            return False
        if is_swing_high:
            return self.prevPrice < self.midPrice and self.midPrice >= self.lastPrice
        return self.prevPrice > self.midPrice and self.midPrice <= self.lastPrice

    def update_pattern(self, price, idx):
        self.isCrossed = False
        self.prevPrice, self.midPrice, self.lastPrice = self.midPrice, self.lastPrice, price
        self.prevIndex, self.midIndex, self.lastIndex = self.midIndex, self.lastIndex, idx


def query_3bar(last_p, mid_p, prev_p, is_high):
    return (prev_p < mid_p and mid_p >= last_p) if is_high else (prev_p > mid_p and mid_p <= last_p)


def is_ict_session_allowed(ts_utc):
    t_ny = ts_utc.astimezone(pytz.timezone("America/New_York"))
    hm = t_ny.hour * 100 + t_ny.minute
    in_asia = (2000 <= hm <= 2359)
    in_ny   = (800 <= hm < 1100)
    in_skip = (900 <= hm < 1000)
    return (in_asia or in_ny) and not in_skip


def evaluate_ict_5m(df_5m, df_1d, sl_buffer=8.0, tp_points=30.0):
    df_1d = df_1d.copy()
    df_1d['ema50'] = df_1d['Close'].ewm(span=50, adjust=False).mean().shift(1)
    daily_ema_map = df_1d['ema50'].to_dict()

    def get_daily_ema(ts):
        d = ts.date()
        val = None
        for idx_ts, ema_val in daily_ema_map.items():
            if idx_ts.date() <= d and not np.isnan(ema_val):
                val = ema_val
        return val

    stLow, stHigh = ICTMS(), ICTMS()
    itLow, itHigh = ICTMS(), ICTMS()
    ltLow, ltHigh = ICTMS(), ICTMS()
    prev_cITL = prev_cITH = prev_cLTL = prev_cLTH = False
    pos, current_sl, current_tp = 0, None, None

    highs, lows, closes, timestamps = df_5m['High'].values, df_5m['Low'].values, df_5m['Close'].values, df_5m.index
    latest_events = []

    for i in range(2, len(df_5m)):
        is_last = (i == len(df_5m) - 1)
        h, l, c, ts = highs[i], lows[i], closes[i], timestamps[i]

        if pos == 1:
            if l <= current_sl:
                if is_last: latest_events.append(("EXIT", "LONG", current_sl, "Stop Loss Hit (-8 pts)"))
                pos, current_sl, current_tp = 0, None, None
            elif h >= current_tp:
                if is_last: latest_events.append(("EXIT", "LONG", current_tp, "Take Profit Hit (+30 pts)"))
                pos, current_sl, current_tp = 0, None, None
        elif pos == -1:
            if h >= current_sl:
                if is_last: latest_events.append(("EXIT", "SHORT", current_sl, "Stop Loss Hit (-8 pts)"))
                pos, current_sl, current_tp = 0, None, None
            elif l <= current_tp:
                if is_last: latest_events.append(("EXIT", "SHORT", current_tp, "Take Profit Hit (+30 pts)"))
                pos, current_sl, current_tp = 0, None, None

        if query_3bar(lows[i], lows[i-1], lows[i-2], False): stLow.update_pattern(lows[i-1], i-1)
        if query_3bar(highs[i], highs[i-1], highs[i-2], True): stHigh.update_pattern(highs[i-1], i-1)

        if stLow.lastPrice is not None and c < stLow.lastPrice and not stLow.isCrossed: stLow.isCrossed = True
        if stHigh.lastPrice is not None and c > stHigh.lastPrice and not stHigh.isCrossed: stHigh.isCrossed = True

        cITL = stLow.query_patterns(False)
        if cITL and cITL != prev_cITL: itLow.update_pattern(stLow.midPrice, stLow.midIndex)
        prev_cITL = cITL

        cITH = stHigh.query_patterns(True)
        if cITH and cITH != prev_cITH: itHigh.update_pattern(stHigh.midPrice, stHigh.midIndex)
        prev_cITH = cITH

        cLTL = itLow.query_patterns(False)
        if cLTL and cLTL != prev_cLTL:
            ltLow.isCrossed, ltLow.lastPrice, ltLow.lastIndex = False, itLow.midPrice, itLow.midIndex
        prev_cLTL = cLTL

        cLTH = itHigh.query_patterns(True)
        if cLTH and cLTH != prev_cLTH:
            ltHigh.isCrossed, ltHigh.lastPrice, ltHigh.lastIndex = False, itHigh.midPrice, itHigh.midIndex
        prev_cLTH = cLTH

        lt_low_broken = ltLow.lastPrice is not None and c < ltLow.lastPrice and not ltLow.isCrossed
        if lt_low_broken: ltLow.isCrossed = True

        lt_high_broken = ltHigh.lastPrice is not None and c > ltHigh.lastPrice and not ltHigh.isCrossed
        if lt_high_broken: ltHigh.isCrossed = True

        is_flat = (pos == 0)
        session_ok = is_ict_session_allowed(ts)
        daily_ema = get_daily_ema(ts)
        trend_long = (daily_ema is None) or (c > daily_ema)
        trend_short = (daily_ema is None) or (c < daily_ema)

        if is_flat and session_ok and lt_high_broken and trend_short:
            current_sl, current_tp, pos = c + sl_buffer, c - tp_points, -1
            if is_last:
                latest_events.append(("ENTRY", "SHORT", c, f"LT High BoS | SL: `{current_sl:.2f}` | TP: `{current_tp:.2f}`"))
        elif is_flat and session_ok and lt_low_broken and trend_long:
            current_sl, current_tp, pos = c - sl_buffer, c + tp_points, 1
            if is_last:
                latest_events.append(("ENTRY", "LONG", c, f"LT Low BoS | SL: `{current_sl:.2f}` | TP: `{current_tp:.2f}`"))

    return latest_events


# ----------------- STRATEGY 2: Positional (2h) -----------------
def compute_supertrend_pine(df, atr_period=25, multiplier=4.0):
    high, low, close = df['High'].values, df['Low'].values, df['Close'].values
    n = len(df)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    atr = pd.Series(tr).ewm(alpha=1.0/atr_period, adjust=False).mean().values
    hl2 = (high + low) / 2.0
    upper_band, lower_band = hl2 + multiplier * atr, hl2 - multiplier * atr
    st, direction = np.zeros(n), np.ones(n)

    for i in range(1, n):
        if not (lower_band[i] > lower_band[i-1] or close[i-1] < lower_band[i-1]):
            lower_band[i] = lower_band[i-1]
        if not (upper_band[i] < upper_band[i-1] or close[i-1] > upper_band[i-1]):
            upper_band[i] = upper_band[i-1]
        if st[i-1] == upper_band[i-1]:
            direction[i] = -1 if close[i] > upper_band[i] else 1
        else:
            direction[i] = 1 if close[i] < lower_band[i] else -1
        st[i] = lower_band[i] if direction[i] == -1 else upper_band[i]
    return st, direction


def evaluate_positional_2h(df_2h, st_len=25, st_mult=4.0, ema_len=100, arm_pct=3.0, trail_pct=4.0):
    df = df_2h.copy()
    _, direction = compute_supertrend_pine(df, atr_period=st_len, multiplier=st_mult)
    ema100 = df['Close'].ewm(span=ema_len, adjust=False).mean().values
    highs, lows, closes = df['High'].values, df['Low'].values, df['Close'].values

    pos, entry_px, best, active_trail_stop = 0, None, None, None
    latest_events = []

    for i in range(1, len(df)):
        is_last = (i == len(df) - 1)
        h, l, c = highs[i], lows[i], closes[i]

        if pos == 1 and active_trail_stop is not None and l <= active_trail_stop:
            if is_last: latest_events.append(("EXIT", "LONG", active_trail_stop, "XAU_LONG_EXIT (TRAIL_4%)"))
            pos, entry_px, best, active_trail_stop = 0, None, None, None
        elif pos == -1 and active_trail_stop is not None and h >= active_trail_stop:
            if is_last: latest_events.append(("EXIT", "SHORT", active_trail_stop, "XAU_SHORT_EXIT (TRAIL_4%)"))
            pos, entry_px, best, active_trail_stop = 0, None, None, None

        flip_up   = (direction[i] < 0 and direction[i-1] > 0)
        flip_down = (direction[i] > 0 and direction[i-1] < 0)

        if flip_up:
            if c > ema100[i]:
                if pos == -1 and is_last:
                    latest_events.append(("EXIT", "SHORT", c, "XAU_SHORT_EXIT (Reversing to Long)"))
                pos, entry_px, best, active_trail_stop = 1, c, None, None
                if is_last:
                    latest_events.append(("ENTRY", "LONG", c, f"XAU_LONG_ENTRY (ST Flip Up > EMA100 `{ema100[i]:.2f}`)"))
            elif pos == -1:
                if is_last: latest_events.append(("EXIT", "SHORT", c, "XAU_SHORT_EXIT (ST_FLIP)"))
                pos, entry_px, best, active_trail_stop = 0, None, None, None

        if flip_down:
            if c < ema100[i]:
                if pos == 1 and is_last:
                    latest_events.append(("EXIT", "LONG", c, "XAU_LONG_EXIT (Reversing to Short)"))
                pos, entry_px, best, active_trail_stop = -1, c, None, None
                if is_last:
                    latest_events.append(("ENTRY", "SHORT", c, f"XAU_SHORT_ENTRY (ST Flip Down < EMA100 `{ema100[i]:.2f}`)"))
            elif pos == 1:
                if is_last: latest_events.append(("EXIT", "LONG", c, "XAU_LONG_EXIT (ST_FLIP)"))
                pos, entry_px, best, active_trail_stop = 0, None, None, None

        if pos == 1:
            best = max(entry_px, h) if best is None else max(best, h)
            if best >= entry_px * (1.0 + arm_pct / 100.0):
                active_trail_stop = best * (1.0 - trail_pct / 100.0)
        elif pos == -1:
            best = min(entry_px, l) if best is None else min(best, l)
            if best <= entry_px * (1.0 - arm_pct / 100.0):
                active_trail_stop = best * (1.0 + trail_pct / 100.0)
        else:
            best = active_trail_stop = None

    return latest_events


def main():
    start_time = time.time()
    last_5m_ts = None
    last_2h_ts = None

    if EVENT_NAME == "workflow_dispatch":
        send_telegram_Message("✅ *Bot Connected!* Monitoring *ICT (5m)* and *Positional (2h)* on XAUUSD.")

    while (time.time() - start_time) < MAX_RUNTIME_SECONDS:
        try:
            df_5m = fetch_ohlc(SYMBOL, interval="5m", period="30d")
            df_1d = fetch_ohlc(SYMBOL, interval="1d", period="1y")
            if not df_5m.empty and not df_1d.empty:
                latest_5m = df_5m.index[-1]
                if latest_5m != last_5m_ts:
                    if last_5m_ts is not None:
                        for act, d, px, dtl in evaluate_ict_5m(df_5m, df_1d):
                            send_telegram_alert("ICT", "5m", act, d, px, dtl)
                    last_5m_ts = latest_5m

            df_2h = fetch_2h_ohlc(SYMBOL)
            if not df_2h.empty:
                latest_2h = df_2h.index[-1]
                if latest_2h != last_2h_ts:
                    if last_2h_ts is not None:
                        for act, d, px, dtl in evaluate_positional_2h(df_2h):
                            send_telegram_alert("Positional", "2h", act, d, px, dtl)
                    last_2h_ts = latest_2h

        except Exception as e:
            print(f"Loop error: {e}")

        time.sleep(CHECK_INTERVAL_SECONDS)


if __name__ == "__main__":
    main()
