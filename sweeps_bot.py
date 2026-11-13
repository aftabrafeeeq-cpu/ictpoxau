import os
import time
import requests
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
import pytz

# Uses the new dedicated Bot Token and Sweep Chat ID
BOT_TOKEN = os.environ.get("BOT_TOKEN", "").strip()
CHAT_ID   = os.environ.get(
    "SWEEP_CHAT_ID", os.environ.get("CHAT_ID", "")
).strip()
EVENT_NAME = os.environ.get("GITHUB_EVENT_NAME", "")

SWEEP_SYMBOLS = {
    "XAUUSD": "GC=F",
    "USCRUDEOIL": "CL=F",
    "BTC": "BTC-USD",
    "USTECH": "NQ=F",
    "NIFTY": "^NSEI",
    "SENSEX": "^BSESN",
    "BANKNIFTY": "^NSEBANK",
}

CHECK_INTERVAL_SECONDS = 60
MAX_RUNTIME_SECONDS = 5 * 3600 + 45 * 60


def send_telegram_Message(text):
  url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
  try:
    resp = requests.post(
        url,
        data={"chat_id": CHAT_ID, "text": text, "parse_mode": "Markdown"},
        timeout=10,
    )
    if not resp.ok:
      print(f"Telegram API Error: {resp.text}")
  except Exception as e:
    print(f"Failed to send Telegram alert: {e}")


def send_sweep_alert(symbol_name, timeframe, direction, price, details=""):
  emoji = "🟢" if "BULLISH" in direction else "🔴"
  msg = (
      f"🧹 *[ICT-SWEEP] ({timeframe})*\n"
      f"{emoji} *{direction}*\n"
      f"*Symbol:* `{symbol_name}`\n"
      f"*Close Price:* `{price:.2f}`\n"
      f"*Details:* {details}\n"
      f"*Time (UTC):* `{datetime.now(pytz.utc).strftime('%Y-%m-%d %H:%M:%S')}`"
  )
  print(msg)
  send_telegram_Message(msg)


def fetch_1h_ohlc(symbol):
  df = yf.Ticker(symbol).history(period="60d", interval="1h")
  if df.empty:
    return df
  df = df[["Open", "High", "Low", "Close"]].dropna()
  df.index = (
      df.index.tz_localize("UTC")
      if df.index.tz is None
      else df.index.tz_convert("UTC")
  )
  return df.iloc[:-1].copy()


def build_4h_from_1h(df_1h, is_indian_index=False):
  if df_1h.empty:
    return df_1h
  if is_indian_index:
    df_temp = df_1h.copy()
    df_temp["day"] = df_temp.index.date
    df_temp["sub_idx"] = df_temp.groupby("day").cumcount() // 4
    grouped = df_temp.groupby(["day", "sub_idx"])
    df_4h = grouped.agg(
        {"Open": "first", "High": "max", "Low": "min", "Close": "last"}
    )
    first_times = df_temp.groupby(["day", "sub_idx"]).apply(
        lambda x: x.index[0]
    )
    df_4h.index = pd.DatetimeIndex(first_times.values)
    return df_4h.dropna()
  else:
    return (
        df_1h.resample("4h", closed="left", label="left")
        .agg({"Open": "first", "High": "max", "Low": "min", "Close": "last"})
        .dropna()
    )


class ICTMS:

  def __init__(self):
    self.lastPrice = self.midPrice = self.prevPrice = None
    self.lastIndex = self.midIndex = self.prevIndex = None
    self.isCrossed = False
    self.isConfirmed = False

  def query_patterns(self, is_swing_high):
    if None in (self.prevPrice, self.midPrice, self.lastPrice):
      return False
    if is_swing_high:
      return (
          self.prevPrice < self.midPrice and self.midPrice >= self.lastPrice
      )
    return self.prevPrice > self.midPrice and self.midPrice <= self.lastPrice

  def update_pattern(self, price, idx):
    self.isCrossed = False
    self.prevPrice, self.midPrice, self.lastPrice = (
        self.midPrice,
        self.lastPrice,
        price,
    )
    self.prevIndex, self.midIndex, self.lastIndex = (
        self.midIndex,
        self.lastIndex,
        idx,
    )


def query_3bar(last_p, mid_p, prev_p, is_high):
  return (
      (prev_p < mid_p and mid_p >= last_p)
      if is_high
      else (prev_p > mid_p and mid_p <= last_p)
  )


def compute_rma_atr(df, period=17):
  high, low, close = df["High"].values, df["Low"].values, df["Close"].values
  n = len(df)
  tr = np.zeros(n)
  tr[0] = high[0] - low[0]
  for i in range(1, n):
    tr[i] = max(
        high[i] - low[i],
        abs(high[i] - close[i - 1]),
        abs(low[i] - close[i - 1]),
    )
  return pd.Series(tr).ewm(alpha=1.0 / period, adjust=False).mean().values


def render_and_validate_sweep(
    struct_obj, is_bullish, close_px, deviation, term_label
):
  if struct_obj.lastPrice is None or np.isnan(deviation):
    return None

  cond = (
      (close_px > struct_obj.lastPrice)
      if is_bullish
      else (close_px < struct_obj.lastPrice)
  )
  if cond and not struct_obj.isCrossed:
    struct_obj.isCrossed = True
    struct_obj.isConfirmed = False

  if (
      close_px > (struct_obj.lastPrice + deviation)
      or close_px < (struct_obj.lastPrice - deviation)
  ) and not struct_obj.isConfirmed:
    struct_obj.isConfirmed = True
    if is_bullish:
      if close_px < (struct_obj.lastPrice - deviation):
        return (
            "BEARISH SWEEP (High Swept)",
            close_px,
            f"`{term_label}-SWEEP` | Swept Level: `{struct_obj.lastPrice:.2f}`",
        )
    else:
      if close_px > (struct_obj.lastPrice + deviation):
        return (
            "BULLISH SWEEP (Low Swept)",
            close_px,
            f"`{term_label}-SWEEP` | Swept Level: `{struct_obj.lastPrice:.2f}`",
        )
  return None


def evaluate_ict_sweeps(df):
  if len(df) < 20:
    return []

  sweep_atr = compute_rma_atr(df, period=17)
  highs, lows, closes = df["High"].values, df["Low"].values, df["Close"].values

  stLow, stHigh = ICTMS(), ICTMS()
  itLow, itHigh = ICTMS(), ICTMS()
  ltLow, ltHigh = ICTMS(), ICTMS()
  prev_cITL = prev_cITH = prev_cLTL = prev_cLTH = False
  latest_events = []

  for i in range(2, len(df)):
    is_last = i == len(df) - 1
    c, dev = closes[i], sweep_atr[i]

    if query_3bar(lows[i], lows[i - 1], lows[i - 2], False):
      stLow.update_pattern(lows[i - 1], i - 1)
    if query_3bar(highs[i], highs[i - 1], highs[i - 2], True):
      stHigh.update_pattern(highs[i - 1], i - 1)

    ev = render_and_validate_sweep(stLow, False, c, dev, "STL")
    if ev and is_last:
      latest_events.append(ev)

    ev = render_and_validate_sweep(stHigh, True, c, dev, "STH")
    if ev and is_last:
      latest_events.append(ev)

    cITL = stLow.query_patterns(False)
    if cITL and cITL != prev_cITL:
      itLow.update_pattern(stLow.midPrice, stLow.midIndex)
    prev_cITL = cITL

    cITH = stHigh.query_patterns(True)
    if cITH and cITH != prev_cITH:
      itHigh.update_pattern(stHigh.midPrice, stHigh.midIndex)
    prev_cITH = cITH

    ev = render_and_validate_sweep(itLow, False, c, dev, "ITL")
    if ev and is_last:
      latest_events.append(ev)

    ev = render_and_validate_sweep(itHigh, True, c, dev, "ITH")
    if ev and is_last:
      latest_events.append(ev)

    cLTL = itLow.query_patterns(False)
    if cLTL and cLTL != prev_cLTL:
      ltLow.isCrossed = False
      ltLow.lastPrice = itLow.midPrice
      ltLow.lastIndex = itLow.midIndex
    prev_cLTL = cLTL

    cLTH = itHigh.query_patterns(True)
    if cLTH and cLTH != prev_cLTH:
      ltHigh.isCrossed = False
      ltHigh.lastPrice = itHigh.midPrice
      ltHigh.lastIndex = itHigh.midIndex
    prev_cLTH = cLTH

    ev = render_and_validate_sweep(ltLow, False, c, dev, "LTL")
    if ev and is_last:
      latest_events.append(ev)

    ev = render_and_validate_sweep(ltHigh, True, c, dev, "LTH")
    if ev and is_last:
      latest_events.append(ev)

  return latest_events


def main():
  start_time = time.time()
  last_1h_ts = {sym: None for sym in SWEEP_SYMBOLS}
  last_4h_ts = {sym: None for sym in SWEEP_SYMBOLS}

  if EVENT_NAME == "workflow_dispatch":
    send_telegram_Message(
        "🧹 *ICT Sweeps Scanner (1H & 4H) Connected!*\n*Assets:* `XAUUSD,"
        " USCRUDEOIL, BTC, USTECH, NIFTY, SENSEX, BANKNIFTY`"
    )

  while (time.time() - start_time) < MAX_RUNTIME_SECONDS:
    for sym_name, yf_ticker in SWEEP_SYMBOLS.items():
      try:
        df_1h = fetch_1h_ohlc(yf_ticker)
        if df_1h.empty:
          continue

        latest_1h = df_1h.index[-1]
        if latest_1h != last_1h_ts[sym_name]:
          if last_1h_ts[sym_name] is not None:
            for d, px, dtl in evaluate_ict_sweeps(df_1h):
              send_sweep_alert(sym_name, "1H", d, px, dtl)
          last_1h_ts[sym_name] = latest_1h

        is_indian = sym_name in ("NIFTY", "SENSEX", "BANKNIFTY")
        df_4h = build_4h_from_1h(df_1h, is_indian_index=is_indian)
        if not df_4h.empty:
          latest_4h = df_4h.index[-1]
          if latest_4h != last_4h_ts[sym_name]:
            if last_4h_ts[sym_name] is not None:
              for d, px, dtl in evaluate_ict_sweeps(df_4h):
                send_sweep_alert(sym_name, "4H", d, px, dtl)
            last_4h_ts[sym_name] = latest_4h

      except Exception as e:
        print(f"Error scanning {sym_name}: {e}")

    time.sleep(CHECK_INTERVAL_SECONDS)


if __name__ == "__main__":
  main()
