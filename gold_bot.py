"""
Gold Telegram signal bot (XAU/USD)
Trend: H4 (EMA100 + slope, closed candles only)
Entry: M15 pullback to EMA50 + rejection candle + RSI filter
Sends Entry / SL / TP to Telegram. It does NOT place trades.
"""
import os
import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

# ---------- SETTINGS (set as environment variables) ----------
TELEGRAM_TOKEN = os.environ["TELEGRAM_TOKEN"]
CHAT_ID = os.environ["CHAT_ID"]
TWELVE_KEY = os.environ["TWELVE_KEY"]

SYMBOL = "XAU/USD"
TREND_EMA = 100
ENTRY_EMA = 50
RSI_PERIOD = 14
ATR_PERIOD = 14
TOUCH_ATR = 0.3   # pullback tolerance (x ATR)
SL_ATR = 1.5      # SL beyond candle extreme (x ATR)
RR = 2.0          # take profit = RR x risk
SESSION_START = 7   # UTC hour: London open
SESSION_END = 20    # UTC hour: NY late session
RUN_ONCE = os.environ.get("RUN_ONCE", "0") == "1"  # 1 = run once and exit (GitHub Actions)
# --------------------------------------------------------------


def send(text: str) -> None:
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try:
        requests.post(url, data={"chat_id": CHAT_ID, "text": text}, timeout=15)
    except Exception as e:
        print("Telegram error:", e)


def get_candles(interval: str, minutes: int, size: int) -> pd.DataFrame:
    r = requests.get(
        "https://api.twelvedata.com/time_series",
        params={
            "symbol": SYMBOL,
            "interval": interval,
            "outputsize": size,
            "order": "ASC",
            "timezone": "UTC",
            "apikey": TWELVE_KEY,
        },
        timeout=20,
    )
    data = r.json()
    if "values" not in data:
        raise RuntimeError(f"Data error: {data}")
    df = pd.DataFrame(data["values"])
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    for c in ("open", "high", "low", "close"):
        df[c] = df[c].astype(float)
    df = df.sort_values("datetime").reset_index(drop=True)
    # drop the candle that is still forming
    now = datetime.now(timezone.utc)
    if df.iloc[-1]["datetime"] + timedelta(minutes=minutes) > now:
        df = df.iloc[:-1].reset_index(drop=True)
    return df


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def rsi(s: pd.Series, n: int) -> pd.Series:
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = up / dn
    return 100 - 100 / (1 + rs)


def atr(df: pd.DataFrame, n: int) -> pd.Series:
    pc = df["close"].shift()
    tr = pd.concat(
        [df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def trend_direction() -> int:
    h4 = get_candles("4h", 240, 300)
    m = ema(h4["close"], TREND_EMA)
    c = h4["close"].iloc[-1]
    if c < m.iloc[-1] and m.iloc[-1] < m.iloc[-2]:
        return -1
    if c > m.iloc[-1] and m.iloc[-1] > m.iloc[-2]:
        return 1
    return 0


def check_signal(last_signal_time):
    tr = trend_direction()
    if tr == 0:
        print("No clear H4 trend")
        return last_signal_time

    df = get_candles("15min", 15, 300)
    df["ema"] = ema(df["close"], ENTRY_EMA)
    df["rsi"] = rsi(df["close"], RSI_PERIOD)
    df["atr"] = atr(df, ATR_PERIOD)
    k = df.iloc[-1]  # last closed M15 candle

    # session filter: only trade liquid hours
    if not (SESSION_START <= k["datetime"].hour < SESSION_END):
        print("Outside session")
        return last_signal_time
    # in run-once mode only react to a fresh candle (avoids stale/weekend data)
    if RUN_ONCE:
        closed_at = k["datetime"] + timedelta(minutes=15)
        if datetime.now(timezone.utc) - closed_at > timedelta(minutes=16):
            print("Candle too old, skipping")
            return last_signal_time
    tol = TOUCH_ATR * k["atr"]

    sell = (
        tr == -1
        and k["high"] >= k["ema"] - tol
        and k["close"] < k["ema"]
        and k["close"] < k["open"]
        and k["rsi"] < 55
    )
    buy = (
        tr == 1
        and k["low"] <= k["ema"] + tol
        and k["close"] > k["ema"]
        and k["close"] > k["open"]
        and k["rsi"] > 45
    )

    if not (buy or sell) or k["datetime"] == last_signal_time:
        return last_signal_time

    entry = k["close"]
    if sell:
        sl = k["high"] + SL_ATR * k["atr"]
        tp = entry - (sl - entry) * RR
        side = "SELL \U0001F534"
    else:
        sl = k["low"] - SL_ATR * k["atr"]
        tp = entry + (entry - sl) * RR
        side = "BUY \U0001F7E2"

    send(
        f"XAUUSD {side}\n"
        f"Entry ~ {entry:.2f}\n"
        f"SL: {sl:.2f}\n"
        f"TP: {tp:.2f} (RR 1:{RR:g})\n"
        f"H4 trend: {'down' if tr == -1 else 'up'} | RSI {k['rsi']:.0f}\n"
        f"Candle: {k['datetime']:%Y-%m-%d %H:%M} UTC\n"
        "Not financial advice. Always use a stop."
    )
    return k["datetime"]


def sleep_to_next_quarter() -> None:
    now = datetime.now(timezone.utc)
    nxt = (now + timedelta(minutes=15 - now.minute % 15)).replace(second=15, microsecond=0)
    time.sleep(max((nxt - now).total_seconds(), 5))


def main() -> None:
    if RUN_ONCE:
        check_signal(None)
        return
    send("Gold bot started \u2705")
    last = None
    while True:
        try:
            last = check_signal(last)
        except Exception as e:
            print("Error:", e)
        sleep_to_next_quarter()


if __name__ == "__main__":
    main()
