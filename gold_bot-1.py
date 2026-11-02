"""
Gold Telegram signal bot (XAU/USD) v2
- H4 trend (EMA100 + slope) and H4 market structure (HH/HL vs LH/LL)
- M15 entry: pullback to EMA50 + rejection candle + RSI filter
- Support/resistance levels: H4 + M15 swing points + round numbers
- Room filter: skips signals that run straight into a level
- News filter: no signals around high-impact USD news
Sends Entry / SL / TP1 / TP2 to Telegram. It does NOT place trades.
"""
import os
import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

# ---------- SETTINGS ----------
TELEGRAM_TOKEN = os.environ["TELEGRAM_TOKEN"]
CHAT_ID = os.environ["CHAT_ID"]
TWELVE_KEY = os.environ["TWELVE_KEY"]

SYMBOL = "XAU/USD"
TREND_EMA = 100
ENTRY_EMA = 50
RSI_PERIOD = 14
ATR_PERIOD = 14
TOUCH_ATR = 0.3        # pullback tolerance (x ATR)
SL_ATR = 1.5           # SL beyond candle extreme (x ATR)
RR = 2.0               # TP2 = RR x risk
MIN_ROOM_R = 1.2       # nearest opposing level must be >= this many R away
REQUIRE_LEVEL = False  # True = only signals where the rejection happened at a level
ROUND_STEP = 50        # round-number levels (3950, 4000, ...)
SESSION_START = 7      # UTC hour
SESSION_END = 20       # UTC hour
NEWS_BEFORE_MIN = 30   # no signals this many minutes before high-impact USD news
NEWS_AFTER_MIN = 60    # ...and this many minutes after
NEWS_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
RUN_ONCE = os.environ.get("RUN_ONCE", "0") == "1"
# ------------------------------


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
    now = datetime.now(timezone.utc)
    if df.iloc[-1]["datetime"] + timedelta(minutes=minutes) > now:
        df = df.iloc[:-1].reset_index(drop=True)  # drop forming candle
    return df


# ---------- indicators ----------
def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def rsi(s, n):
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn)


def atr(df, n):
    pc = df["close"].shift()
    tr = pd.concat(
        [df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


# ---------- structure and levels ----------
def swings(df, n):
    """Swing highs/lows: a candle that is the extreme of n candles on each side."""
    highs, lows = [], []
    for i in range(n, len(df) - n):
        h, l = df["high"].iloc[i], df["low"].iloc[i]
        if h == df["high"].iloc[i - n:i + n + 1].max():
            highs.append((i, h))
        if l == df["low"].iloc[i - n:i + n + 1].min():
            lows.append((i, l))
    return highs, lows


def h4_trend(h4):
    m = ema(h4["close"], TREND_EMA)
    c = h4["close"].iloc[-1]
    if c < m.iloc[-1] and m.iloc[-1] < m.iloc[-2]:
        return -1
    if c > m.iloc[-1] and m.iloc[-1] > m.iloc[-2]:
        return 1
    return 0


def h4_structure(h4):
    """+1 = higher highs & higher lows, -1 = lower highs & lower lows, 0 = mixed."""
    hi, lo = swings(h4.iloc[-150:].reset_index(drop=True), 2)
    if len(hi) < 2 or len(lo) < 2:
        return 0
    hh = hi[-1][1] > hi[-2][1]
    hl = lo[-1][1] > lo[-2][1]
    if hh and hl:
        return 1
    if (not hh) and (not hl):
        return -1
    return 0


def build_levels(h4, m15, atr15):
    pts = []
    hi, lo = swings(h4.iloc[-200:].reset_index(drop=True), 3)
    pts += [p for _, p in hi + lo]
    hi, lo = swings(m15.iloc[-150:].reset_index(drop=True), 8)
    pts += [p for _, p in hi + lo]
    price = m15["close"].iloc[-1]
    base = round(price / ROUND_STEP) * ROUND_STEP
    pts += [base + j * ROUND_STEP for j in range(-6, 7)]
    pts.sort()
    groups, tol = [], 0.5 * atr15
    for p in pts:
        if groups and p - groups[-1][-1] <= tol:
            groups[-1].append(p)
        else:
            groups.append([p])
    return [sum(g) / len(g) for g in groups]


# ---------- news ----------
def news_status(now):
    """Returns (blocking_event_title or None, next_event_text or None, ok)."""
    try:
        r = requests.get(NEWS_URL, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
        events = r.json()
        block, nxt = None, None
        for ev in events:
            if ev.get("country") != "USD" or ev.get("impact") != "High":
                continue
            t = pd.to_datetime(ev["date"], utc=True)
            if t - timedelta(minutes=NEWS_BEFORE_MIN) <= now <= t + timedelta(minutes=NEWS_AFTER_MIN):
                block = f"{ev.get('title', 'news')} ({t:%H:%M} UTC)"
            if t > now and (nxt is None or t < nxt[1]):
                nxt = (ev.get("title", "news"), t)
        nxt_text = f"{nxt[0]} {nxt[1]:%a %H:%M} UTC" if nxt else None
        return block, nxt_text, True
    except Exception as e:
        print("News calendar error:", e)
        return None, None, False


# ---------- main logic ----------
def check_signal(last_signal_time):
    h4 = get_candles("4h", 240, 300)
    tr = h4_trend(h4)
    if tr == 0:
        print("No clear H4 trend")
        return last_signal_time
    st = h4_structure(h4)
    if st == -tr:
        print("H4 structure conflicts with trend, skipping")
        return last_signal_time

    df = get_candles("15min", 15, 300)
    df["ema"] = ema(df["close"], ENTRY_EMA)
    df["rsi"] = rsi(df["close"], RSI_PERIOD)
    df["atr"] = atr(df, ATR_PERIOD)
    k = df.iloc[-1]

    if not (SESSION_START <= k["datetime"].hour < SESSION_END):
        print("Outside session")
        return last_signal_time
    now = datetime.now(timezone.utc)
    if RUN_ONCE and now - (k["datetime"] + timedelta(minutes=15)) > timedelta(minutes=16):
        print("Candle too old, skipping")
        return last_signal_time

    tol = TOUCH_ATR * k["atr"]
    sell = (tr == -1 and k["high"] >= k["ema"] - tol and k["close"] < k["ema"]
            and k["close"] < k["open"] and k["rsi"] < 55)
    buy = (tr == 1 and k["low"] <= k["ema"] + tol and k["close"] > k["ema"]
           and k["close"] > k["open"] and k["rsi"] > 45)
    if not (buy or sell) or k["datetime"] == last_signal_time:
        return last_signal_time

    # news filter
    block, nxt_news, news_ok = news_status(now)
    if block:
        print("News block:", block)
        return last_signal_time

    # levels and room
    levels = build_levels(h4, df, k["atr"])
    entry = k["close"]
    gap = 0.1 * k["atr"]
    if sell:
        sl = k["high"] + SL_ATR * k["atr"]
        risk = sl - entry
        tp2 = entry - RR * risk
        below = [L for L in levels if L < entry - gap]
        nearest = max(below) if below else None
        if nearest is not None and entry - nearest < MIN_ROOM_R * risk:
            print(f"No room: support {nearest:.2f} too close")
            return last_signal_time
        tp1 = max(nearest, tp2) if nearest is not None else tp2
        at_level = any(abs(L - k["high"]) <= 0.5 * k["atr"] for L in levels)
        side = "SELL \U0001F534"
    else:
        sl = k["low"] - SL_ATR * k["atr"]
        risk = entry - sl
        tp2 = entry + RR * risk
        above = [L for L in levels if L > entry + gap]
        nearest = min(above) if above else None
        if nearest is not None and nearest - entry < MIN_ROOM_R * risk:
            print(f"No room: resistance {nearest:.2f} too close")
            return last_signal_time
        tp1 = min(nearest, tp2) if nearest is not None else tp2
        at_level = any(abs(L - k["low"]) <= 0.5 * k["atr"] for L in levels)
        side = "BUY \U0001F7E2"

    if REQUIRE_LEVEL and not at_level:
        print("Rejection not at a level, skipping")
        return last_signal_time

    near = sorted(levels, key=lambda L: abs(L - entry))[:4]
    near_txt = ", ".join(f"{L:.0f}" for L in sorted(near))
    struct_txt = {1: "HH/HL (up)", -1: "LH/LL (down)", 0: "mixed"}[st]
    lines = [
        f"XAUUSD {side}",
        f"Entry ~ {entry:.2f}",
        f"SL: {sl:.2f}  (risk {risk:.1f})",
        f"TP1: {tp1:.2f}  (nearest level)",
        f"TP2: {tp2:.2f}  (RR 1:{RR:g})",
        "",
        f"H4 trend: {'down' if tr == -1 else 'up'} | structure: {struct_txt}",
        f"RSI {k['rsi']:.0f} | rejection at level: {'yes' if at_level else 'no'}",
        f"Nearby levels: {near_txt}",
    ]
    if nxt_news:
        lines.append(f"Next high-impact USD news: {nxt_news}")
    if not news_ok:
        lines.append("WARNING: news calendar unavailable, check news manually")
    lines.append(f"Candle: {k['datetime']:%Y-%m-%d %H:%M} UTC")
    lines.append("Tip: take part profit at TP1 and move SL to entry.")
    lines.append("Not financial advice. Always use a stop.")
    send("\n".join(lines))
    return k["datetime"]


def sleep_to_next_quarter():
    now = datetime.now(timezone.utc)
    nxt = (now + timedelta(minutes=15 - now.minute % 15)).replace(second=15, microsecond=0)
    time.sleep(max((nxt - now).total_seconds(), 5))


def main():
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
