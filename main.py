"""Toobit Scalp Backtest - Kivy app (numpy only, no pandas)."""
import json
import threading
import urllib.parse
import urllib.request

import numpy as np
from kivy.app import App
from kivy.clock import Clock
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.scrollview import ScrollView
from kivy.uix.spinner import Spinner

# ---------- Settings ----------
KLINES_URL = "https://api.toobit.com/quote/v1/klines"  # verify with Toobit docs
INTERVAL = "5m"
PAGES = 5                  # 5 x 1000 candles
START_BALANCE = 1000.0
RISK = 0.005               # 0.5% per trade
FEE = 0.0006               # per side
SLIP = 0.0002              # per side
SL_ATR, TP_ATR = 1.2, 1.8
MAX_BARS = 24
DAILY_LOSS_LIMIT = 0.03
MAX_TRADES_DAY = 10


# ---------- Data ----------
def get_json(params):
    url = KLINES_URL + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.loads(r.read().decode())


def fetch(symbol):
    rows, end = [], None
    for _ in range(PAGES):
        p = {"symbol": symbol, "interval": INTERVAL, "limit": 1000}
        if end:
            p["endTime"] = end
        page = get_json(p)
        if not page:
            break
        rows = page + rows
        end = int(page[0][0]) - 1
    a = np.array([[float(x) for x in r[:6]] for r in rows])
    a = a[np.argsort(a[:, 0])]
    _, idx = np.unique(a[:, 0], return_index=True)
    return a[idx]  # time, open, high, low, close, volume


# ---------- Indicators ----------
def ema(x, alpha):
    out = np.empty_like(x)
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = alpha * x[i] + (1 - alpha) * out[i - 1]
    return out


def indicators(d):
    t, o, h, l, c, v = d.T
    ef, es = ema(c, 2 / 10), ema(c, 2 / 22)
    diff = np.diff(c, prepend=c[0])
    up = ema(np.clip(diff, 0, None), 1 / 14)
    dn = ema(np.clip(-diff, 0, None), 1 / 14)
    rsi = 100 - 100 / (1 + up / np.where(dn == 0, np.nan, dn))
    pc = np.roll(c, 1)
    pc[0] = c[0]
    tr = np.maximum(h - l, np.maximum(abs(h - pc), abs(l - pc)))
    atr = ema(tr, 1 / 14)
    day = (t // 86400000).astype(int)
    tp = (h + l + c) / 3
    vwap = np.empty_like(c)
    cp = cv = 0.0
    for i in range(len(c)):
        if i == 0 or day[i] != day[i - 1]:
            cp = cv = 0.0
        cp += tp[i] * v[i]
        cv += v[i]
        vwap[i] = cp / cv if cv else c[i]
    vma = np.convolve(v, np.ones(20) / 20, mode="full")[: len(v)]
    return ef, es, rsi, atr, vwap, vma, day


# ---------- Backtest ----------
def backtest(d):
    t, o, h, l, c, v = d.T
    ef, es, rsi, atr, vwap, vma, day = indicators(d)
    bal, trades = START_BALANCE, []
    cur_day, day_bal, day_n = None, bal, 0
    i = 30
    while i < len(c) - 1:
        if day[i] != cur_day:
            cur_day, day_bal, day_n = day[i], bal, 0
        if day_n >= MAX_TRADES_DAY or bal <= day_bal * (1 - DAILY_LOSS_LIMIT):
            i += 1
            continue
        side = 0
        if not np.isnan(rsi[i]) and v[i] > vma[i]:
            if ef[i - 1] <= es[i - 1] and ef[i] > es[i] and c[i] > vwap[i] and 50 < rsi[i] < 70:
                side = 1
            elif ef[i - 1] >= es[i - 1] and ef[i] < es[i] and c[i] < vwap[i] and 30 < rsi[i] < 50:
                side = -1
        if side == 0:
            i += 1
            continue
        entry = o[i + 1] * (1 + SLIP * side)
        dist = SL_ATR * atr[i]
        sl, tp_ = entry - side * dist, entry + side * TP_ATR * atr[i]
        size = bal * RISK / dist
        ex, xi = None, None
        for j in range(i + 1, min(i + 1 + MAX_BARS, len(c))):
            hit_sl = l[j] <= sl if side == 1 else h[j] >= sl
            hit_tp = h[j] >= tp_ if side == 1 else l[j] <= tp_
            if hit_sl:
                ex, xi = sl, j
                break
            if hit_tp:
                ex, xi = tp_, j
                break
        if ex is None:
            xi = min(i + MAX_BARS, len(c) - 1)
            ex = c[xi]
        ex *= 1 - SLIP * side
        pnl = (ex - entry) * side * size - (entry + ex) * size * FEE
        bal += pnl
        day_n += 1
        trades.append((day[i], pnl, bal))
        i = xi + 1
    return trades


def report(name, trades):
    if not trades:
        return f"{name}: no trades"
    pnl = np.array([x[1] for x in trades])
    eq = np.array([x[2] for x in trades])
    wins, loss = pnl[pnl > 0], pnl[pnl <= 0]
    pf = wins.sum() / abs(loss.sum()) if len(loss) else float("inf")
    dd = ((np.maximum.accumulate(eq) - eq) / np.maximum.accumulate(eq)).max()
    days = {}
    for dday, p, _ in trades:
        days[dday] = days.get(dday, 0) + p
    prof_days = sum(1 for x in days.values() if x > 0) / len(days)
    return (f"== {name} ==\nTrades: {len(pnl)}\nWin rate: {len(wins) / len(pnl):.1%}\n"
            f"Profit factor: {pf:.2f}\nNet P/L: {eq[-1] - START_BALANCE:.2f} USDT\n"
            f"Max drawdown: {dd:.1%}\nProfitable days: {prof_days:.1%} of {len(days)}\n")


# ---------- UI ----------
class Root(BoxLayout):
    def __init__(self, **kw):
        super().__init__(orientation="vertical", padding=10, spacing=10, **kw)
        self.spin = Spinner(text="BTCUSDT", values=("BTCUSDT", "ETHUSDT", "BOTH"),
                            size_hint_y=None, height=120)
        self.btn = Button(text="Run Backtest", size_hint_y=None, height=140)
        self.btn.bind(on_release=self.run)
        self.out = Label(text="Ready.", size_hint_y=None, halign="left", valign="top")
        self.out.bind(width=lambda *_: setattr(self.out, "text_size", (self.out.width, None)),
                      texture_size=lambda *_: setattr(self.out, "height", self.out.texture_size[1]))
        sv = ScrollView()
        sv.add_widget(self.out)
        for w in (self.spin, self.btn, sv):
            self.add_widget(w)

    def run(self, *_):
        self.btn.disabled = True
        self.out.text = "Downloading data and running..."
        threading.Thread(target=self.work, args=(self.spin.text,), daemon=True).start()

    def work(self, choice):
        syms = ["BTCUSDT", "ETHUSDT"] if choice == "BOTH" else [choice]
        text = ""
        for s in syms:
            try:
                text += report(s, backtest(fetch(s))) + "\n"
            except Exception as e:
                text += f"{s}: error - {e}\n\n"
        Clock.schedule_once(lambda dt: self.done(text))

    def done(self, text):
        self.out.text = text
        self.btn.disabled = False


class ScalpApp(App):
    def build(self):
        return Root()


if __name__ == "__main__":
    ScalpApp().run()
