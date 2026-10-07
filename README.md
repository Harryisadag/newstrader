# NewsTrader

NewsTrader is a Windows desktop app that:

1. **Watches live news**:
   - TV livestreams (Bloomberg, Yahoo Finance, Schwab Network, …)
   - Alpaca/Benzinga real-time news
   - RSS feeds (CNBC, MarketWatch, WSJ, …)
   - Social posts (Truth Social)
2. **Transcribes TV audio in real time** on your NVIDIA GPU (faster-whisper large-v3).
3. **Asks Claude** whether each story is bullish or bearish for a specific stock. Claude argues the bull case and the bear case before giving a 0–100 confidence.
4. **Places PAPER trades on Alpaca** when confidence is high enough. Every buy has a stop-loss and take-profit, and strict risk limits apply.
5. **Tracks whether the AI is actually right.** It checks the price 5 minutes, 1 hour and 1 trading day after every call. It can also **backtest** on historical news.

> ⚠️ **Paper trading only by default.** NewsTrader always starts in **PAPER** mode (fake money). Real-money trading is locked in the code. Unlocking it takes all three of these:
> - separate live keys
> - turning it on in Settings → Live trading
> - typing `I UNDERSTAND THIS USES REAL MONEY`
>
> Even then, it switches back to paper every time the app restarts. NewsTrader is an experiment, not financial advice.

---

## Contents
1. [Setup (one time, ~20 minutes)](#setup-one-time-20-minutes)
2. [Using NewsTrader](#using-newstrader)
3. [Settings explained](#settings-explained)
4. [How a trade happens](#how-a-trade-happens)
5. [Kill switch and live trading](#kill-switch-and-live-trading)
6. [Make a real .exe](#make-a-real-exe)
7. [Updating](#updating)
8. [Troubleshooting](#troubleshooting)
9. [Truth Social and X (Twitter)](#truth-social-and-x-twitter)
10. [What it costs](#what-it-costs)
11. [Where your files are](#where-your-files-are)
12. [For developers](#for-developers)

---

## Setup (one time, ~20 minutes)

You need Windows 10/11, an NVIDIA graphics card (yours is an RTX 5070 Ti), and an internet connection.

### Step 1: Install Python 3.12
1. Go to **https://www.python.org/downloads/** and download the **Python 3.12.x Windows installer (64-bit)**. (3.13 also works; 3.12 is the tested version.)
2. Run it. **On the first screen, tick "Add python.exe to PATH".** Then click **Install Now**.
3. Click **Close** when it's done.

### Step 2: Update your NVIDIA driver
RTX 50-series cards need a 2025-or-newer driver.
1. Open the **NVIDIA App** (or get it at https://www.nvidia.com/en-us/software/nvidia-app/).
2. Go to **Drivers**, install the latest **Game Ready** or **Studio** driver, and restart if asked.

You do **not** need to install the CUDA Toolkit or ffmpeg. NewsTrader installs the CUDA libraries it needs, plus its own copy of ffmpeg, automatically.

### Step 3: Get your API keys
Keep these private. You'll paste them into the app in Step 5.

**Alpaca (paper trading, free)**
1. Sign up / log in at **https://app.alpaca.markets**.
2. Top-left, make sure you're in your **Paper** account (not Live).
3. On the right side of the home page find **API Keys** → **Generate New Keys**.
4. Copy the **Key** and the **Secret**. The secret is shown only once.

**Anthropic (Claude)**
1. Go to **https://platform.claude.com** and sign in.
2. Under **Billing**, add a little credit. $10 lasts a while (see [costs](#what-it-costs)).
3. Recommended: under **Limits**, set a monthly spending limit as a second safety net.
4. Go to **API Keys** → **Create Key**, then copy it. It starts with `sk-ant-`.

**Discord webhook (optional, for phone alerts)**
1. In Discord, open your server → **Server Settings** → **Integrations** → **Webhooks** → **New Webhook**.
2. Pick the channel for alerts, then **Copy Webhook URL**.

### Step 4: Download NewsTrader
- **Easiest:** on the GitHub page click the green **Code** button → **Download ZIP**. Unzip it to an easy place like `C:\NewsTrader`.
  - Don't run it from inside the ZIP.
  - Avoid `C:\Program Files`.
- **Or** with GitHub Desktop: *File → Clone repository → Harryisadag/newstrader*. This makes updating easier.

### Step 5: Start it
1. Open the NewsTrader folder and **double-click `run.bat`**.
2. The first time, it sets everything up. This takes 5–10 minutes and downloads about 1 GB.
   - If Windows SmartScreen pops up, click **More info → Run anyway**.
3. The NewsTrader window opens. The **Welcome** card shows three steps: click **Open API keys**, paste your keys, and click **Save keys**.
4. Go to **Logs → Run diagnostics**. Everything should be green (or yellow for optional things). Each red line tells you how to fix it.

That's it. From now on just double-click `run.bat`. The black window behind the app shows the engine log, and closing it closes NewsTrader.

The first time a TV stream starts, the speech model downloads once (Whisper large-v3, about 3 GB). Watch Logs → System status for "speech-to-text: ready".

---

## Using NewsTrader

The header always shows:
- **PAPER/LIVE** mode
- Market open/closed
- Equity and today's P/L
- Today's Claude cost vs your cap
- Trading status: **ACTIVE**, **MONITOR ONLY**, **HALTED** (daily loss limit) or **KILLED**
- The big red **KILL SWITCH**

| Tab | What it shows |
|---|---|
| **Live** | One pane per TV stream with the live transcript (on/off switch on each). Headlines from every text source scroll below, tagged with the tickers found and what happened (not relevant, same story, analysed, signal…). |
| **Signals** | Every AI call: ticker, bullish/bearish, confidence, outcome (bought / review / blocked and why), source and one-line reasoning. Click a row for the bull case, bear case, speaker and headline. The **Manual review** box has **Approve & Buy / Dismiss** buttons. **Test the AI** lets you paste any headline and see what Claude says (never trades). |
| **Portfolio** | Account numbers, an account-value chart (1D–1Y), open positions with their stop/target, and open orders. You can close a position or cancel an order by hand. |
| **Trade Log** | Every order (entries, exits, stop-loss/take-profit fills) with realized P/L. **Export CSV** saves it to the exports folder. |
| **Performance** | **Is the AI right?** Win rate by confidence level and by source, at +5 min / +1 hour / +1 trading day, for every signal (traded or not). |
| **Backtest** | Run the AI over historical Benzinga news for a date range and see what it would have done. It shows a cost estimate first. |
| **Settings** | Everything below, plus news sources (add any YouTube `/live` link, RSS feed, …), API keys and the live-trading lock. |
| **Logs** | System status (GPU, Alpaca, Claude, every stream/feed), **Run diagnostics**, the full log, and **Rejected AI responses** (anything Claude returned that failed validation). |

---

## Settings explained

All settings are saved to `config.json` and changed in the app (Settings tab). The defaults are listed here.

**Trading**
| Setting | Default | Meaning |
|---|---|---|
| Auto-trade | on | Off = still record and alert signals, but never place orders ("monitor only"). |
| Buy threshold | 80 | Bullish signals at/above this confidence are bought automatically. |
| Manual review threshold | 60 | Between this and the buy threshold you get a "manual review" alert instead. |
| Stop-loss / Take-profit | 2% / 4% | Every buy is a bracket order with these exits. |
| Sell on bearish | on | A strong bearish signal sells a position you already hold. |
| Allow short selling | off | Strong bearish signals open a short when you don't hold the stock. |

**Risk controls**
| Setting | Default | Meaning |
|---|---|---|
| Max $ per trade | $1,000 | Position size cap (whole shares only). |
| Max % of account in one stock | 10% | Includes what you already hold. |
| Max open positions | 5 | No new stocks once this many are held/pending. |
| Daily loss limit | $500 | When today's loss hits this, trading stops until the next trading day (and you get an alert). |
| Cooldown per ticker | 30 min | Don't trade the same stock again within this window. |
| Market hours only | on | Only trade 9:30–4:00 ET. If off, orders sent while the market is closed are queued by Alpaca for the next open. Alpaca doesn't allow bracket orders before/after hours. |
| Minimum share price | $1 | Skip penny stocks. |
| Blacklist / Whitelist | empty | Never trade / only trade these tickers. |

**AI engine**
| Setting | Default | Meaning |
|---|---|---|
| Claude model | Sonnet 5.5 | Haiku 4.5 is cheaper and faster; Opus 5.5 is smartest. |
| Thinking effort | low | Higher = more careful, slower, pricier (Sonnet/Opus only). |
| Daily spend cap | $5 | Hard stop on Claude calls for the day. |
| Same-story / same-signal window | 15 min | The same headline from several outlets is analysed once. The same ticker + direction is one signal, traded once. |
| Analyse macro news | on | Lets "Fed cuts rates" style news through even without a company name. |

**Transcription**
| Setting | Default | Meaning |
|---|---|---|
| Model | large-v3 on GPU, float16 | large-v3-turbo is faster. CPU is a slow fallback. |
| Max streams at once | 4 | The rest wait for a free slot. |
| Chunk length / context | 10 s / 60 s | How audio is cut, and how much recent transcript Claude sees. |
| YouTube cookies from browser | none | Only needed if YouTube asks to "sign in to confirm you're not a bot". |

**Alerts:** Windows pop-ups and Discord, each on/off. You can also turn each alert type on/off:
- trade placed
- trade filled/closed
- manual review
- errors
- daily loss limit hit
- spend cap hit
- kill switch

Use **Send a test alert** to check they work.

---

## How a trade happens

```
News item ──► Stage 1: local pre-filter (free)
              Does it mention a listed company, $TICKER, "(NYSE: X)", a known alias
              (e.g. "Jensen Huang" → NVDA) or a market-moving keyword?   no → ignored
          ──► Same story already seen from another outlet?               yes → counted, not re-analysed
          ──► Stage 2: Claude (structured JSON)
              For each affected stock: bull case → bear case → direction, confidence 0-100,
              speaker, one-line reasoning, time-sensitivity
          ──► Validation: valid JSON, real tradable US ticker, confidence 0-100, allowed values
              (anything else is rejected and listed in Logs → Rejected AI responses)
          ──► Same ticker + direction in the last 15 min? → merged into one signal
          ──► Bullish ≥ 80        → risk checks → bracket BUY (stop-loss + take-profit)
              60-79              → "manual review" alert (Approve/Dismiss in Signals)
              Bearish ≥ 80       → sell if you hold it (or short, if you turned that on)
```

**Risk checks, in order** (any failure blocks the trade, and the reason is shown in Signals):
1. Kill switch
2. Auto-trade
3. Daily loss limit
4. Market hours
5. Blacklist/whitelist
6. Cooldown
7. Pending order already open for that stock
8. Max open positions
9. Max $ per trade
10. Max % per stock
11. Buying power
12. Minimum price
13. Whole shares

---

## Kill switch and live trading

**KILL SWITCH** (top right) immediately:
- stops all new trades, and
- cancels **every open order** on your Alpaca account.

You can also tick **"close every position"** in its confirmation box.

Cancelling orders also removes the stop-loss/take-profit protecting your positions, so close them or watch them. Trading stays stopped, even after a restart, until you click **RE-ARM TRADING**.

**Live trading** (Settings → Live trading) needs all of these:
- live keys (`ALPACA_LIVE_API_KEY` / `ALPACA_LIVE_SECRET_KEY`)
- the confirmation box ticked
- the exact phrase `I UNDERSTAND THIS USES REAL MONEY`

When it's on, a red **LIVE · REAL MONEY** badge and banner show. It always resets to paper when the app restarts.

Our advice: stay on paper until the Performance tab shows the AI is reliably right over weeks.

---

## Make a real .exe

After `run.bat` has worked once:
1. Double-click **`build_exe.bat`** (3–10 minutes).
2. Your app is in **`dist\NewsTrader\NewsTrader.exe`**. Keep the whole `dist\NewsTrader` folder together, because the exe needs the `_internal` folder next to it.
3. Right-click `NewsTrader.exe` → **Send to → Desktop (create shortcut)**.

The exe keeps its own settings, database and keys in `%LOCALAPPDATA%\NewsTrader`. Paste your keys again in Settings → API keys, or copy your `.env` file next to `NewsTrader.exe`.

The exe isn't code-signed, so SmartScreen may warn the first time: click **More info → Run anyway**.

---

## Updating
Double-click **`update.bat`**. It:
- pulls the latest code if you used GitHub Desktop / git (otherwise it tells you to re-download the ZIP), and
- updates all packages, including **yt-dlp**, which needs updating whenever YouTube changes something.

If you use the .exe, run `build_exe.bat` again after updating.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `run.bat` says Python wasn't found | Reinstall Python 3.12 and tick **"Add python.exe to PATH"**. |
| A TV stream says "offline - not live right now" | That channel isn't broadcasting. It re-checks every 5 minutes. Turn on another stream or add one. |
| YouTube error "Sign in to confirm you're not a bot" | Settings → Transcription → **YouTube cookies from browser** → pick the browser where you're logged in to YouTube. Then run `update.bat`. |
| Streams fail after working before | Run **`update.bat`** (YouTube changed; yt-dlp needs updating). |
| "GPU unavailable … using CPU" | Update the NVIDIA driver, restart, then run Logs → Run diagnostics. If it still fails, run `update.bat`. |
| "Transcription falling behind" | Turn off a stream, or switch to `large-v3-turbo`. |
| An RSS feed shows a red dot | That site blocked or changed its feed. Turn it off or remove it in Settings → News sources. |
| No signals at all | Check Logs → System status: Claude key OK? Ticker list loaded (needs Alpaca keys)? Spend cap reached? |
| Nothing trades | Signals → look at the **Outcome** column. The reason (market closed, cooldown, max positions…) is there. |
| Desktop pop-ups don't appear | Turn off Windows "Do not disturb" / Focus. Use Settings → Alerts → Send a test alert. |

For anything else, look at **Logs**. The log files are in the `data\logs` folder (Logs → Open log files).

---

## Truth Social and X (Twitter)

**Truth Social** has no official API. NewsTrader comes with Donald Trump's posts preloaded through the free public archive feed at trumpstruth.org. Free feeds can lag a bit behind the real posts.

**X / Twitter** has no free, legal way to read posts automatically (the free API can only post). Your options:
1. **Buy X API access** (Basic tier, about $200/month). Paste the bearer token in Settings → API keys (`X_BEARER_TOKEN`). Then add an **X account (API)** source with the username. The adapter is already built in.
2. **Use an RSS service** such as rss.app (paid) that turns an X account into a feed. Add the feed URL as a **Social** source.
3. Self-hosted scrapers (Nitter/RSSHub) break often and are against X's terms, so they aren't included.

You can add more social accounts the same way (any RSS feed works as a "Social" source, with the poster's name as the speaker).

---

## What it costs
- **Alpaca paper trading and news:** free.
- **Claude:** each analysed story costs roughly **$0.002–$0.02** with Sonnet 5.5 at low effort, and about half that with Haiku 4.5. Most news is filtered out for free before reaching Claude. A busy day with all default sources is typically **$1–$5**, and the **daily spend cap ($5)** is a hard stop.
- **Backtests** show a cost estimate first and stop at the budget you set. They don't count toward the live daily cap.

> **Backtest caveat:** Claude was trained on data up to a cutoff date. For older news it may already "know" what happened next, so treat backtests on older dates as optimistic.

---

## Where your files are

| What | Dev mode (`run.bat`) | .exe |
|---|---|---|
| API keys | `.env` in the NewsTrader folder | `%LOCALAPPDATA%\NewsTrader\.env` (or `.env` next to the exe) |
| Settings | `data\config.json` | `%LOCALAPPDATA%\NewsTrader\config.json` |
| Database (signals, trades, transcripts) | `data\newstrader.db` | `%LOCALAPPDATA%\NewsTrader\newstrader.db` |
| Logs | `data\logs\` | `%LOCALAPPDATA%\NewsTrader\logs\` |
| CSV exports | `data\exports\` | `%LOCALAPPDATA%\NewsTrader\exports\` |
| Whisper models (~3 GB) | `data\models\` | `%LOCALAPPDATA%\NewsTrader\models\` |

`.env` and `data\` are never uploaded to GitHub (they're in `.gitignore`).

---

## For developers

- **Run tests:** `.venv\Scripts\python -m pip install -r requirements-dev.txt`, then `.venv\Scripts\python -m pytest`. GitHub Actions also runs them on Windows and Linux for every push.
- **Lint:** `.venv\Scripts\ruff check .`
- **Modes:**
  - `run.bat --browser` opens the dashboard in your normal browser
  - `run.bat --headless` runs the server only
  - `set NEWSTRADER_FAKE_BROKER=1` uses an offline fake Alpaca account for UI work (never places real orders)
- **Exe build in CI:** Actions → **build-exe** → Run workflow. It builds and smoke-tests the exe on Windows (uploading the ~1 GB result is optional).

**Layout**

```
newstrader/
  app.py              desktop window (pywebview) + local server (FastAPI/uvicorn) startup
  config.py keys.py   settings (config.json) and API keys (.env)
  db.py state.py      SQLite + kill switch / daily halt / paper-live state
  orchestrator.py     starts every service
  api/                REST + websocket endpoints (token-protected, 127.0.0.1 only)
  sources/            RSS, Alpaca news websocket, social RSS, X API, presets
  audio/              yt-dlp resolver, ffmpeg capture, chunker, faster-whisper, stream manager
  ai/                 ticker table, pre-filter, de-dupe, Claude client, prompts, validator, costs, pipeline
  trading/            Alpaca broker, live-trading lock, risk checks, trader, P/L
  alerts/             Windows pop-ups, Discord, alert manager
  performance/        price checkpoints and win-rate stats
  backtest/           historical replay + bracket simulator
  web/                the dashboard (HTML/CSS/Alpine.js/Chart.js, no build step)
tests/                pytest suite (risk rules, AI validation, pipeline, audio, alerts, backtest...)
```
