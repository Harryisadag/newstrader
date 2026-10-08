# NewsTrader

NewsTrader is a desktop app for **Windows and Mac** that:

1. **Watches live news** (about 120 ready-made sources, 25 on by default; turn any on/off in Settings):
   - TV livestreams (Bloomberg, Yahoo Finance, Schwab Network, LiveNOW from FOX, …) and international channels (DW, France 24, NHK World, CNBC-TV18, …). Non-English TV is translated to English as it's transcribed.
   - Live events: White House briefings and Trump remarks, Fed press conferences, C-SPAN, Trump interviews and rallies. These channels only take a transcription slot while they're live.
   - Alpaca/Benzinga real-time news, and RSS feeds (CNBC, MarketWatch, WSJ, Reuters, BBC, FDA, Fed, international business news, …)
   - Social posts (Truth Social)
2. **Transcribes TV audio in real time**. It uses your NVIDIA GPU on Windows, or the Apple GPU on a Mac with an M-series chip.
3. **Decides whether each story is good or bad for a specific stock**, using machine learning that runs free on your own computer:
   - **FinBERT** is a language model trained on financial news. It reads the wording about each company and scores it as positive, negative or neutral.
   - A **price model** that *you* train from free Alpaca history. It learns how stocks actually moved (compared with the S&P 500) in the hour after similar headlines, and it is tested on news it never saw.
   - Optional: switch the engine to **Claude** (paid, needs an Anthropic key). Claude reads context better and argues the bull and bear case before deciding.
4. **Places PAPER trades on Alpaca** when confidence is high enough. Every buy has a stop-loss and take-profit, and strict risk limits apply.
5. **Tracks whether the AI is actually right.** It checks the price 5 minutes, 1 hour and 1 trading day after every call. It can also **backtest** on historical news.
6. **Watches the market**: sudden price spikes on unusual volume, big moves in the whole market or world markets, and today's top movers, with the news behind them (Market tab and alerts).

> ⚠️ **Paper trading only by default.** NewsTrader always starts in **PAPER** mode (fake money). Real-money trading is locked in the code. Unlocking it takes all three of these:
> - separate live keys
> - turning it on in Settings → Live trading
> - typing `I UNDERSTAND THIS USES REAL MONEY`
>
> Even then, it switches back to paper every time the app restarts. NewsTrader is an experiment, not financial advice.

---

## Download the ready-made app (easiest)

You don't need Python to use a release:
1. Download the zip for your computer from the **[Releases page](https://github.com/Harryisadag/newstrader/releases)**:
   - Windows 10 / 11: `windows-x64`
   - Mac with an M-series chip, macOS 15 Sequoia or newer: `mac-apple-silicon`
   - Intel Mac, macOS 15 Sequoia or newer: `mac-intel`
2. Unzip it and open NewsTrader.

[RELEASES.md](RELEASES.md) is the catalogue of every version. It covers what changed, which file to pick, the "Run anyway" / "Open Anyway" steps for apps that aren't from a store, and how to update. To run the latest code instead, or on a Mac with macOS 14 or older, follow the setup below.

## Contents
1. [Setup on Windows (one time, ~20 minutes)](#setup-on-windows-one-time-20-minutes)
2. [Setup on a Mac (one time, ~20 minutes)](#setup-on-a-mac-one-time-20-minutes)
3. [Using NewsTrader](#using-newstrader)
4. [The AI engine: local machine learning or Claude](#the-ai-engine-local-machine-learning-or-claude)
5. [Settings explained](#settings-explained)
6. [How a trade happens](#how-a-trade-happens)
7. [Kill switch and live trading](#kill-switch-and-live-trading)
8. [Make a real app (.exe / .app)](#make-a-real-app-exe--app)
9. [Updating](#updating)
10. [Troubleshooting](#troubleshooting)
11. [Truth Social and X (Twitter)](#truth-social-and-x-twitter)
12. [What it costs](#what-it-costs)
13. [Where your files are](#where-your-files-are)
14. [For developers](#for-developers)

---

## Setup on Windows (one time, ~20 minutes)

You need Windows 10/11, an NVIDIA graphics card for fast transcription (yours is an RTX 5070 Ti), and an internet connection.

### Step 1: Install Python 3.12
1. Go to **https://www.python.org/downloads/** and download the **Python 3.12.x Windows installer (64-bit)**. (3.13 also works; 3.12 is the tested version.)
2. Run it. **On the first screen, tick "Add python.exe to PATH".** Then click **Install Now**.
3. Click **Close** when it's done.

### Step 2: Update your NVIDIA driver
RTX 50-series cards need a 2025-or-newer driver.
1. Open the **NVIDIA App** (or get it at https://www.nvidia.com/en-us/software/nvidia-app/).
2. Go to **Drivers**, install the latest **Game Ready** or **Studio** driver, and restart if asked.

You do **not** need to install the CUDA Toolkit or ffmpeg. NewsTrader installs the CUDA libraries it needs, plus its own copy of ffmpeg, automatically.

### Step 3: Get your keys
See [Getting your keys](#getting-your-keys) below.

### Step 4: Download NewsTrader
- **Easiest:** on the GitHub page click the green **Code** button → **Download ZIP**. Unzip it to an easy place like `C:\NewsTrader`.
  - Don't run it from inside the ZIP.
  - Avoid `C:\Program Files`.
- **Or** with GitHub Desktop: *File → Clone repository → Harryisadag/newstrader*. This makes updating easier.

### Step 5: Start it
1. Open the NewsTrader folder and **double-click `run.bat`**.
2. The first time, it sets everything up. This takes 5–10 minutes and downloads about 1 GB.
   - If Windows SmartScreen pops up, click **More info → Run anyway**.
3. The NewsTrader window opens. The **Welcome** card shows three steps: click **Open API keys**, paste your Alpaca keys, and click **Save keys**.
4. Go to **Logs → Run diagnostics**. Everything should be green (or yellow for optional things). Each red line tells you how to fix it.

That's it. From now on just double-click `run.bat`. The black window behind the app shows the engine log, and closing it closes NewsTrader.

Two downloads happen once, in the background:
- the FinBERT sentiment model (about 110 MB) when the app first starts
- the speech model (Whisper large-v3, about 3 GB) the first time a TV stream starts

Watch Logs → System status for "Local ML" and "speech-to-text: ready".

---

## Setup on a Mac (one time, ~20 minutes)

Any Mac works.
- **Apple Silicon (M1–M4) on macOS 14 Sonoma or newer:** TV audio is transcribed fast on the Apple GPU.
- **Intel Macs, or macOS 13:** transcription runs on the CPU (slower). Pick a smaller Whisper model in Settings → Transcription.

Everything else works the same.

### Step 1: Install Python 3.12
1. Go to **https://www.python.org/downloads/macos/** and download the **Python 3.12 "macOS 64-bit universal2 installer"**. 3.13 also works. On an Intel Mac, don't use 3.14 yet.
2. Open the downloaded `.pkg` and click through the installer.
3. When it finishes, a Finder window opens. Double-click **Install Certificates.command** in it. This lets Python make secure internet connections.

(If you use Homebrew, `brew install python@3.12` works too.)

### Step 2: Get your keys
See [Getting your keys](#getting-your-keys) below.

### Step 3: Download NewsTrader
On the GitHub page click the green **Code** button → **Download ZIP**. Double-click the ZIP to unzip it, then move the `newstrader` folder somewhere easy, like your **Documents** folder. (Or clone it with GitHub Desktop.)

### Step 4: Start it
1. Open the NewsTrader folder and **double-click `run.command`**.
   - If macOS says it "can't be opened because it is from an unidentified developer", click **Done**. Then open **System Settings → Privacy & Security**, scroll down, and click **Open Anyway** next to `run.command`. Enter your password.
   - If it says "permission denied", open Terminal and type `chmod +x ` (with a space at the end). Drag `run.command` into the Terminal window, then press Enter. This happens when the ZIP was unpacked by a tool that drops file permissions.
2. A Terminal window opens and sets everything up. This takes 5–10 minutes the first time. On an M-series Mac it also installs the Apple-GPU speech engine.
3. The NewsTrader window opens. Paste your Alpaca keys in **Settings → API keys**, then go to **Logs → Run diagnostics**.

From now on just double-click `run.command`. Closing the Terminal window closes NewsTrader.

Desktop pop-ups on a Mac come from "Script Editor". The first time, allow them in **System Settings → Notifications → Script Editor**.

---

## Getting your keys

Keep these private. You paste them into the app (Settings → API keys), which saves them in a `.env` file on your computer.

**Alpaca (paper trading, free) — required**
1. Sign up / log in at **https://app.alpaca.markets**.
2. Top-left, make sure you're in your **Paper** account (not Live).
3. On the right side of the home page find **API Keys** → **Generate New Keys**.
4. Copy the **Key** and the **Secret**. The secret is shown only once.

**Discord webhook (optional, for phone alerts)**
1. In Discord, open your server → **Server Settings** → **Integrations** → **Webhooks** → **New Webhook**.
2. Pick the channel for alerts, then **Copy Webhook URL**.

**Anthropic / Claude (optional — only if you switch the AI engine to Claude)**
1. Go to **https://platform.claude.com** and sign in.
2. Under **Billing**, add a little credit. $10 lasts a while (see [costs](#what-it-costs)).
3. Recommended: under **Limits**, set a monthly spending limit as a second safety net.
4. Go to **API Keys** → **Create Key**, then copy it. It starts with `sk-ant-`.

---

## Using NewsTrader

The header always shows:
- **PAPER/LIVE** mode
- Market open/closed
- Equity and today's P/L
- The AI engine. With Claude, it shows today's Claude cost vs your cap instead.
- Trading status: **ACTIVE**, **MONITOR ONLY**, **HALTED** (daily loss limit) or **KILLED**
- The big red **KILL SWITCH**

| Tab | What it shows |
|---|---|
| **Live** | One pane per TV stream with the live transcript (on/off switch on each, and a language badge such as DE → EN when it's translated). Live-event channels that are off air wait in a compact row and get a pane when they go live. Headlines from every text source scroll below, tagged with the tickers found and what happened (not relevant, same story, analysed, signal…). |
| **Signals** | Every AI call: ticker, bullish/bearish, confidence, outcome (bought / review / blocked and why), source and one-line reasoning. The news event it recognised (e.g. "Cut its forecast") shows as a blue label, and warnings such as "not confirmed" in orange. Click a row for details, the speaker and the headline. The **Manual review** box has **Approve & Buy / Dismiss** buttons. **Test the AI** lets you paste any headline and see what the engine says about each company, including why one came out neutral (never trades). |
| **Market** | The S&P 500, Nasdaq 100, Russell 2000 and Dow (through their ETFs), world markets (Japan, China, Germany, UK, India, Brazil, Europe, Korea), today's top movers, every sudden spike found with the news behind it, and the list of stocks being watched. |
| **Portfolio** | Account numbers, an account-value chart (1D–1Y), open positions with their stop/target, and open orders. You can close a position or cancel an order by hand. |
| **Trade Log** | Every order (entries, exits, stop-loss/take-profit fills) with realized P/L. **Export CSV** saves it to the exports folder. |
| **Performance** | **Is the AI right?** Win rate by confidence level and by source, at +5 min / +1 hour / +1 trading day, for every signal (traded or not). |
| **Backtest** | **Local ML model**: train or retrain the price model and see how it did on news it never saw. **Backtests**: run the AI over historical Benzinga news for a date range and see what it would have done. |
| **Settings** | Everything below, plus news sources (search, filter by region/kind, turn whole groups on/off, edit or add any YouTube `/live` link or RSS feed), API keys and the live-trading lock. |
| **Logs** | System status (GPU, Alpaca, AI engine, every stream/feed), **Run diagnostics**, the full log, and **Rejected AI responses** (anything that failed validation). |

---

## The AI engine: local machine learning or Claude

Pick it in **Settings → AI engine**. The default is **Local machine learning**.

### Local machine learning (free, default)
It runs on your computer, needs no API key, and keeps working offline after the first download. For every company named in a story:

1. It picks out the **headline plus the sentences about that company**. In "Apple sues Samsung", Apple and Samsung each get their own read.
2. **News-event rules** look for the kind of news that actually moves stocks: earnings beats and misses (including the numbers, e.g. "EPS $1.20 vs $1.35 est"), raised or cut forecasts, buyouts, analyst up/downgrades and price targets, FDA decisions and trial results, share offerings, buybacks, dividend changes, index changes, lawsuits and fines, recalls, hacks, strikes, contract wins, bankruptcy risk and more (about 70 kinds). They also work out **who** the news is for:
   - the company being bought goes up, the buyer is neutral
   - the analyst firm is neutral, the stock it rated gets the call
   - the winner of a contract goes up, the company it replaced goes down; a company only named in passing gets nothing
   - a forecast cut beats an earnings beat in the same headline ("beats estimates but cuts outlook" is bearish)
   - "denies", "no plans to", "fails to" turn the event around; "reportedly" / "sources say" lowers confidence
   - stories about a move that already happened ("Why Nvidia stock jumped"), opinion pieces, round-ups of many stocks and law-firm adverts come out neutral
3. **FinBERT** scores the wording, for example "92% positive / 3% negative / 5% neutral".
   - With a recognised event, the event sets the direction, and FinBERT agreeing or disagreeing moves the confidence up or down a little.
   - With **no** recognised event, the wording alone decides, but those signals are always sent to **manual review**, never auto-bought.
   - Confidence is a bit lower when the company is only mentioned in the article, not the headline.
4. If you've trained the **price model** (below), it takes over the decision for written news (it also learns from the event the rules found). If the model and the event point opposite ways, the signal goes to manual review. It answers: "after headlines like this, how often did the stock beat the S&P 500 over the next hour?" Confidence is then that tested probability. TV transcripts are always scored on wording only, because the price model learned from written articles.

What it can't do (Claude can):
- understand context, such as whether news was already expected, or who is speaking
- handle US macro news that names no company ("Fed cuts rates"). The local engine skips those stories. International macro news ("Bank of Japan raises rates") can give a signal for that country's US-listed fund (EWJ, FXI, EWG…), always for manual review.
- read news that isn't in English. Non-English text feeds are skipped by the local engine (TV is translated, so that works). Claude reads any language.

How good is it? On 150 labelled headlines written by people who never saw the rules, the right call (good / bad / neutral for each company) was 66% at the first look, vs 45% for v0.2. Those misses were then used to fix general gaps (it's about 92% on that set now, but that number is flattering because the set was used for the fixes). In those tests the engine never auto-traded the wrong way; unclear cases go to manual review. You can run the tests yourself: `python scripts/eval_detection.py --compare`.

### Training the price model (recommended)
Go to **Backtest → Local ML model → Train the price model**. It's free and runs in the background while trading continues.

1. It downloads about 6 months of Benzinga articles plus minute-by-minute prices from Alpaca, using your paper keys.
   - It takes a short sample from every hour of every trading day, and uses the same text the live feed sees.
   - The first time takes 5–20 minutes; later trainings only download the new days.
2. For each headline it measures what the stock did from 1 minute after the news to 1 hour later, minus what the S&P 500 did.
3. It learns from the older 80% of headlines, using only the ones where the stock clearly reacted (moved 0.3%+ vs the market).
4. Then it **tests itself on the newest 20%, which it never saw**. That test uses *every* headline, including ones where the stock barely moved, because live trading can't know that in advance.
5. The card shows:
   - how often it picked the right direction on that unseen news, next to the score you'd get by always guessing the same thing
   - its "edge over chance" (AUC, where 0.5 is a coin flip) with a likely range
   - a table of what each confidence threshold would have done, including what plain buying on bullish calls earned

The price model is **only used if it passed that test**. Its edge over chance has to hold up when the test days are reshuffled (so it isn't luck), over at least 10 trading days. Honestly, predicting the next hour from headlines is hard, so don't be surprised if it doesn't pass. The usual fix is more history (Settings → AI engine → Training history).

**If it doesn't pass:** the engine keeps scoring with FinBERT alone. Those signals then go to **manual review instead of auto-buying**, because your own history just showed the wording didn't reliably predict moves. You can change this with Settings → AI engine → *Sentiment-only signals can auto-buy*. While LIVE (real-money) trading is on, sentiment-only signals always need your approval.

> **Confidence with the price model is lower — that's normal.** It's an honest, tested probability, so 60–70 is already a strong call. Use the threshold table on the Backtest tab to pick your **buy** and **review** thresholds in Settings → Trading. With FinBERT alone, the default 80/60 thresholds make sense.

Retrain every month or so, so the model learns from recent news.

### Claude (optional, paid)
Claude reads each story, argues the bull case and the bear case, then gives a direction and a 0–100 confidence. It understands context much better, but every analysed story costs money (see [costs](#what-it-costs)) and it needs an Anthropic key. A daily spend cap is a hard stop.

You can backtest either engine on the Backtest tab. With the local engine, only test dates **after** the price model's training range. It has already seen the news inside that range, and the app warns you if the dates overlap.

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
| Don't chase | 3% | If the price already moved this much in the signal's direction since the news came out, the signal goes to manual review instead of auto-trading. 0 = off. |

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
| AI engine | Local machine learning | Or Claude (paid). |
| Sentiment model | FinBERT | Or the built-in word list. It needs no download but is crude, and its confidence is capped at 79, so it never auto-buys on default settings. |
| Use the price-trained model | Auto | Auto = only if it passed its test. Always / Never are also available. |
| Sentiment-only signals can auto-buy | Auto | Yes, unless a trained price model failed its test (then manual review only). Or Yes always / No (review only). |
| Training history | 180 days | How much history the price model learns from. |
| Measure each stock's move over | 1 hour | 30 min / 1 hour / 2 hours. |
| Smallest move that counts | 0.3% | Moves smaller than this (vs the S&P 500) are left out of training. |
| Claude model / effort / daily spend cap | Sonnet 5.5 / low / $5 | Only when Claude is the engine. |
| Same-story / same-signal window | 15 min | The same headline from several outlets is analysed once. The same ticker + direction is one signal, traded once. |
| Max stocks per story | 3 | |
| Recognise news events | on | The event rules above. Off = the v0.2 behaviour (wording score only). |
| Country funds for world news | on | International macro news can create signals for country ETFs (always manual review). |

**Transcription**
| Setting | Default | Meaning |
|---|---|---|
| Model | Windows: large-v3 on the NVIDIA GPU. Mac: large-v3-turbo, Auto | large-v3-turbo is ~4x faster with nearly the same accuracy. Intel Macs: try small or base. |
| Device | Windows: CUDA. Mac: Auto | Auto picks the NVIDIA GPU, the Apple GPU, or the CPU. |
| Max streams at once | 4 | The rest wait for a free slot. |
| Chunk length / context | 10 s / 60 s | How audio is cut, and how much recent transcript is analysed. |
| YouTube cookies from browser | none | Only needed if YouTube asks to "sign in to confirm you're not a bot". Chrome, Edge, Firefox, Brave or Safari. |

**News sources:** each source has a region, a kind (TV, news, social, live events), a language and, for TV, a **Translate to English** switch. Live-event channels (White House, Fed…) only use one of the "max streams at once" slots while they're live, and take one from the lowest-priority always-on stream if needed.

**Market monitor**
| Setting | Default | Meaning |
|---|---|---|
| Market monitor | on | Scans every 60 seconds during market hours (pre/after-market scanning is optional, the data there is thin). |
| What to watch | positions, stocks with signals in the last 2 hours, your watchlist, today's top 20 movers | Up to 100 stocks. |
| Sudden spike | 3% within 5 minutes, on 3x normal volume | Stocks under $2 are skipped. |
| Market-wide move | 1% within 15 minutes, or each 1% step in the day's change | For the S&P 500, Nasdaq 100, Russell 2000 and Dow ETFs. |
| Look up the news behind a spike | on | Shows the latest Benzinga headlines for that stock. |
| Alert cooldown | 30 min per stock | Plus at most 3 spike alerts per scan. |

The free Alpaca data plan (IEX) sees part of the market's trading, so volume numbers are lower than the full market's. Spike detection compares a stock with its own normal volume, so it still works.

**Display:** times in the app can follow this computer's time zone, New York (market) time, or UTC. **Check for updates** (on) looks at GitHub once a day and shows a banner when a newer NewsTrader is out. It never installs anything by itself.

**Alerts:** desktop pop-ups (Windows or Mac) and Discord, each on/off. You can also turn each alert type on/off:
- trade placed
- trade filled/closed
- manual review
- errors
- daily loss limit hit
- spend cap hit
- kill switch
- sudden spike in a watched stock
- big move in the whole market

Use **Send a test alert** to check they work.

---

## How a trade happens

```
News item ──► Stage 1: local pre-filter (free)
              Does it mention a listed company, $TICKER, "(NYSE: X)" or a known alias
              (e.g. "Jensen Huang" → NVDA)?                               no → ignored
          ──► Same story already seen from another outlet?               yes → counted, not re-analysed
          ──► Stage 2: AI engine
              Local ML: news-event rules + FinBERT read the sentences about each company
                        (+ the trained price model)
              Claude:   bull case → bear case → direction, confidence 0-100
          ──► Validation: real tradable US ticker, confidence 0-100, allowed values
              (anything else is rejected and listed in Logs → Rejected AI responses)
          ──► Same ticker + direction in the last 15 min? → merged into one signal
          ──► Bullish ≥ buy threshold   → risk checks → price already ran > 3%? → review
                                          → bracket BUY (stop-loss + take-profit)
              between the thresholds    → "manual review" alert (Approve/Dismiss in Signals)
              Bearish ≥ buy threshold   → sell if you hold it (or short, if you turned that on)
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

## Make a real app (.exe / .app)

### Windows
After `run.bat` has worked once:
1. Double-click **`build_exe.bat`** (3–10 minutes).
2. Your app is in **`dist\NewsTrader\NewsTrader.exe`**. Keep the whole `dist\NewsTrader` folder together, because the exe needs the `_internal` folder next to it.
3. Right-click `NewsTrader.exe` → **Send to → Desktop (create shortcut)**.

The exe keeps its own settings, database and keys in `%LOCALAPPDATA%\NewsTrader`. Paste your keys again in Settings → API keys, or copy your `.env` file next to `NewsTrader.exe`.

The exe isn't code-signed, so SmartScreen may warn the first time: click **More info → Run anyway**.

### Mac
After `run.command` has worked once:
1. Double-click **`build_app.command`** (3–10 minutes).
2. Your app is **`dist/NewsTrader.app`**. Drag it into your **Applications** folder.
3. The first time you open it, macOS blocks it because it isn't from the App Store:
   - Click **Done**.
   - Open **System Settings → Privacy & Security**, scroll down, and click **Open Anyway**. The button shows for about an hour after you tried to open the app.
   - Enter your password.

   On macOS 15 and newer, right-click → Open no longer skips this.

The app keeps its settings, database and keys in `~/Library/Application Support/NewsTrader`. Paste your keys again in Settings → API keys.

---

## Updating
The app checks GitHub once a day and shows a banner at the top when a newer version is out, with a link to it (turn this off in Settings → Display). Nothing is downloaded or installed by itself.

Using a downloaded release? Download the newest one instead (see "Updating" in [RELEASES.md](RELEASES.md)).

From the source code: double-click **`update.bat`** (Windows) or **`update.command`** (Mac). It:
- pulls the latest code if you used GitHub Desktop / git (otherwise it tells you to re-download the ZIP), and
- updates all packages, including **yt-dlp**, which needs updating whenever YouTube changes something.

If you use the built app, run `build_exe.bat` / `build_app.command` again after updating.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `run.bat` says Python wasn't found | Reinstall Python 3.12 and tick **"Add python.exe to PATH"**. |
| `run.command` says Python wasn't found | Install Python 3.12 from python.org (Mac section, Step 1). |
| Mac: "can't be opened" / "unidentified developer" | System Settings → Privacy & Security → **Open Anyway** (see the Mac setup). |
| Mac: certificate / SSL errors | Run **Install Certificates.command** in `/Applications/Python 3.12`. |
| "FinBERT unavailable … using the built-in word list" | The one-time ~110 MB download from huggingface.co failed. Check your internet, then click **Retry FinBERT download** on the Backtest tab. |
| The price model "isn't used" | It didn't beat chance on unseen news. Train with more history (Settings → AI engine → Training history), or switch "Use the price-trained model" to Always at your own risk. |
| Training fails with Alpaca errors | You hit Alpaca's free rate limit or a network blip. Days already downloaded are saved, so click Train again later. |
| A TV stream says "offline - not live right now" | That channel isn't broadcasting. It re-checks every 5 minutes. Turn on another stream or add one. |
| YouTube error "Sign in to confirm you're not a bot" | Settings → Transcription → **YouTube cookies from browser** → pick the browser where you're logged in to YouTube. Then update (source version: run the update script; downloaded app: get the newest release). On a Mac, Chrome asks for Keychain access, and Safari needs Full Disk Access for NewsTrader/Terminal. |
| Streams fail after working before | YouTube changed and yt-dlp needs updating. Source version: run the **update** script. Downloaded app: get the newest release from the Releases page. |
| "GPU unavailable … using CPU" (Windows) | No NVIDIA card? That's expected: it uses the CPU. Otherwise update the NVIDIA driver, restart, then run Logs → Run diagnostics. If it still fails, run `update.bat` (or get the newest release). |
| Mac: "Apple-GPU speech engine isn't installed" | Run `run.command` again (it needs macOS 14+ on an M-series Mac). |
| "Transcription falling behind" | Turn off a stream, or switch to `large-v3-turbo` (or `small` on an Intel Mac). |
| An RSS feed shows a red dot | That site blocked or changed its feed. Turn it off or remove it in Settings → News sources. |
| Headlines say "not in English - the local engine reads English only" | That feed is in another language. Turn it off, or switch Settings → AI engine to Claude. For TV, turn on **Translate to English** for that stream. |
| The Market tab is empty | It needs your Alpaca keys and only scans during market hours (unless pre/after-market scanning is on). |
| No signals at all | Check Logs → System status. Is the ticker list loaded (needs Alpaca keys)? Is the AI engine OK? With Claude: is the key OK and the spend cap not reached? |
| Nothing trades | Signals → look at the **Outcome** column. The reason (market closed, cooldown, max positions, confidence below threshold…) is there. |
| Desktop pop-ups don't appear | Turn off "Do not disturb" / Focus. On a Mac, allow notifications for **Script Editor**. Use Settings → Alerts → Send a test alert. |

For anything else, look at **Logs**. Logs → Open log files shows the log folder (see [Where your files are](#where-your-files-are)).

---

## Truth Social and X (Twitter)

**Truth Social** has no official API. NewsTrader comes with Donald Trump's posts preloaded through the free public archive feed at trumpstruth.org. Free feeds can lag a bit behind the real posts.

**Trump interviews and remarks** come in through:
- live TV: The White House channel (briefings and remarks, on by default), Fox News and Fox Business live events, Right Side Broadcasting (rallies and speeches), Forbes Breaking News, PBS NewsHour, C-SPAN. These are "live event" channels: they only use a transcription slot while they're live.
- news: a "Trump interviews" Google News feed (on by default) and White House announcements.

**X / Twitter** has no free, legal way to read posts automatically (the free API can only post). Your options:
1. **Buy X API access** (Basic tier, about $200/month). Paste the bearer token in Settings → API keys (`X_BEARER_TOKEN`). Then add an **X account (API)** source with the username. The adapter is already built in.
2. **Use an RSS service** such as rss.app (paid) that turns an X account into a feed. Add the feed URL as a **Social** source.
3. Self-hosted scrapers (Nitter/RSSHub) break often and are against X's terms, so they aren't included.

You can add more social accounts the same way (any RSS feed works as a "Social" source, with the poster's name as the speaker).

---

## What it costs
- **Alpaca paper trading, news and price history:** free.
- **Local ML engine (default):** free. It uses your computer's CPU, and needs one ~110 MB download.
- **Claude (optional):** each analysed story costs roughly **$0.002–$0.02** with Sonnet 5.5 at low effort, and about half that with Haiku 4.5.
  - Most news is filtered out for free before reaching Claude.
  - A busy day with all default sources is typically **$1–$5**.
  - The **daily spend cap ($5)** is a hard stop.
  - Claude backtests show a cost estimate first and stop at the budget you set. They don't count toward the live daily cap.

> **Backtest caveats:**
> - **Claude** was trained on data up to a cutoff date, so for older news it may already "know" what happened next.
> - The **local price model** has seen the news inside its training range.
>
> Treat backtests over those dates as optimistic.

---

## Where your files are

| What | Dev mode (`run.bat` / `run.command`) | Windows .exe | Mac .app |
|---|---|---|---|
| API keys | `.env` in the NewsTrader folder | `%LOCALAPPDATA%\NewsTrader\.env` (or `.env` next to the exe) | `~/Library/Application Support/NewsTrader/.env` |
| Settings | `data/config.json` | `%LOCALAPPDATA%\NewsTrader\config.json` | `~/Library/Application Support/NewsTrader/config.json` |
| Database (signals, trades, transcripts, training data) | `data/newstrader.db` | `%LOCALAPPDATA%\NewsTrader\newstrader.db` | `…/NewsTrader/newstrader.db` |
| Logs | `data/logs/` | `%LOCALAPPDATA%\NewsTrader\logs\` | `…/NewsTrader/logs/` |
| CSV exports | `data/exports/` | `%LOCALAPPDATA%\NewsTrader\exports\` | `…/NewsTrader/exports/` |
| Models (Whisper ~3 GB, FinBERT ~110 MB, your price model) | `data/models/` | `%LOCALAPPDATA%\NewsTrader\models\` | `…/NewsTrader/models/` |

On a Mac, Apple-GPU Whisper models are kept in the Hugging Face cache (`~/.cache/huggingface`).

`.env` and `data/` are never uploaded to GitHub (they're in `.gitignore`).

---

## For developers

- **Run tests:**
  - Windows: `.venv\Scripts\python -m pip install -r requirements-dev.txt`, then `.venv\Scripts\python -m pytest`
  - Mac: `.venv/bin/python -m pip install -r requirements-dev.txt`, then `.venv/bin/python -m pytest`

  GitHub Actions runs them on Windows, Linux and macOS for every push. It also runs a job that downloads the real FinBERT model, checks it on known headlines, and checks that the Apple-GPU Whisper repos exist (`scripts/check_models.py`).
- **Lint:** `ruff check .`
- **Modes:**
  - `run.bat --browser` / `./run.command --browser` opens the dashboard in your normal browser
  - `--headless` runs the server only
  - `NEWSTRADER_FAKE_BROKER=1` uses an offline fake Alpaca account for UI work (never places real orders)
- **App builds in CI:** Actions → **build-exe** (Windows) or **build-app-mac** (Apple Silicon) → Run workflow. Each one builds the app, then smoke-tests it: it starts, finds ffmpeg/yt-dlp/deno, and loads FinBERT.
- **Making a release:**
  1. Raise `__version__` in `newstrader/__init__.py` and add a `## vX.Y.Z` entry to `RELEASES.md`.
  2. Run Actions → **release** → Run workflow.

  A published version is never rebuilt, so to ship a fix (or a newer yt-dlp after a YouTube change), raise the version (e.g. 0.2.1 → 0.2.2) and add its entry.

  The workflow builds Windows, Apple Silicon and Intel Mac apps and smoke-tests each one. It attaches the zips and SHA-256 checksums to a draft release, then publishes it. The release notes come from `RELEASES.md` (`scripts/release_notes.py`).

**Layout**

```
newstrader/
  app.py              desktop window (pywebview) + local server (FastAPI/uvicorn) startup
  config.py keys.py   settings (config.json) and API keys (.env)
  db.py state.py      SQLite + kill switch / daily halt / paper-live state
  orchestrator.py     starts every service
  api/                REST + websocket endpoints (token-protected, 127.0.0.1 only)
  sources/            RSS, Alpaca news websocket, social RSS, X API, presets
  audio/              yt-dlp resolver, ffmpeg capture, chunker, Whisper (CUDA / Apple GPU / CPU), stream manager
  ai/                 ticker table, pre-filter, de-dupe, engine choice, Claude client, prompts, validator, pipeline
  ml/                 local engine: FinBERT sentiment, text targeting, price model, training data, trainer
  trading/            Alpaca broker, live-trading lock, risk checks, trader, P/L
  alerts/             desktop pop-ups (Windows / Mac), Discord, alert manager
  performance/        price checkpoints and win-rate stats
  backtest/           historical replay + bracket simulator
  web/                the dashboard (HTML/CSS/Alpine.js/Chart.js, no build step)
tests/                pytest suite (risk rules, AI validation, ML engine, Mac support, pipeline, audio, alerts, backtest...)
```
