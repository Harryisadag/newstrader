# NewsTrader

NewsTrader is a Windows desktop app that:

1. **Watches live news** — TV livestreams (Bloomberg, Yahoo Finance, …), Alpaca/Benzinga news, RSS feeds and social posts.
2. **Transcribes TV audio in real time** on your NVIDIA GPU (faster-whisper).
3. **Asks Claude** whether each story is bullish or bearish for a specific stock, with a confidence score.
4. **Places PAPER trades on Alpaca** when confidence is high enough, with stop-loss / take-profit on every buy.

> **Safety first:** NewsTrader always starts in **PAPER** mode (fake money). Live trading is locked in the code. It only unlocks if you add separate live keys, flip the setting, **and** type a confirmation phrase. It goes back to paper every time the app restarts.

---

## Setup (about 20 minutes, one time)

You don't need to be a developer. Follow the steps in order.

### Step 1 — Install Python 3.12

1. Go to https://www.python.org/downloads/ and download **Python 3.12.x (Windows installer, 64-bit)**.
2. Run the installer. **On the first screen, tick the box "Add python.exe to PATH"**. Then click **Install Now**.
3. When it finishes, click **Close**.

### Step 2 — Update your NVIDIA driver (for the GPU)

Your RTX 5070 Ti needs a recent driver (2025 or newer).

1. Open the **NVIDIA App** (or download it from https://www.nvidia.com/en-us/software/nvidia-app/).
2. Go to **Drivers** and install the latest **Game Ready** or **Studio** driver.
3. Restart your PC if it asks.

You do **not** need to install the CUDA Toolkit. NewsTrader installs the CUDA libraries it needs automatically.

You also don't need to install **ffmpeg**. A copy is included automatically. If you already have ffmpeg installed, NewsTrader uses yours.

### Step 3 — Get your API keys

You'll paste these into the app in Step 5. Keep them private.

**Alpaca (paper trading)**
1. Sign up / log in at https://app.alpaca.markets
2. Top-left, make sure you're in your **Paper** account (not Live).
3. On the home page, find **API Keys** on the right side → **Generate New Keys**.
4. Copy both the **Key** and the **Secret**. The secret is only shown once.

**Anthropic (Claude)**
1. Go to https://platform.claude.com and sign in.
2. Add a little credit under **Billing** ($10 is plenty to start).
3. Recommended: under **Limits**, set a monthly spend limit.
4. Go to **API Keys** → **Create Key** → copy it (starts with `sk-ant-`).

**Discord webhook (optional, for phone alerts)**
1. In Discord, open your server → **Server Settings** → **Integrations** → **Webhooks** → **New Webhook**.
2. Pick the channel for alerts, then click **Copy Webhook URL**.

### Step 4 — Download NewsTrader

- **Easiest:** on the GitHub page, click the green **Code** button → **Download ZIP**. Unzip it somewhere easy, like `C:\NewsTrader`.
  - Don't run it from inside the ZIP, and avoid `C:\Program Files`.
- Or, if you use **GitHub Desktop**: File → Clone repository → `Harryisadag/newstrader`.

### Step 5 — Start it

1. Open the NewsTrader folder and **double-click `run.bat`**.
2. The first time, it sets everything up. This takes 5–10 minutes and downloads about 1 GB.
   - If Windows SmartScreen warns you, click **More info** → **Run anyway**.
3. The NewsTrader window opens. Go to **Settings → API keys**, paste your keys and click **Save keys**.
4. Go to **Logs → Run diagnostics**. Everything should be green or yellow. Each red line tells you how to fix it.

From now on, just double-click `run.bat` to start. The black console window shows the engine log. Closing it closes the app.

The first time a TV stream starts, the speech model (Whisper large-v3, about 3 GB) downloads once.

---

## Where your stuff is stored

| What | Where (dev mode with run.bat) |
|---|---|
| API keys | `.env` in the NewsTrader folder |
| Settings | `data\config.json` |
| Database (signals, trades, transcripts) | `data\newstrader.db` |
| Log files | `data\logs\` |
| CSV exports | `data\exports\` |
| Whisper models | `data\models\` |

The `.exe` version keeps the same files in `%LOCALAPPDATA%\NewsTrader\` instead.

---

## Social media sources (Truth Social and X)

- **Truth Social:** there's no official API. NewsTrader comes with Donald Trump's posts preloaded through the free public archive feed at trumpstruth.org. Free feeds can lag a little behind the real posts.
- **X / Twitter:** there's currently no free, legal way to read posts automatically. Your options:
  1. **Buy X API access** (Basic tier, roughly $200/month). Paste the bearer token in Settings → API keys. Then add an "X account" source with the username.
  2. **Use an RSS service** like rss.app (paid) that turns an X account into a feed. Add it as a **Social** source.
  3. Self-hosted scrapers (Nitter/RSSHub) break often and are against X's terms, so they aren't included.

---

## Developer notes

- Run tests: `.venv\Scripts\python -m pytest`
- Run in your normal browser instead of the app window: `run.bat --browser`
- Server only: `run.bat --headless`
