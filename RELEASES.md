# NewsTrader releases

Ready-made apps, so you don't need to install Python. Download them from the
**[Releases page](https://github.com/Harryisadag/newstrader/releases)**: open the newest release, then look
under **Assets**.

## Which file do I need?

| Your computer | Download | Needs |
|---|---|---|
| Windows 10 / 11 (64-bit) | `NewsTrader-<version>-windows-x64.zip` | An NVIDIA GPU for fast TV transcription (without one it runs on the CPU, slower) |
| Mac with an Apple M-series chip (M1 or newer) | `NewsTrader-<version>-mac-apple-silicon.zip` | macOS 15 Sequoia or newer |
| Older Mac with an Intel chip | `NewsTrader-<version>-mac-intel.zip` | macOS 15 Sequoia or newer. TV transcription runs on the CPU (slower) |

To check which Mac you have: Apple menu → **About This Mac**. "Chip: Apple M…" means the Apple Silicon zip;
"Processor: … Intel …" means the Intel zip.

On macOS 14 or older, run NewsTrader from the source code instead. See "Setup on a Mac" in the [README](https://github.com/Harryisadag/newstrader/blob/claude/zen-gauss-5ktkek/README.md#setup-on-a-mac-one-time-20-minutes).

## How to install a release

**Windows**
1. Download the Windows zip, right-click it → **Extract All…**, and pick an easy folder such as `C:\NewsTrader`.
2. Open the extracted `NewsTrader` folder and double-click **`NewsTrader.exe`**. Keep the whole folder together, because the exe needs the `_internal` folder next to it.
3. If Windows SmartScreen says it protected your PC, click **More info → Run anyway**. The app isn't code-signed (that costs money), so this happens once for each new version.
   - If it says **Smart App Control** blocked NewsTrader, there's no Run anyway button. Run NewsTrader from the source code instead (README, "Setup on Windows"). Turning Smart App Control off also works, but it's a security setting for the whole PC.
4. Paste your Alpaca paper keys in **Settings → API keys**, then run **Logs → Run diagnostics**.

**Mac**
1. Download the Mac zip for your chip and double-click it. Drag **NewsTrader.app** into **Applications**.
2. Open it. macOS says Apple couldn't check it for malicious software, because the app isn't notarized by Apple (that costs money). Click **Done**, not Move to Trash.
   - If it says instead that NewsTrader "is damaged and can't be opened", click **Cancel** (not Move to Trash). Open Terminal, run
     `xattr -dr com.apple.quarantine /Applications/NewsTrader.app`, open the app again and skip step 3.
3. Open **System Settings → Privacy & Security**, scroll down, and click **Open Anyway** next to "NewsTrader was blocked". Click **Open Anyway** again and enter your password.
   - The button only shows for about an hour after you tried to open the app. If it's missing, open the app again first.
   - You do this once for each new version you download.
4. Paste your Alpaca paper keys in **Settings → API keys**, then run **Logs → Run diagnostics**.

The first start downloads the FinBERT model (~110 MB). As soon as one of the default TV streams is live, it also downloads the speech model:
- Windows with an NVIDIA GPU: about 3 GB
- Mac with an M-series chip: about 1.6 GB
- Windows without NVIDIA, or an Intel Mac: about 0.5 GB (transcription runs on the CPU, slower)

Your settings and keys live outside the app, so you can replace the app with a newer version without losing them:

| | Folder |
|---|---|
| Windows | `%LOCALAPPDATA%\NewsTrader` |
| Mac | `~/Library/Application Support/NewsTrader` |

To open it: on Windows, paste `%LOCALAPPDATA%\NewsTrader` into the File Explorer address bar. On a Mac, Finder → **Go → Go to Folder…** and paste the path.

**Used NewsTrader from the source code before (`run.bat` / `run.command`)?** The app doesn't look in that folder, so it starts fresh. To bring your keys, settings, history and trained price model across, close NewsTrader and copy these into the folder above:
- `.env` from your NewsTrader folder
- `config.json`, `newstrader.db` and the `models` folder from its `data` folder

**Updating:** from v0.3.0 the app checks for a newer release once a day and shows a banner with a link (it never installs anything by itself). Download the newest release and replace the app. Your settings stay.
- Windows: delete the old `NewsTrader` folder, then extract the new one in its place.
- Mac: drag the new NewsTrader.app into Applications and choose **Replace**.

The app has its own copy of yt-dlp (the YouTube downloader). If TV streams stop working after YouTube changes something, the fix is a newer release. `update.bat` / `update.command` are only for the source version.

Every release also lists a SHA-256 checksum for each file, so you can check a download is intact.

## Versions

## v0.3.0 — 8 October 2026

**New: market alerts and the Market tab**
- Sudden spikes: a watched stock moving 3%+ within 5 minutes on unusual volume, with the news behind it. Watched = your positions, stocks with recent signals, your watchlist and today's top movers.
- Big moves in the whole market (S&P 500, Nasdaq 100, Russell 2000, Dow) and world markets (Japan, China, Germany, UK, India, Brazil, Europe, Korea).
- Alerts for both (desktop and Discord), each with its own on/off switch.
- "Don't chase": if a stock already ran more than 3% in the signal's direction since the news, the signal goes to manual review instead of auto-trading.

**New: many more news sources (about 120 ready-made, 25 on by default)**
- More US TV and live-event channels: the White House (briefings, Trump remarks), the Fed's press conferences, C-SPAN, Fox News / Fox Business live events, rallies and interviews. Live-event channels only use a transcription slot while they're live.
- Trump interviews and White House announcements news feeds, plus Reuters, Bloomberg, Fox Business, BBC, FDA and Fed feeds, tariff news and more.
- International TV and news from Europe, the UK, Asia, India, the Middle East, Canada, Latin America and Australia. Non-English TV is translated to English while it's transcribed.
- A new sources screen: search, filters by region and kind, turn whole groups on/off, and edit any source.
- Two default feeds that had stopped working (Yahoo Finance and MarketWatch real-time) were replaced automatically.

**Better news detection (local engine)**
- Recognises about 70 kinds of market-moving news (earnings and the numbers behind them, forecasts, buyouts, analyst actions, FDA and trial results, offerings, lawsuits, recalls…) and works out which company each one is good or bad for (the target vs the buyer, the rated stock vs the analyst firm, the winner vs the company it replaced).
- Recaps of moves that already happened, opinion pieces, round-ups and law-firm adverts come out neutral. Reports and rumours get lower confidence; denials turn the event around.
- Signals scored on wording alone (no recognised event) now always go to manual review.
- International macro news can give a signal for that country's fund (always manual review).
- Better company-name matching (short names like "Sarepta", everyday-word names like "Target" only when used as the company, CEO names, lower-case TV transcripts).
- On 150 labelled headlines written by people who never saw the rules, the right call was 66% at the first look (v0.2: 45%). After fixing the general gaps that test showed it's about 92% on that set, though that's flattering because the set was used for the fixes.
- The Signals screen shows the news event behind each call, and the Test box explains why a company came out neutral.
- If you trained a price model with v0.2, retrain it once (Backtest tab): it now also learns from the news event.

**Other**
- Update check: a banner when a newer version is out (Settings → Display).
- Times follow your choice of this computer's time zone, New York time or UTC.
- Logs → System status hides turned-off and off-air sources unless you ask for them.
- When Alpaca blocks a trade, the message now says which Alpaca setting is causing it and how to fix it.

## v0.2.1 — 7 October 2026

Fixes for the downloadable apps.
- Windows PCs without an NVIDIA graphics card no longer download a 3 GB speech model they can't use. They go straight to the smaller CPU model, and Run diagnostics shows this as a warning instead of an error.
- In the downloaded app, Run diagnostics now says to get the newest release instead of running `update.bat` / `update.command` (those are only for the source version).
- Clearer install steps: what to click on a Mac, when "Open Anyway" appears, Windows Smart App Control, moving over from the source version, and how to update.

## v0.2.0 — 7 October 2026

The first downloadable release.

**New: free machine-learning engine (now the default)**
- **FinBERT**, a language model trained on financial news, reads the sentences about each company and scores them good / bad / neutral. It runs on your own computer, needs no API key, and costs nothing per story.
- Optional **price model** you train yourself on the Backtest tab, from free Alpaca history:
  - It learns how stocks actually moved compared with the S&P 500 in the hour after similar headlines.
  - It's tested on the newest news it never saw, and only switched on if it beats chance.
- Claude is still available as an optional engine (Settings → AI engine).
- If you run from the source code and used the Claude-only version, updating switches you to the local engine and tells you so once.

**New: Mac support**
- Runs on Apple Silicon and Intel Macs.
- On M-series Macs, TV audio is transcribed on the Apple GPU.
- Mac desktop notifications, plus `run.command` / `update.command` / `build_app.command` launchers for running from source.

**Safety**
- Still **paper trading only** by default.
- While live trading is switched on, signals scored on wording alone always need your manual approval.

**Everything from the first build**
- Live news: TV streams with real-time transcription, Alpaca/Benzinga, RSS and Truth Social.
- Bracket orders with risk limits and a kill switch.
- Alerts: desktop and Discord.
- Performance tracking, backtests and a dark dashboard.

## v0.1.0 — not published

The first version, Windows-only with Claude as the only AI engine. It was only ever run from the source code.
