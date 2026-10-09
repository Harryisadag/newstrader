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

**Updating:** the app checks for a newer release once a day and shows a banner at the top (turn it off in Settings → Display).
- **From v0.4:** click **Update now** in the banner. NewsTrader downloads the zip for your computer, checks its SHA-256 checksum, swaps in the new version and restarts by itself (about a minute). Nothing changes until you click it, it won't start while an order is being placed, and if a step fails it puts the old version back. Your settings, keys, history and the kill switch stay.
- **Coming from v0.3** (its banner only has a Download link): update by hand this once. Download the newest release and replace the app:
  - Windows: close NewsTrader, delete the old `NewsTrader` folder, then extract the new one in its place.
  - Mac: close NewsTrader, drag the new NewsTrader.app into Applications and choose **Replace**.
- Update now can't replace an app in a folder your account can't change (such as `C:\Program Files`, or a Mac app that isn't in Applications), or a Windows `NewsTrader` folder that holds other files too. It then saves the checked zip in your Downloads folder and shows it; replace the app by hand as above. Keep the Mac app in Applications so Update now works.

The app has its own copy of yt-dlp (the YouTube downloader). If TV streams stop working after YouTube changes something, the fix is a newer release. `update.bat` / `update.command` are only for the source version.

Every release also lists a SHA-256 checksum for each file, so you can check a download is intact.

## Versions

## v0.4.0 — 9 October 2026

**New: Pro AI for strong PCs (optional, free, offline)**
- A real AI model that runs on your own graphics card and reads the news like a person: who's speaking, denials, who wins a deal, TV speech. No cloud and no cost per story.
- Settings → Pro AI → **Set up Pro AI** checks your graphics card and picks a model that fits next to TV transcription. For example, an RTX 5070 Ti (16 GB) gets Qwen3.5 9B, a one-time download of about 5.7 GB plus about 0.6 GB for the server program. **Test my PC** shows seconds per story, whether the whole model fits on the card, and how many of 20 sample headlines it got right.
- It starts **watch-only**: it judges the hard stories the rules struggle with (TV, news with no recognised event, market-wide news, non-English news) and logs what it would have done, but never trades. Performance → By AI engine compares it with the built-in AI. **Judge** mode lets it send disputed signals to manual review; it still never trades by itself.
- Windows needs an NVIDIA card (driver 551 or newer; 580+ for the faster build) or any card through Vulkan; Macs need an M-series chip. Intel Macs aren't supported.

**New: charts, patterns and indicators**
- The app reads each stock's chart: RSI, MACD, moving averages, VWAP, Bollinger bands, ATR, relative volume, support and resistance, breakouts, gaps, double tops and bottoms, flags and candlestick patterns.
- **Chart check on news signals** (on by default): if the stock already ran ("stretched": RSI over 80 or far above VWAP), the signal goes to manual review instead of buying the top. A chart that disagrees lowers the confidence; one that agrees raises it a little, but never enough to turn a review into an automatic trade. Sells are never held back.
- **Chart signals:** breakouts and similar setups with heavy volume show up on the Signals tab, **watch-only** to start, so the scoreboard can show whether they actually work.
- Market tab: click a stock to see its chart with VWAP and support/resistance, the indicators and the patterns in plain words.

**Faster**
- Every trade shows how long it took from the news coming out to the order, and how late the website itself was. Performance has a Speed card.
- News sites are checked every 30 seconds (was 60). Press-release wires, SEC filings, the Fed and Truth Social are checked every 15 seconds.
- TV: the AI waits 4 seconds after a company is mentioned (was 10). Also fixed: a company mentioned while the previous clip was still being judged was skipped.

**An honest scoreboard**
- Performance shows results by AI engine (built-in, Pro AI, chart patterns) and by news type.
- A new sealed test of 195 fresh headlines that nobody tunes the rules on. The built-in AI gets about **65%** right there; that's the honest number to beat.

**One-click updates**
- From this version, the update banner has an **Update now** button: the app downloads the new version, checks its checksum, swaps itself and restarts. Your settings, keys and history stay. (Coming from v0.3, update by hand this once.)

**Coming next:** crypto (24/7 paper trading), shared training (people with strong PCs label news, everyone's AI learns from it), the built-in simulator screens, Cautious / Normal / Aggressive presets and a "why did it trade?" card.

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
