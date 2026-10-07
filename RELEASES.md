# NewsTrader releases

Ready-made apps, so you don't need to install Python. Download them from the
**[Releases page](https://github.com/Harryisadag/newstrader/releases)**: open the newest release, then look
under **Assets**.

> The repository is private, so the Releases page only works while you're signed in to GitHub with access to it.

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

**Updating:** download the newest release and replace the app. Your settings stay.
- Windows: delete the old `NewsTrader` folder, then extract the new one in its place.
- Mac: drag the new NewsTrader.app into Applications and choose **Replace**.

The app has its own copy of yt-dlp (the YouTube downloader). If TV streams stop working after YouTube changes something, the fix is a newer release. `update.bat` / `update.command` are only for the source version.

Every release also lists a SHA-256 checksum for each file, so you can check a download is intact.

## Versions

## v0.3.0 — in progress

Market alerts and sudden-spike detection, more news networks (including international ones), and better
good/bad news detection. Full notes are written when it is released.

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
