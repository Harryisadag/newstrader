# NewsTrader releases

Ready-made apps, so you don't need to install Python. Download them from the
**[Releases page](https://github.com/Harryisadag/newstrader/releases)**: open the newest release, then look
under **Assets**.

> The repository is private, so the Releases page only works while you're signed in to GitHub with access to it.

## Which file do I need?

| Your computer | Download | Needs |
|---|---|---|
| Windows 10 / 11 (64-bit) | `NewsTrader-<version>-windows-x64.zip` | An NVIDIA GPU for fast transcription (it still runs without one, just slower) |
| Mac with an Apple M-series chip (M1 or newer) | `NewsTrader-<version>-mac-apple-silicon.zip` | macOS 15 Sequoia or newer |
| Older Mac with an Intel chip | `NewsTrader-<version>-mac-intel.zip` | macOS 15 Sequoia or newer |

To check which Mac you have: Apple menu → **About This Mac** → "Chip".

On macOS 14 or older, run NewsTrader from the source code instead. See "Setup on a Mac" in the [README](https://github.com/Harryisadag/newstrader/blob/claude/zen-gauss-5ktkek/README.md#setup-on-a-mac-one-time-20-minutes).

## How to install a release

**Windows**
1. Download the Windows zip, right-click it → **Extract All…**, and pick an easy folder such as `C:\NewsTrader`.
2. Open the extracted `NewsTrader` folder and double-click **`NewsTrader.exe`**. Keep the whole folder together, because the exe needs the `_internal` folder next to it.
3. If Windows SmartScreen says it protected your PC, click **More info → Run anyway**. The app isn't code-signed, which costs money.
4. Paste your Alpaca paper keys in **Settings → API keys**, then run **Logs → Run diagnostics**.

**Mac**
1. Download the Mac zip for your chip and double-click it. Drag **NewsTrader.app** into **Applications**.
2. Open it. macOS will say it can't check the app for malicious software, because it isn't from the App Store. Click **Done**.
3. Open **System Settings → Privacy & Security**, scroll down, and click **Open Anyway** next to NewsTrader. Enter your password. This only happens the first time.
   - If it says the app "is damaged", open Terminal and run:
     `xattr -dr com.apple.quarantine /Applications/NewsTrader.app`
4. Paste your Alpaca paper keys in **Settings → API keys**, then run **Logs → Run diagnostics**.

The first start downloads the FinBERT model (~110 MB). The first TV stream downloads the speech model:
- Windows: about 3 GB
- Mac: about 1.6 GB

Your settings and keys live outside the app, so you can replace the app with a newer version without losing them:

| | Folder |
|---|---|
| Windows | `%LOCALAPPDATA%\NewsTrader` |
| Mac | `~/Library/Application Support/NewsTrader` |

Every release also lists a SHA-256 checksum for each file, so you can check a download is intact.

## Versions

## v0.2.0 — 7 October 2026

The first downloadable release.

**New: free machine-learning engine (now the default)**
- **FinBERT**, a language model trained on financial news, reads the sentences about each company and scores them good / bad / neutral. It runs on your own computer, needs no API key, and costs nothing per story.
- Optional **price model** you train yourself on the Backtest tab, from free Alpaca history:
  - It learns how stocks actually moved compared with the S&P 500 in the hour after similar headlines.
  - It's tested on the newest news it never saw, and only switched on if it beats chance.
- Claude is still available as an optional engine (Settings → AI engine).
- If you used the Claude-only version, this update switches you to the local engine and tells you so once.

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
