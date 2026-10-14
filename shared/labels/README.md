# Shared labels

This folder holds **labels for old news**, made by NewsTrader users who let **Pro AI** (the optional AI that runs on
a strong graphics card) read old headlines. NewsTrader uses them to train its small, fast **student model**, so
everyone's app reads headlines a bit more like Pro AI does, without a graphics card and without any AI service at
trade time. See "Shared training" in the [main README](../../README.md#shared-training).

## What's in a file

Each file is named `newstrader-labels-<date>-<code>.jsonl.gz`. `<code>` is a random code NewsTrader made once on the
PC it came from. It only lets the app count how many different PCs contributed and isn't linked to anyone.

A file is gzip-compressed text with one label per line, and nothing else:

```json
{"v":1,"id":24843171,"h":"3f9a1c0b7d2e4f61","t":"NKE","d":"bearish","c":82,"e":"earnings_miss","m":"<Pro AI model key>","day":"2026-06-26"}
```

| Key | What it is |
|---|---|
| `v` | Format version, always `1` |
| `id` | The Alpaca (Benzinga) news id of the story |
| `h` | The first 16 hex characters of the SHA-256 of the headline (leading and trailing spaces removed). It checks that everyone's copy of the story has the same headline. |
| `t` | The stock's ticker, e.g. `AAPL` or `BRK.B` |
| `d` | What Pro AI expects for that stock: `bullish`, `bearish` or `neutral` |
| `c` | How sure Pro AI was, 0-100 (0 = it didn't name the stock at all, which counts as neutral) |
| `e` | The kind of news event NewsTrader's own rules recognised for the stock (e.g. `earnings_miss`), or `""` |
| `m` | Which Pro AI model made the label (a key from `newstrader/llm/catalog.py`) |
| `day` | The day the news came out (New York date) |

**No headline text, no API keys, no trades and nothing about you** are in a file. Everyone downloads the same
Benzinga headlines from Alpaca themselves, and the app matches a label to them by the news id (with `h` as a check).

## How to contribute

You need Pro AI set up (Settings → Pro AI) and the price model trained once (Backtest → Local ML model), which
downloads the old headlines.

1. Settings → Pro AI → **Label old news** → **Start labelling**. It runs in the background, one story at a time, and pauses
   while the US stock market is open.
2. Backtest → Local ML model → **Share my labels**. The file is saved in your Downloads folder, the folder opens, and
   this folder's upload page opens on GitHub.
3. On that page (sign in, or make a free GitHub account): drag the file in, click **Propose changes** (or **Commit
   changes**), then **Create pull request**.

## What happens to an upload

Every upload is a pull request. An automatic check (`scripts/check_shared_labels.py`, run by
`.github/workflows/check-shared-labels.yml`) reads every file and rejects one that:

- isn't named like above, isn't gzip, is bigger than 20 MB or has more than 300,000 labels
- has a line with missing or extra keys, a wrong type, a direction other than the three above, a confidence outside
  0-100, a badly formed ticker, an unknown event kind or Pro AI model, or a day in the future or more than 3 years ago
- lists the same story and ticker twice
- has more than 500 labels of which over 90% point the same way (that looks broken or made up)

The check only reads the files; it never runs anything from an upload, and it fails if a pull request changes
anything outside this folder. It always runs the repository's own copy of the check (from the branch the upload goes
into), so a pull request can't change or skip the check it is judged by.

**The owner reviews every upload before merging it.** Merge only pull requests that add files to this folder and pass
the check. A newer file with the same `<code>` contains that PC's older labels again, so older files from the same
code can be deleted. Removing a file here removes it from everyone's app at their next daily check.
