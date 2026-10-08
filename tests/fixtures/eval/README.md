# Detection test sets

Headlines (and a few live-TV transcript lines) with the companies they are about and what a trader would expect each
stock to do in the next hour: `bullish`, `bearish` or `neutral`. `absent` lists companies that must NOT be picked up
(e.g. "S&P 500" in an index-inclusion story is not SPY). Every item was labelled twice, blind, by different
reviewers; disagreements were settled and the reason is kept in `note`.

| File | Items | How it was used |
|---|---|---|
| `dev.jsonl` | 156 | Written first and used to build the event rules. |
| `set2.jsonl` | 280 | Kept unseen until the rules were done, then measured once: 79.9% right with the event rules vs 56.9% for v0.2. After that first look its mistakes were used to fix general gaps, so it is no longer unseen. |
| `set3.jsonl` | 150 | Written by two separate agents who never saw the rules (and labelled again blind by a third: 98.6% agreement, 3 items settled by hand). First look: 65.9% right with the event rules vs 45.3% for v0.2. Then used for a second round of general fixes (91.6% after them). |

`eval_tickers.json` is a fixed slice of Alpaca's asset list so the results don't depend on today's listings.

A fresh, sealed set should be written for the next honest number. If a `holdout.jsonl` is added here, CI measures it too.

Run `python scripts/eval_detection.py --set set2 --compare --mistakes 30` to see the numbers and the misses.
`tests/test_events.py` fails if accuracy on these sets drops, so a rule change can't silently break other cases.
