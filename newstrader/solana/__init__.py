"""The Solana memecoin desk: PAPER ONLY.

It watches brand-new Solana memecoin pools, runs every coin that looks like it's taking off through a strict 7-gate
pre-flight checklist, and makes pretend buys and sales with a pretend wallet of SOL. It never holds a wallet or any
keys, and never builds or sends a transaction: it only reads public data (GeckoTerminal, the Solana network,
Jupiter's price quotes and RugCheck). Real memecoins are mostly rugs, and paper fills are kinder than real ones.

- limiter.py   the rate limits every data source shares
- parsers.py   what each source's answer means (pure functions)
- data.py      the read-only data sources
- gates.py     the seven gates (pure functions over the fetched evidence)
- feed.py      which new pools are worth a look (the entry trigger)
- checklist.py fetches the evidence and runs the gates in order
- desk.py      the paper wallet, paper fills and exits
- audit.py     the reject audit: what happened to each coin 1 hour and 24 hours later
- service.py   the background service that ties it together
"""
