"""Sources that come preloaded on first launch. All of them can be toggled or removed in the app.

Live streams use YouTube channel "/live" URLs, so they keep working when the channel starts a new
broadcast with a new video ID.
"""

from __future__ import annotations

_PRESETS: list[dict] = [
    # ---- Live TV / video streams (audio is transcribed) ----
    {"id": "bloomberg-tv", "type": "stream", "name": "Bloomberg TV",
     "url": "https://www.youtube.com/@markets/live", "enabled": True},
    {"id": "yahoo-finance-live", "type": "stream", "name": "Yahoo Finance Live",
     "url": "https://www.youtube.com/@YahooFinance/live", "enabled": True},
    {"id": "schwab-network", "type": "stream", "name": "Schwab Network",
     "url": "https://www.youtube.com/@SchwabNetwork/live", "enabled": True},
    {"id": "livenow-fox", "type": "stream", "name": "LiveNOW from FOX",
     "url": "https://www.youtube.com/@LiveNOWFOX/live", "enabled": True},
    {"id": "sky-news", "type": "stream", "name": "Sky News",
     "url": "https://www.youtube.com/@SkyNews/live", "enabled": False},
    {"id": "abc-news-live", "type": "stream", "name": "ABC News Live",
     "url": "https://www.youtube.com/@ABCNews/live", "enabled": False},
    {"id": "nbc-news-now", "type": "stream", "name": "NBC News NOW",
     "url": "https://www.youtube.com/@NBCNews/live", "enabled": False},
    {"id": "cbs-news-247", "type": "stream", "name": "CBS News 24/7",
     "url": "https://www.youtube.com/@CBSNews/live", "enabled": False},

    # ---- Alpaca / Benzinga real-time news websocket ----
    {"id": "alpaca-news", "type": "alpaca_news", "name": "Alpaca / Benzinga news",
     "url": "", "enabled": True},

    # ---- RSS feeds ----
    {"id": "cnbc-top", "type": "rss", "name": "CNBC Top News",
     "url": "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=100003114",
     "enabled": True},
    {"id": "cnbc-earnings", "type": "rss", "name": "CNBC Earnings",
     "url": "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=15839135",
     "enabled": True},
    {"id": "marketwatch-top", "type": "rss", "name": "MarketWatch Top Stories",
     "url": "https://feeds.content.dowjones.io/public/rss/mw_topstories", "enabled": True},
    {"id": "marketwatch-realtime", "type": "rss", "name": "MarketWatch Real-time Headlines",
     "url": "https://feeds.content.dowjones.io/public/rss/mw_realtimeheadlines", "enabled": True},
    {"id": "wsj-markets", "type": "rss", "name": "WSJ Markets",
     "url": "https://feeds.content.dowjones.io/public/rss/RSSMarketsMain", "enabled": True},
    {"id": "yahoo-finance-news", "type": "rss", "name": "Yahoo Finance News",
     "url": "https://finance.yahoo.com/news/rssindex", "enabled": True},
    {"id": "seeking-alpha-currents", "type": "rss", "name": "Seeking Alpha Market Currents",
     "url": "https://seekingalpha.com/market_currents.xml", "enabled": True},
    {"id": "investing-stock-news", "type": "rss", "name": "Investing.com Stock Market News",
     "url": "https://www.investing.com/rss/news_25.rss", "enabled": True},
    {"id": "google-news-business", "type": "rss", "name": "Google News - Business",
     "url": "https://news.google.com/rss/headlines/section/topic/BUSINESS?hl=en-US&gl=US&ceid=US:en",
     "enabled": True},
    {"id": "fed-press", "type": "rss", "name": "Federal Reserve press releases",
     "url": "https://www.federalreserve.gov/feeds/press_all.xml", "enabled": True},
    # High volume press-release / filing feeds - off by default (more Claude calls)
    {"id": "prnewswire", "type": "rss", "name": "PR Newswire (all releases)",
     "url": "https://www.prnewswire.com/rss/news-releases-list.rss", "enabled": False},
    {"id": "globenewswire", "type": "rss", "name": "GlobeNewswire (public companies)",
     "url": "https://www.globenewswire.com/RssFeed/orgclass/1/feedTitle/GlobeNewswire%20-%20News%20about%20Public%20Companies",
     "enabled": False},
    {"id": "sec-8k", "type": "rss", "name": "SEC EDGAR 8-K filings",
     "url": "https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&type=8-K&company=&dateb=&owner=include&count=40&output=atom",
     "enabled": False},

    # ---- Social posts ----
    # Truth Social has no public API. trumpstruth.org is a free public archive with an RSS feed.
    {"id": "truth-social-trump", "type": "social_rss", "name": "Donald Trump (Truth Social)",
     "url": "https://www.trumpstruth.org/feed", "enabled": True,
     "speaker": "Donald Trump", "poll_seconds": 30},
]


def default_sources() -> list[dict]:
    """Fresh copies of the preset sources (marked builtin so the UI can label them)."""
    out = []
    for preset in _PRESETS:
        item = dict(preset)
        item["builtin"] = True
        out.append(item)
    return out
