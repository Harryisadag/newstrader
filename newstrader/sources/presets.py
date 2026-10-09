"""Sources that come preloaded. All of them can be turned on/off or removed in the app.

Live streams use YouTube channel "/live" URLs, so they keep working when the channel starts a new broadcast with a
new video ID. Channels that only go live for events (press conferences, interviews, hearings) are marked
live_events: they are checked every few minutes and only use a transcription slot while they are live.

Only a handful are on by default - the rest are a catalogue to pick from in Settings -> News sources (grouped by
region and kind). Non-English TV is translated to English while it is transcribed. Non-English text feeds can only
be read by the Claude engine (the local engine reads English).

Every URL here is checked by scripts/check_sources.py (run by the "check-sources" GitHub workflow).
"""

from __future__ import annotations

YT = "https://www.youtube.com/"


def _tv(id: str, name: str, path: str, region: str, *, on: bool = False, language: str = "en",
        category: str = "TV", events: bool = False) -> dict:
    out = {"id": id, "type": "stream", "name": name, "url": YT + path.strip("/") + "/live", "enabled": on,
           "region": region, "category": "Live events" if events else category, "language": language}
    if language != "en":
        out["translate"] = True
    if events:
        out["live_events"] = True
    return out


def _rss(id: str, name: str, url: str, region: str, category: str, *, on: bool = False, language: str = "en",
         poll: int | None = None) -> dict:
    out = {"id": id, "type": "rss", "name": name, "url": url, "enabled": on, "region": region, "category": category}
    if language != "en":
        out["language"] = language
    if poll:
        out["poll_seconds"] = poll
    return out


def _gnews(query: str, *, hl: str = "en-US", gl: str = "US") -> str:
    """Google News search feed (unofficial, a few minutes behind - poll it every 5 minutes, not faster)."""
    return f"https://news.google.com/rss/search?q={query}&hl={hl}&gl={gl}&ceid={gl}:{hl.split('-')[0]}"


def _gnews_business(hl: str, gl: str) -> str:
    return f"https://news.google.com/rss/headlines/section/topic/BUSINESS?hl={hl}&gl={gl}&ceid={gl}:{hl.split('-')[0]}"


GN = 300  # Google News: poll every 5 minutes
# Breaking-news wires (press releases, SEC filings, the Fed, Trump's posts) are checked every 15 seconds; every other
# feed every 30 seconds (the SourceConfig default). Feeds are asked "anything new since last time?", so a check
# with no news costs almost nothing.
FAST = 15

_PRESETS: list[dict] = [
    # ================= Live TV / video streams (audio is transcribed) =================
    # ---- US business TV (always on air) ----
    _tv("bloomberg-tv", "Bloomberg TV", "@markets", "US", on=True),
    _tv("yahoo-finance-live", "Yahoo Finance Live", "@YahooFinance", "US", on=True),
    _tv("schwab-network", "Schwab Network", "@SchwabNetwork", "US", on=True),
    _tv("livenow-fox", "LiveNOW from FOX", "@LiveNOWFOX", "US", on=True, category="TV"),
    _tv("benzinga-live", "Benzinga Live (pre-market shows)", "@Benzinga", "US"),
    _tv("newsmax", "Newsmax (free live stream)", "@NewsmaxTV", "US"),
    _tv("scripps-news", "Scripps News", "@scrippsnews", "US"),
    _tv("sky-news", "Sky News", "@SkyNews", "UK"),
    _tv("abc-news-live", "ABC News Live", "@ABCNews", "US"),
    _tv("nbc-news-now", "NBC News NOW", "@NBCNews", "US"),
    _tv("cbs-news-247", "CBS News 24/7", "@CBSNews", "US"),
    # ---- US live events: Trump / White House, Fed, Congress (only live during events) ----
    _tv("white-house", "The White House (press briefings, Trump remarks)", "@WhiteHouse", "US", on=True, events=True),
    _tv("fox-news-live", "Fox News (live events and interviews)", "@FoxNews", "US", events=True),
    _tv("fox-business-live", "Fox Business (live events)", "@FoxBusiness", "US", events=True),
    _tv("rsbn", "Right Side Broadcasting (Trump rallies and speeches)", "@RSBNetwork", "US", events=True),
    _tv("forbes-breaking-news", "Forbes Breaking News (hearings, White House)", "@ForbesBreakingNews", "US",
        events=True),
    _tv("pbs-newshour", "PBS NewsHour (live events)", "@PBSNewsHour", "US", events=True),
    _tv("cspan", "C-SPAN", "@cspan", "US", events=True),
    _tv("federal-reserve", "Federal Reserve (FOMC press conferences)", "@federalreserve", "US", on=True, events=True),
    _tv("reuters-live", "Reuters (live press conferences)", "@Reuters", "Global", events=True),
    # ---- International, in English ----
    _tv("dw-news", "DW News (Germany)", "@dwnews", "Europe"),
    _tv("france24-en", "France 24 English", "@France24_en", "Europe"),
    _tv("euronews-en", "Euronews English", "@euronews", "Europe"),
    _tv("aljazeera-en", "Al Jazeera English", "@aljazeeraenglish", "Middle East"),
    _tv("cna", "CNA (Singapore)", "@channelnewsasia", "Asia"),
    _tv("nhk-world", "NHK World-Japan", "@NHKWORLDJAPAN", "Asia"),
    _tv("wion", "WION (India)", "@WION", "India"),
    _tv("cnbc-tv18", "CNBC-TV18 (Indian markets)", "@CNBC-TV18", "India"),
    _tv("abc-australia", "ABC News (Australia)", "@abcnewsaustralia", "Australia"),
    _tv("cbc-news", "CBC News (Canada)", "@CBCNews", "Canada"),
    _tv("trt-world", "TRT World (Turkish state broadcaster)", "@trtworld", "Middle East"),
    _tv("cgtn", "CGTN (Chinese state media)", "@CGTN", "Asia"),
    # ---- International, other languages (translated to English while transcribing) ----
    _tv("dw-deutsch", "DW Deutsch", "c/dwdeutsch", "Europe", language="de"),
    _tv("france24-fr", "France 24 (French)", "c/FRANCE24", "Europe", language="fr"),
    _tv("france24-es", "France 24 Español", "@France24_es", "Latin America", language="es"),
    _tv("rtve-24h", "RTVE Canal 24 Horas (Spain)", "channel/UC7QZIf0dta-XPXsp9Hv4dTw", "Europe", language="es"),
    _tv("milenio", "Milenio (Mexico)", "user/milenio", "Latin America", language="es"),
    _tv("cnn-brasil", "CNN Brasil", "c/CNNbrasil", "Latin America", language="pt"),
    _tv("bloomberg-ht", "Bloomberg HT (Turkey)", "@BloombergHT", "Middle East", language="tr"),
    _tv("asharq-business", "Asharq Business with Bloomberg (Arabic)", "@AsharqBusiness", "Middle East", language="ar"),
    _tv("zee-business", "Zee Business (Hindi)", "@ZeeBusiness", "India", language="hi"),
    _tv("ann-news", "ANN News / TV Asahi (Japan)", "@ANNnewsCH", "Asia", language="ja"),
    _tv("ytn", "YTN (South Korea)", "@ytnnews24", "Asia", language="ko"),
    _tv("korea-economic-tv", "Korea Economic TV", "channel/UCF8AeLlUbEpKju6v1H6p8Eg", "Asia", language="ko"),

    # ================= Alpaca / Benzinga real-time news websocket =================
    {"id": "alpaca-news", "type": "alpaca_news", "name": "Alpaca / Benzinga news", "url": "", "enabled": True,
     "region": "US", "category": "Business news"},

    # ================= RSS feeds =================
    # ---- US business news ----
    _rss("cnbc-top", "CNBC Top News",
         "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=100003114", "US",
         "Business news", on=True),
    _rss("cnbc-earnings", "CNBC Earnings",
         "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=15839135", "US",
         "Business news", on=True),
    _rss("marketwatch-top", "MarketWatch Top Stories",
         "https://feeds.content.dowjones.io/public/rss/mw_topstories", "US", "Business news", on=True),
    _rss("marketwatch-realtime", "MarketWatch (via Google News)", _gnews("when:24h+site:marketwatch.com"), "US",
         "Business news", on=True, poll=GN),
    _rss("wsj-markets", "WSJ Markets", "https://feeds.content.dowjones.io/public/rss/RSSMarketsMain", "US",
         "Business news", on=True),
    _rss("yahoo-finance-news", "Yahoo Finance News (via Google News)", _gnews("when:24h+site:finance.yahoo.com"),
         "US", "Business news", on=True, poll=GN),
    _rss("seeking-alpha-currents", "Seeking Alpha Market Currents", "https://seekingalpha.com/market_currents.xml",
         "US", "Business news", on=True),
    _rss("investing-stock-news", "Investing.com Stock Market News", "https://www.investing.com/rss/news_25.rss",
         "US", "Business news", on=True),
    _rss("google-news-business", "Google News - Business",
         "https://news.google.com/rss/headlines/section/topic/BUSINESS?hl=en-US&gl=US&ceid=US:en", "US",
         "Business news", on=True, poll=GN),
    _rss("bloomberg-markets", "Bloomberg Markets", "https://feeds.bloomberg.com/markets/news.rss", "US",
         "Business news", on=True),
    _rss("bloomberg-economics", "Bloomberg Economics", "https://feeds.bloomberg.com/economics/news.rss", "Global",
         "Business news"),
    _rss("bloomberg-technology", "Bloomberg Technology", "https://feeds.bloomberg.com/technology/news.rss", "US",
         "Business news"),
    _rss("bloomberg-politics", "Bloomberg Politics", "https://feeds.bloomberg.com/politics/news.rss", "US",
         "Politics & Trump"),
    _rss("reuters-business", "Reuters Business (via Google News)", _gnews("when:24h+site:reuters.com/business"),
         "Global", "Business news", on=True, poll=GN),
    _rss("reuters-markets", "Reuters Markets (via Google News)", _gnews("when:24h+site:reuters.com/markets"),
         "Global", "Business news", poll=GN),
    _rss("reuters-tech", "Reuters Technology (via Google News)", _gnews("when:24h+site:reuters.com/technology"),
         "Global", "Business news", poll=GN),
    _rss("ap-news", "AP News (via Google News)", _gnews("when:24h+site:apnews.com"), "US", "Business news",
         poll=GN),
    _rss("fox-business-latest", "Fox Business - Latest", "https://moxie.foxbusiness.com/google-publisher/latest.xml",
         "US", "Business news", on=True),
    _rss("fox-business-markets", "Fox Business - Markets",
         "https://moxie.foxbusiness.com/google-publisher/markets.xml", "US", "Business news"),
    _rss("fox-business-economy", "Fox Business - Economy",
         "https://moxie.foxbusiness.com/google-publisher/economy.xml", "US", "Business news"),
    _rss("cnbc-world", "CNBC World", "https://www.cnbc.com/id/100727362/device/rss/rss.html", "Global",
         "Business news"),
    _rss("cnbc-tech", "CNBC Technology", "https://www.cnbc.com/id/19854910/device/rss/rss.html", "US",
         "Business news"),
    _rss("cnbc-finance", "CNBC Finance", "https://www.cnbc.com/id/10000664/device/rss/rss.html", "US",
         "Business news"),
    _rss("cnbc-economy", "CNBC Economy", "https://www.cnbc.com/id/20910258/device/rss/rss.html", "US",
         "Business news"),
    _rss("wsj-business", "WSJ US Business", "https://feeds.content.dowjones.io/public/rss/WSJcomUSBusiness", "US",
         "Business news"),
    _rss("wsj-tech", "WSJ Technology", "https://feeds.content.dowjones.io/public/rss/RSSWSJD", "US",
         "Business news"),
    _rss("business-insider", "Business Insider", "https://feeds.businessinsider.com/custom/all", "US",
         "Business news"),
    _rss("fortune", "Fortune", "https://fortune.com/feed/fortune-feeds/?id=3230629", "US", "Business news"),
    _rss("techcrunch", "TechCrunch", "https://techcrunch.com/feed/", "US", "Business news"),
    _rss("nyt-business", "New York Times - Business", "https://rss.nytimes.com/services/xml/rss/nyt/Business.xml",
         "US", "Business news"),
    _rss("cbs-moneywatch", "CBS MoneyWatch", "https://www.cbsnews.com/latest/rss/moneywatch", "US",
         "Business news"),
    _rss("investing-earnings", "Investing.com Earnings", "https://www.investing.com/rss/news_1062.rss", "US",
         "Business news"),
    _rss("investing-economy", "Investing.com Economy", "https://www.investing.com/rss/news_14.rss", "Global",
         "Business news"),
    # ---- Politics: Trump interviews, tariffs, the White House ----
    _rss("trump-interviews", "Trump interviews (Google News)", _gnews("Trump+interview+when:1d"), "US",
         "Politics & Trump", on=True, poll=GN),
    _rss("tariff-news", "Tariffs and trade deals (Google News)",
         _gnews("tariffs+OR+%22trade+deal%22+OR+%22trade+war%22+when:1d"), "Global", "Politics & Trump", on=True,
         poll=GN),
    _rss("white-house-news", "White House announcements (Google News)",
         _gnews("%22White+House%22+announces+OR+%22executive+order%22+when:1d"), "US", "Politics & Trump",
         poll=GN),
    # ---- US regulators and data ----
    _rss("fed-press", "Federal Reserve press releases", "https://www.federalreserve.gov/feeds/press_all.xml", "US",
         "Central banks", on=True, poll=FAST),
    _rss("fed-speeches", "Federal Reserve speeches", "https://www.federalreserve.gov/feeds/speeches.xml", "US",
         "Central banks"),
    _rss("fda-press", "FDA press releases (drug approvals)",
         "https://www.fda.gov/about-fda/contact-fda/stay-informed/rss-feeds/press-releases/rss.xml", "US",
         "Regulators", on=True),
    _rss("fda-recalls", "FDA recalls", "https://www.fda.gov/about-fda/contact-fda/stay-informed/rss-feeds/recalls/rss.xml",
         "US", "Regulators"),
    _rss("sec-press", "SEC press releases", "https://www.sec.gov/news/pressreleases.rss", "US", "Regulators"),
    _rss("bls-latest", "BLS economic data (jobs, inflation)", "https://www.bls.gov/feed/bls_latest.rss", "US",
         "Regulators"),
    _rss("eia-press", "EIA energy data", "https://www.eia.gov/rss/press_rss.xml", "US", "Regulators"),
    # High-volume press-release / filing feeds - off by default
    _rss("prnewswire", "PR Newswire (all releases)", "https://www.prnewswire.com/rss/news-releases-list.rss", "US",
         "Press releases", poll=FAST),
    _rss("globenewswire", "GlobeNewswire (public companies)",
         "https://www.globenewswire.com/RssFeed/orgclass/1/feedTitle/GlobeNewswire%20-%20News%20about%20Public%20Companies",
         "US", "Press releases", poll=FAST),
    _rss("sec-8k", "SEC EDGAR 8-K filings",
         "https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&type=8-K&company=&dateb=&owner=include&count=40&output=atom",
         "US", "Press releases", poll=FAST),  # SEC allows 10 requests a second; never poll it faster than 15 s
    # ---- Central banks around the world ----
    _rss("ecb-press", "European Central Bank", "https://www.ecb.europa.eu/rss/press.html", "Europe",
         "Central banks"),
    _rss("boe-news", "Bank of England", "https://www.bankofengland.co.uk/rss/news", "UK", "Central banks"),
    _rss("bank-of-canada", "Bank of Canada", "https://www.bankofcanada.ca/utility/news/feed/", "Canada",
         "Central banks"),
    # ---- UK and Europe ----
    _rss("bbc-business", "BBC Business", "https://feeds.bbci.co.uk/news/business/rss.xml", "UK", "Business news",
         on=True),
    _rss("ft-home", "Financial Times (headlines)", "https://www.ft.com/rss/home/international", "UK",
         "Business news"),
    _rss("ft-markets", "Financial Times - Markets", "https://www.ft.com/markets?format=rss", "UK", "Business news"),
    _rss("economist-finance", "The Economist - Finance", "https://www.economist.com/finance-and-economics/rss.xml",
         "UK", "Business news"),
    _rss("guardian-business", "The Guardian - Business", "https://www.theguardian.com/uk/business/rss", "UK",
         "Business news"),
    _rss("dw-business", "DW Business", "https://rss.dw.com/rdf/rss-en-bus", "Europe", "Business news"),
    _rss("euronews-business", "Euronews Business", "https://www.euronews.com/rss?level=vertical&name=business",
         "Europe", "Business news"),
    _rss("france24-news", "France 24 (English)", "https://www.france24.com/en/rss", "Europe", "Business news"),
    _rss("google-news-uk", "Google News - UK Business", _gnews_business("en-GB", "GB"), "UK", "Business news",
         poll=GN),
    # ---- Asia-Pacific ----
    _rss("nikkei-asia", "Nikkei Asia (headlines)", "https://asia.nikkei.com/rss/feed/nar", "Asia", "Business news"),
    _rss("scmp-business", "South China Morning Post - Business", "https://www.scmp.com/rss/92/feed", "Asia",
         "Business news"),
    _rss("japan-times", "The Japan Times", "https://www.japantimes.co.jp/feed/", "Asia", "Business news"),
    _rss("kyodo", "Kyodo News (English)", "https://english.kyodonews.net/rss/kyodonews-fzone.xml", "Asia",
         "Business news"),
    _rss("yonhap-en", "Yonhap (Korea, English)", "https://en.yna.co.kr/RSS/news.xml", "Asia", "Business news"),
    _rss("korea-herald", "The Korea Herald", "https://www.koreaherald.com/rss/newsAll", "Asia", "Business news"),
    _rss("straits-times-asia", "The Straits Times - Asia", "https://www.straitstimes.com/news/asia/rss.xml", "Asia",
         "Business news"),
    _rss("google-news-singapore", "Google News - Singapore Business", _gnews_business("en-SG", "SG"), "Asia",
         "Business news", poll=GN),
    _rss("economic-times", "The Economic Times (India)", "https://economictimes.indiatimes.com/rssfeedsdefault.cms",
         "India", "Business news"),
    _rss("business-standard", "Business Standard (India)", "https://www.business-standard.com/rss/latest.rss",
         "India", "Business news"),
    _rss("livemint", "Mint (India)", "https://www.livemint.com/rss/news", "India", "Business news"),
    _rss("google-news-india", "Google News - India Business", _gnews_business("en-IN", "IN"), "India",
         "Business news", poll=GN),
    _rss("smh-business", "Sydney Morning Herald - Business", "https://www.smh.com.au/rss/business.xml", "Australia",
         "Business news"),
    _rss("google-news-australia", "Google News - Australia Business", _gnews_business("en-AU", "AU"), "Australia",
         "Business news", poll=GN),
    # ---- Americas outside the US, Middle East ----
    _rss("financial-post", "Financial Post (Canada)", "https://financialpost.com/category/news/feed.xml", "Canada",
         "Business news"),
    _rss("google-news-canada", "Google News - Canada Business", _gnews_business("en-CA", "CA"), "Canada",
         "Business news", poll=GN),
    _rss("aljazeera-news", "Al Jazeera (English)", "https://www.aljazeera.com/xml/rss/all.xml", "Middle East",
         "Business news"),
    # ---- Other languages (only the Claude engine can read these) ----
    _rss("les-echos", "Les Echos - Marchés (French)",
         "https://services.lesechos.fr/rss/les-echos-finance-marches.xml", "Europe", "Business news", language="fr"),
    _rss("expansion-es", "Expansión (Spain)", "https://e00-expansion.uecdn.es/rss/economia.xml", "Europe",
         "Business news", language="es"),
    _rss("el-financiero", "El Financiero (Mexico)", "https://www.elfinanciero.com.mx/rss", "Latin America",
         "Business news", language="es"),

    # ================= Social posts =================
    # Truth Social has no public API. trumpstruth.org is a free public archive with an RSS feed.
    {"id": "truth-social-trump", "type": "social_rss", "name": "Donald Trump (Truth Social)",
     "url": "https://www.trumpstruth.org/feed", "enabled": True, "speaker": "Donald Trump", "poll_seconds": FAST,
     "region": "US", "category": "Politics & Trump"},
]

# Built-in sources whose feed moved or died: users still on the old URL are moved to the preset's current URL (and
# its name / check interval); a URL the user edited themselves is never touched. id -> old URL.
PRESET_URL_FIXES: dict[str, str] = {
    "yahoo-finance-news": "https://finance.yahoo.com/news/rssindex",  # 404 since 2026
    "marketwatch-realtime": "https://feeds.content.dowjones.io/public/rss/mw_realtimeheadlines",  # stopped in 2025
}


# Check intervals before v0.4: every feed was checked each 60 s, except these (Google News feeds kept 300 s).
_OLD_POLL = {"truth-social-trump": 30}


def old_poll_seconds(preset: dict) -> int:
    """A preset's check interval before v0.4 (saved configs still on it move to the new one)."""
    if preset["id"] in _OLD_POLL:
        return _OLD_POLL[preset["id"]]
    return GN if preset.get("poll_seconds") == GN else 60


def default_sources() -> list[dict]:
    """Fresh copies of the preset sources (marked builtin so the UI can label them)."""
    out = []
    for preset in _PRESETS:
        item = dict(preset)
        item["builtin"] = True
        out.append(item)
    return out
