"""Word lists for the stage-1 pre-filter."""

from __future__ import annotations

# Phrases that usually mean "this could move a stock or the market".
MARKET_KEYWORDS: list[str] = [
    # company events
    "earnings", "quarterly results", "revenue", "guidance", "outlook", "forecast", "profit warning",
    "beats estimates", "misses estimates", "eps", "buyback", "share repurchase", "dividend", "stock split",
    "merger", "acquisition", "acquire", "acquires", "to buy", "takeover", "buyout", "tender offer", "deal talks",
    "spin off", "spinoff", "ipo", "files for", "bankruptcy", "chapter 11", "default", "restructuring", "layoffs",
    "job cuts", "ceo resigns", "ceo steps down", "names new ceo", "fda", "approval", "approves", "clinical trial",
    "phase 3", "recall", "lawsuit", "settlement", "verdict", "antitrust", "investigation", "subpoena", "probe",
    "sec charges", "fraud", "downgrade", "upgrade", "price target", "short seller", "short report", "contract",
    "awarded", "partnership", "guidance raise", "raises guidance", "cuts guidance", "delisting", "halted",
    "trading halt", "data breach", "cyberattack", "outage", "strike", "plant closure", "production halt",
    # macro / policy
    "federal reserve", "the fed", "fomc", "rate cut", "rate hike", "interest rates", "powell", "inflation",
    "cpi", "ppi", "jobs report", "nonfarm payrolls", "unemployment", "gdp", "recession", "tariff", "tariffs",
    "trade deal", "trade war", "sanctions", "export controls", "export ban", "executive order", "stimulus",
    "government shutdown", "debt ceiling", "treasury yields", "oil prices", "opec", "chips act", "subsidy",
    "ban on", "nationalize", "price cap", "windfall tax",
]

# Upper-case words that look like tickers but almost never are, in news text.
NOT_TICKERS: set[str] = {
    "A", "I", "AI", "AM", "PM", "ET", "EST", "EDT", "PT", "GMT", "UTC", "US", "USA", "UK", "EU", "UN", "UAE",
    "CEO", "CFO", "COO", "CTO", "CIO", "VP", "EVP", "SVP", "IR", "PR", "HR", "IT", "TV", "AP", "UPI",
    "GDP", "CPI", "PPI", "PCE", "FOMC", "FED", "ECB", "BOJ", "BOE", "IMF", "OPEC", "NATO", "WHO", "WTO",
    "FDA", "SEC", "FTC", "FCC", "DOJ", "DOE", "DOD", "EPA", "IRS", "CDC", "NIH", "FAA", "NHTSA", "CFPB",
    "IPO", "ETF", "EPS", "YOY", "QOQ", "Q1", "Q2", "Q3", "Q4", "FY", "H1", "H2", "YTD", "ATH", "M&A",
    "NYSE", "NASDAQ", "AMEX", "OTC", "DJIA", "SPX", "VIX", "LLC", "INC", "CO", "CORP", "LTD", "PLC", "NV",
    "SA", "AG", "SE", "GOP", "DNC", "RNC", "POTUS", "SCOTUS", "BREAKING", "UPDATE", "UPDATED", "LIVE",
    "NEWS", "WATCH", "VIDEO", "EXCLUSIVE", "REPORT", "ALERT", "OK", "ON", "OR", "AND", "THE", "FOR", "NOT",
    "ALL", "NOW", "ARE", "CAN", "NEW", "ONE", "TWO", "BIG", "GO", "SO", "BE", "HE", "IS", "AT", "BY", "IN",
    "OF", "TO", "UP", "DO", "AN", "AS", "IF", "NO", "MY", "WE", "ME", "IT'S", "HAS", "WAS", "WILL", "OUT",
    "OFF", "NET", "ANY", "WAY", "KEY", "LOW", "HIGH", "TOP", "BEST", "REAL", "TRUE", "FREE", "PLAY", "FAST",
    "OPEN", "CASH", "MAIN", "BOX", "ARM", "LIFE", "SAFE", "SHE", "HIS", "HER", "WHY", "HOW", "WHAT", "WHO'S",
    "YES", "GDPNOW", "LNG", "EV", "EVS", "AR", "VR", "5G", "API", "CPU", "GPU", "PC", "PCS", "RAM", "SSD",
    "USD", "EUR", "JPY", "GBP", "CNY", "BTC", "ETH", "NFT", "ESG", "DEI", "COVID", "NFL", "NBA", "MLB",
    "NHL", "NCAA", "FIFA", "UFC", "CNN", "CNBC", "BBC", "ABC", "CBS", "NBC", "FOX", "MSNBC", "WSJ", "NYT",
}

# Single-word company names that are also everyday words - only matched when written as a $cashtag
# or ticker, never from plain text.
AMBIGUOUS_NAMES: set[str] = {
    "target", "block", "gap", "match", "box", "snap", "zoom", "shift", "unity", "progressive", "carnival",
    "general", "american", "national", "united", "first", "global", "international", "energy", "digital",
    "capital", "health", "power", "public", "world", "focus", "alpha", "beta", "delta", "visa",
    "sun", "star", "bank", "trust", "life", "home", "best", "ideal", "smart", "summit", "pioneer", "apex",
    "edge", "core", "prime", "vital", "liberty", "frontier", "pacific", "atlantic", "southern", "northern",
    "western", "eastern", "central", "standard", "premier", "select", "superior", "royal", "crown", "eagle",
    "ford", "dollar", "gold", "silver", "copper", "steel", "oil", "gas", "water", "solar", "wind", "fuel",
    "mobile", "cloud", "data", "net", "web", "link", "max", "plus", "one", "clear", "open", "true", "real",
    "evolution", "genesis", "nuance", "progress", "harmony", "spirit", "pulse", "vector", "matrix",
    "office", "chemical", "ventures", "partners", "holdings", "resources", "solutions", "systems",
    "technologies", "therapeutics", "pharmaceuticals", "biosciences", "bancorp", "financial", "realty",
}

# Hand-made aliases: brand, product or person -> ticker.
ALIASES: dict[str, str] = {
    "google": "GOOGL", "alphabet": "GOOGL", "youtube": "GOOGL", "waymo": "GOOGL", "sundar pichai": "GOOGL",
    "facebook": "META", "instagram": "META", "whatsapp": "META", "meta platforms": "META",
    "mark zuckerberg": "META", "zuckerberg": "META",
    "apple": "AAPL", "iphone": "AAPL", "tim cook": "AAPL",
    "microsoft": "MSFT", "satya nadella": "MSFT", "azure": "MSFT",
    "amazon": "AMZN", "aws": "AMZN", "andy jassy": "AMZN", "jeff bezos": "AMZN",
    "nvidia": "NVDA", "jensen huang": "NVDA",
    "tesla": "TSLA", "elon musk": "TSLA",
    "netflix": "NFLX", "walmart": "WMT", "costco": "COST", "home depot": "HD",
    "jpmorgan": "JPM", "jp morgan": "JPM", "jamie dimon": "JPM", "goldman sachs": "GS", "morgan stanley": "MS",
    "bank of america": "BAC", "wells fargo": "WFC", "citigroup": "C", "citi": "C",
    "berkshire hathaway": "BRK.B", "berkshire": "BRK.B", "warren buffett": "BRK.B",
    "coca-cola": "KO", "coca cola": "KO", "pepsico": "PEP", "pepsi": "PEP", "mcdonald's": "MCD", "mcdonalds": "MCD",
    "starbucks": "SBUX", "nike": "NKE", "disney": "DIS", "boeing": "BA", "lockheed martin": "LMT",
    "lockheed": "LMT", "northrop grumman": "NOC", "raytheon": "RTX", "general electric": "GE",
    "ge aerospace": "GE", "caterpillar": "CAT", "3m": "MMM", "general motors": "GM",
    "intel": "INTC", "amd": "AMD", "lisa su": "AMD", "broadcom": "AVGO", "qualcomm": "QCOM", "micron": "MU",
    "tsmc": "TSM", "taiwan semiconductor": "TSM", "super micro": "SMCI", "supermicro": "SMCI",
    "arm holdings": "ARM", "palantir": "PLTR", "salesforce": "CRM", "oracle": "ORCL", "ibm": "IBM",
    "adobe": "ADBE", "crowdstrike": "CRWD", "palo alto networks": "PANW", "shopify": "SHOP", "dell": "DELL",
    "exxon": "XOM", "exxonmobil": "XOM", "exxon mobil": "XOM", "chevron": "CVX", "conocophillips": "COP",
    "pfizer": "PFE", "moderna": "MRNA", "eli lilly": "LLY", "lilly": "LLY", "novo nordisk": "NVO",
    "johnson & johnson": "JNJ", "merck": "MRK", "unitedhealth": "UNH", "abbvie": "ABBV",
    "uber": "UBER", "airbnb": "ABNB", "doordash": "DASH", "paypal": "PYPL", "mastercard": "MA",
    "coinbase": "COIN", "robinhood": "HOOD", "microstrategy": "MSTR", "michael saylor": "MSTR",
    "spotify": "SPOT", "pinterest": "PINS", "reddit": "RDDT", "alibaba": "BABA",
    "at&t": "T", "verizon": "VZ", "t-mobile": "TMUS", "comcast": "CMCSA",
    "rivian": "RIVN", "lucid": "LCID", "gamestop": "GME",
    "s&p 500": "SPY", "nasdaq 100": "QQQ", "nasdaq-100": "QQQ", "russell 2000": "IWM",
}
