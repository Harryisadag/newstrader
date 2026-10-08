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
    # everyday abbreviations that are also tickers (MS = multiple sclerosis, HD video, PG-13, MA degree...)
    "MS", "HD", "COST", "CAR", "CARS", "DE", "ED", "MA", "PG", "WELL", "LOVE", "FUN", "EAT", "EYE", "HOME", "TECH",
    "AIR", "JOBS", "SHOP", "WOW", "PAY", "DOG", "CAT", "RUN", "SEE", "SAY", "TELL", "MOVE", "NICE",
    "HUGE", "BOOM", "PEAK", "RACE", "WIN", "WINS", "PLUS", "SAVE", "STAY", "TRIP", "BEAT", "HOPE", "CARE", "BRO",
    "GLAD", "VERY", "ALL-IN", "FY24", "FY25", "FY26", "FY27", "RV", "SUV", "OEM", "SPAC", "PE", "VC", "CRE",
    "IPOS", "LBO", "ROI", "ROE", "EBIT", "EBITDA", "FCF", "ARR", "AUM", "NIM", "TAM", "SAAS", "LLM", "AGI", "GPT",
    "ASIC", "HBM", "DRAM", "NAND", "TPU", "DC", "NY", "LA", "SF", "DC'S", "EU'S", "UK'S", "US'S", "IMF'S",
}

# Ordinary words that must never be used as a company's short name ("Applied Materials" is never just "Applied")
COMMON_WORDS: set[str] = {
    "applied", "advanced", "first", "new", "great", "good", "big", "global", "american", "national", "united",
    "general", "international", "digital", "energy", "power", "health", "bio", "pharma", "medical", "financial",
    "capital", "home", "life", "smart", "green", "blue", "red", "black", "white", "gold", "silver", "golden",
    "royal", "western", "southern", "northern", "eastern", "central", "pacific", "atlantic", "star", "sun",
    "bright", "clear", "open", "next", "future", "modern", "total", "universal", "integrated", "innovative",
    "precision", "quantum", "alpha", "omega", "delta", "prime", "core", "edge", "summit", "peak", "vision",
    "insight", "fusion", "nova", "nexus", "matrix", "vector", "pulse", "spark", "bridge", "harbor", "anchor",
    "beacon", "compass", "heritage", "legacy", "liberty", "freedom", "patriot", "pioneer", "frontier", "horizon",
    "apex", "zenith", "stellar", "solid", "strong", "cross", "main", "river", "lake", "mountain", "valley",
    "ocean", "bay", "coast", "island", "city", "town", "state", "county", "world", "planet", "earth", "sky",
    "water", "fire", "allied", "consolidated", "associated", "standard", "premier", "select", "superior",
    "dynamic", "creative", "direct", "express", "rapid", "simple", "secure", "trusted", "ultra", "super", "mega",
    "micro", "nano", "data", "cloud", "mobile", "media", "network", "systems", "service", "services", "products",
    "industrial", "commercial", "residential", "realty", "property", "properties", "land", "farm", "food", "foods",
    "beverage", "auto", "motor", "air", "sea", "rail", "transport", "logistics", "marine", "aviation", "space",
    "defense", "security", "safety", "rocket", "lab", "labs", "genetics", "gene", "cell", "cells", "body",
    "mind", "heart", "brain", "eye", "vital", "care", "cure", "hope", "sage", "true", "real", "ideal", "best",
    "better", "plus", "max", "one", "two", "three", "ten", "hundred", "thousand", "million", "texas",
    "california", "florida", "alaska", "hawaii", "arizona", "nevada", "carolina", "virginia", "georgia",
    "oregon", "washington", "boston", "chicago", "denver", "dallas", "houston", "atlanta", "miami", "phoenix",
    "seattle", "detroit", "canada", "mexico", "china", "japan", "india", "brazil", "europe", "asia", "africa",
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
    "dow", "southwest",
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
    "rivian": "RIVN", "lucid": "LCID", "gamestop": "GME", "bristol myers": "BMY", "bristol-myers": "BMY",
    "e.l.f. beauty": "ELF", "elf beauty": "ELF", "delta air lines": "DAL", "hoka": "DECK", "ugg": "DECK",
    "deckers": "DECK", "ge healthcare": "GEHC", "ge vernova": "GEV", "keytruda": "MRK", "wegovy": "NVO",
    "s&p 500": "SPY", "nasdaq 100": "QQQ", "nasdaq-100": "QQQ", "russell 2000": "IWM",
    # more US brands, products and people
    "copilot": "MSFT", "xbox": "MSFT", "linkedin": "MSFT", "android": "GOOGL", "chrome": "GOOGL", "whole foods": "AMZN", "prime video": "AMZN", "alexa": "AMZN",
    "blackwell": "NVDA", "geforce": "NVDA", "cuda": "NVDA", "cybertruck": "TSLA", "model y": "TSLA",
    "ozempic": "NVO", "zepbound": "LLY", "mounjaro": "LLY",
    "humira": "ABBV", "chipotle": "CMG", "lululemon": "LULU", "petco": "WOOF",
    "peloton": "PTON", "alaska airlines": "ALK", "southwest airlines": "LUV", "american airlines": "AAL", "united airlines": "UAL", "jetblue": "JBLU", "estee lauder": "EL",
    "capital one": "COF", "synchrony": "SYF", "lowe's": "LOW", "lowes": "LOW", "kroger": "KR",
    "albertsons": "ACI", "dollar general": "DG", "dollar tree": "DLTR", "walgreens": "WBA", "cvs": "CVS",
    "ups": "UPS", "fedex": "FDX", "occidental": "OXY", "halliburton": "HAL",
    "schlumberger": "SLB", "nextera": "NEE", "duke energy": "DUK", "constellation energy": "CEG",
    "vistra": "VST", "applovin": "APP", "emcor": "EME", "snowflake": "SNOW", "datadog": "DDOG",
    "workday": "WDAY", "servicenow": "NOW", "marvell": "MRVL", "arista": "ANET", "cisco": "CSCO",
    "texas instruments": "TXN", "applied materials": "AMAT", "lam research": "LRCX", "kla": "KLAC",
    "on semiconductor": "ON", "onsemi": "ON", "western digital": "WDC", "seagate": "STX",
    "mike wirth": "CVX", "darren woods": "XOM",
    "mary barra": "GM", "jim farley": "F", "bob iger": "DIS",
    "ted sarandos": "NFLX", "brian moynihan": "BAC", "david solomon": "GS", "ted pick": "MS", "jane fraser": "C",
    "dara khosrowshahi": "UBER", "brian chesky": "ABNB", "marc benioff": "CRM", "larry ellison": "ORCL",
    "safra catz": "ORCL", "michael dell": "DELL", "arvind krishna": "IBM", "hock tan": "AVGO",
    "cristiano amon": "QCOM", "lip-bu tan": "INTC", "pat gelsinger": "INTC", "doug mcmillon": "WMT",
    "john furner": "WMT", "brian niccol": "SBUX", "albert bourla": "PFE", "dave ricks": "LLY",
    "david ricks": "LLY", "kelly ortberg": "BA", "chuck robbins": "CSCO", "ryan cohen": "GME",
    # international companies, by the name headlines use -> their US listing (ADR)
    "toyota": "TM", "sony": "SONY", "honda": "HMC", "nissan": "NSANY", "mitsubishi ufj": "MUFG", "mufg": "MUFG",
    "sumitomo mitsui": "SMFG", "nomura": "NMR", "softbank": "SFTBY", "nintendo": "NTDOY",
    "asml": "ASML", "sap": "SAP", "novartis": "NVS", "astrazeneca": "AZN", "gsk": "GSK", "glaxosmithkline": "GSK",
    "sanofi": "SNY", "shell": "SHEL", "bp": "BP", "hsbc": "HSBC", "ubs": "UBS", "barclays": "BCS",
    "deutsche bank": "DB", "santander": "SAN", "bbva": "BBVA", "ing": "ING", "unilever": "UL", "diageo": "DEO",
    "british american tobacco": "BTI", "rio tinto": "RIO", "bhp": "BHP", "vale": "VALE", "petrobras": "PBR",
    "infosys": "INFY", "wipro": "WIT", "hdfc bank": "HDB", "icici bank": "IBN", "icici": "IBN", "jd.com": "JD",
    "jd com": "JD", "baidu": "BIDU", "pinduoduo": "PDD", "temu": "PDD", "nio": "NIO", "xpeng": "XPEV",
    "li auto": "LI", "stellantis": "STLA", "ferrari": "RACE", "spotify technology": "SPOT", "arm": "ARM",
    "sea limited": "SE", "mercadolibre": "MELI", "nu holdings": "NU", "nubank": "NU", "totalenergies": "TTE",
    "equinor": "EQNR", "nokia": "NOK", "ericsson": "ERIC", "philips": "PHG", "royal philips": "PHG",
    "siemens energy": "SMNEY", "novo": "NVO", "tencent": "TCEHY", "byd": "BYDDY", "lvmh": "LVMUY",
    "nestle": "NSRGY", "roche": "RHHBY", "airbus": "EADSY", "volkswagen": "VWAGY",
    "mercedes-benz": "MBGYY", "bmw": "BMWYY", "rheinmetall": "RNMBY", "taiwan semiconductor manufacturing": "TSM",
}

# Aliases that are people: a person talking about the economy isn't news about their company
PERSON_ALIASES: set[str] = {
    "sundar pichai", "mark zuckerberg", "zuckerberg", "tim cook", "satya nadella", "andy jassy",
    "jeff bezos", "jensen huang", "elon musk", "jamie dimon", "warren buffett", "lisa su", "michael saylor",
    "mike wirth", "darren woods", "mary barra", "jim farley", "bob iger", "ted sarandos", "brian moynihan",
    "david solomon", "ted pick", "jane fraser", "dara khosrowshahi", "brian chesky", "marc benioff",
    "larry ellison", "safra catz", "michael dell", "arvind krishna", "hock tan", "cristiano amon", "lip-bu tan",
    "pat gelsinger", "doug mcmillon", "john furner", "brian niccol", "albert bourla", "dave ricks", "david ricks",
    "kelly ortberg", "chuck robbins", "ryan cohen",
}
# Elon Musk runs several companies; only Tesla is listed
MUSK_OTHER_COMPANIES = {"spacex", "xai", "x", "twitter", "neuralink", "starlink", "boring", "doge", "grok"}

# A single-word alias followed by one of these words means something else ("Amazon rainforest", "apple pie")
ALIAS_NOT_BEFORE: dict[str, set[str]] = {
    "amazon": {"rainforest", "river", "basin", "jungle", "forest", "region"},
    "apple": {"pie", "pies", "juice", "cider", "orchard", "orchards", "tree", "trees", "sauce", "picking"},
    "oracle": {"of"}, "berkshire": {"county", "hills", "mountains"}, "citi": {"bike", "field"},
    "shell": {"company", "companies", "shock", "game", "out"}, "vale": {"of"}, "nio": set(),
    "sea": set(), "arm": {"of", "wrestling"}, "ing": set(), "novo": set(), "lilly": {"pad"},
    "delta": {"variant", "region"}, "target": {"price", "rate", "range", "date"},
    "dow": {"jones", "industrials", "futures", "average", "transports", "utilities", "index"},
}
# Words next to a name that show it's the company ("Target shares", "Ford recalls", "shares of Gap")
COMPANY_CONTEXT_AFTER: set[str] = {
    "shares", "share", "stock", "stocks", "inc", "corp", "corporation", "co", "plc", "earnings", "quarterly",
    "q1", "q2", "q3", "q4", "profit", "profits", "sales", "revenue", "results", "forecast", "outlook", "guidance",
    "ceo", "cfo", "chief", "beats", "misses", "tops", "cuts", "raises", "reports", "posts", "recalls", "recall",
    "deliveries", "announces", "said", "says", "lays", "slashes", "hikes", "lifts", "trims", "warns", "plans",
    "motor", "motors", "air", "airlines", "lines", "cruise", "cruises", "stores", "store", "investors",
    "executives", "employees", "workers", "dividend", "buyback", "inc's", "corp's",
    "moving", "trading",
}
# A euro-area country's news also counts for the euro-area fund
EUROZONE_ETF = "EZU"
EUROZONE_COUNTRY_ETFS: set[str] = {"EWG", "EWQ", "EWI", "EWP"}
# Index ETFs named after their index: "joins the S&P 500" is about the joining company, not SPY
INDEX_ETF_ALIASES: set[str] = {"SPY", "QQQ", "IWM", "DIA"}
INDEX_MEMBERSHIP_WORDS: set[str] = {"join", "joins", "joining", "joined", "to", "from", "enter", "enters", "entering",
                                    "into", "in", "leave", "leaves", "leaving", "for", "exit", "exits", "exiting"}
INDEX_CHANGE_WORDS_AFTER: set[str] = {"adds", "add", "removes", "remove", "drops", "drop", "inclusion", "rebalance",
                                      "rebalancing", "reshuffle", "constituent", "constituents", "membership"}
COMPANY_CONTEXT_BEFORE: set[str] = {
    "of", "rival", "retailer", "automaker", "carrier", "airline", "chipmaker", "lender", "insurer", "maker",
    "giant", "company", "firm", "owner", "shares", "stock", "brand", "operator", "processor",
    # verbs that take a company as their object ("Wells Fargo downgrades Target", "DOJ sues Visa")
    "upgrades", "downgrades", "upgraded", "downgraded", "initiates", "reiterates", "sues", "sued", "fines", "fined",
    "probes", "acquire", "acquires", "buy", "buys", "versus", "vs", "against", "take", "taking",
}
# Short names headlines use that aren't in the official name; they still need company context when everyday words
CONTEXT_NAMES: dict[str, str] = {"delta": "DAL", "southwest": "LUV", "royal caribbean": "RCL"}
# Buyout firms: in deal news they're the buyer, not the company the deal is about
PRIVATE_EQUITY: set[str] = {"BX", "KKR", "APO", "CG", "ARES", "TPG", "BAM", "BN"}

# Big brokers and banks: when they are the ones doing an analyst action, the action is about another company
BROKERS: set[str] = {
    "GS", "MS", "JPM", "BAC", "C", "WFC", "UBS", "DB", "BCS", "HSBC", "JEF", "RJF", "SF", "PIPR", "EVR", "LAZ",
    "MUFG", "CS", "NMR", "BMO", "RY", "TD", "SCHW", "COWN", "OPY",
}

# International macro news -> the US-listed fund that holds that country's stocks (always manual review)
COUNTRY_ETFS: dict[str, str] = {
    "japan": "EWJ", "japanese": "EWJ", "bank of japan": "EWJ", "boj": "EWJ", "tokyo": "EWJ", "nikkei": "EWJ",
    "china": "FXI", "chinese": "FXI", "beijing": "FXI", "pboc": "FXI", "people's bank of china": "FXI",
    "germany": "EWG", "german": "EWG", "bundesbank": "EWG", "berlin": "EWG",
    "britain": "EWU", "uk": "EWU", "british": "EWU", "bank of england": "EWU", "boe": "EWU",
    "india": "INDA", "indian": "INDA", "rbi": "INDA", "reserve bank of india": "INDA",
    "brazil": "EWZ", "brazilian": "EWZ", "south korea": "EWY", "korea": "EWY", "korean": "EWY", "seoul": "EWY",
    "taiwan": "EWT", "taiwanese": "EWT", "eurozone": "EZU", "euro zone": "EZU", "euro area": "EZU",
    "ecb": "EZU", "european central bank": "EZU", "canada": "EWC", "canadian": "EWC", "bank of canada": "EWC",
    "australia": "EWA", "australian": "EWA", "rba": "EWA", "mexico": "EWW", "mexican": "EWW",
    "france": "EWQ", "french": "EWQ", "italy": "EWI", "italian": "EWI", "spain": "EWP", "spanish": "EWP",
    "switzerland": "EWL", "swiss": "EWL", "hong kong": "EWH",
}
MACRO_WORDS: set[str] = {
    "rate", "rates", "stimulus", "economy", "economic", "gdp", "inflation", "recession", "growth", "central",
    "election", "tariff", "tariffs", "sanctions", "martial", "currency", "yen", "yuan", "won", "rupee", "euro",
    "pound", "bond", "bonds", "yields", "contraction", "shrinks", "shrank", "contracts", "expands", "unemployment",
    "jobs", "exports", "imports", "trade", "budget", "deficit", "stocks", "market", "markets", "coup", "war",
    "invasion", "default", "downgrade", "easing", "tightening", "hike", "hikes", "cut", "cuts", "policy",
    "infrastructure", "spending", "fiscal", "debt", "repo", "runoff", "presidential", "parliament", "assets",
    "reserve", "requirement", "property", "no-confidence", "government", "lawmakers",
}
