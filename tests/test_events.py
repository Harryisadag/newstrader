"""The event rules: which news event a headline describes, and which company it is good or bad for."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from newstrader.ml.events import (
    ascii_fold,
    find_spans,
    guidance_surprise,
    macro_reading,
    number_surprise,
    read_events,
)
from newstrader.ml.sentiment import SentimentService

ROOT = Path(__file__).resolve().parent.parent


def read(text: str, *companies: tuple[str, str], actors: dict | None = None) -> dict:
    """{symbol: Reading} for companies given as (symbol, name)."""
    folded = ascii_fold(text)
    spans = {i: find_spans(folded, sym, [tuple(name.lower().split())]) for i, (sym, name) in enumerate(companies)}
    out = read_events(folded, spans, actors=actors)
    return {sym: out[i] for i, (sym, _n) in enumerate(companies)}


def test_analyst_house_is_neutral_and_the_rated_company_gets_the_call():
    r = read("Morgan Stanley upgrades Nike to Overweight, citing turnaround progress",
             ("MS", "Morgan Stanley"), ("NKE", "Nike"), actors={0: ("analyst", "target")})
    assert r["NKE"].direction == 1 and r["NKE"].event.kind == "analyst_upgrade"
    assert r["MS"].direction == 0
    # "upgrades ... to Buy" is a rating, not a takeover
    r = read("Goldman Sachs Upgrades Coinbase Global to Buy, Raises PT to $400",
             ("GS", "Goldman Sachs"), ("COIN", "Coinbase Global"), actors={0: ("analyst", "target")})
    assert r["COIN"].direction == 1 and r["GS"].direction == 0
    assert not any(e.kind.startswith("mna") for e in r["COIN"].events)
    r = read("TD Cowen cuts Lululemon target to $200 from $260", ("TD", "TD"), ("LULU", "Lululemon"),
             actors={0: ("analyst", "target")})
    assert r["LULU"].direction == -1 and r["TD"].direction == 0


def test_buyer_and_target():
    r = read("Blackstone agrees to take Hilton Grand Vacations private in $7 billion deal",
             ("BX", "Blackstone"), ("HGV", "Hilton Grand Vacations"))
    assert r["HGV"].direction == 1 and r["HGV"].strength >= 0.9
    assert r["BX"].direction == 0
    r = read("Apple in early talks to acquire Perplexity AI - Bloomberg", ("AAPL", "Apple"))
    assert r["AAPL"].direction == 0 and "buyer" in r["AAPL"].neutral_reason
    r = read("Thermo Fisher drops bid for Bio-Techne, citing due diligence findings", ("TECH", "Bio-Techne"))
    assert r["TECH"].direction == -1  # the bid is over, not "a bid for"


def test_forecast_beats_the_quarter():
    r = read("Southwest Airlines beats on profit, cuts unit revenue forecast for current quarter",
             ("LUV", "Southwest Airlines"))
    assert r["LUV"].direction == -1 and r["LUV"].event.kind == "guidance_cut"
    r = read("Datadog misses on EPS but raises full-year revenue guidance", ("DDOG", "Datadog"))
    assert r["DDOG"].direction == 1
    # the company that guides is the subject - Apple is only mentioned in passing
    r = read("Qualcomm tops estimates, but forecast disappoints as Apple modem business winds down",
             ("QCOM", "Qualcomm"), ("AAPL", "Apple"))
    assert r["QCOM"].direction == -1 and r["AAPL"].direction == 0
    # a price target is not a company forecast
    r = read("Analysts raised the price target on Nike", ("NKE", "Nike"))
    assert not any(e.kind.startswith("guidance") for e in r["NKE"].events)


def test_numbers():
    miss = number_surprise("Q3 adj EPS $1.20 vs $1.35 est; revenue $85B vs $89B est")
    assert miss.direction == -1 and "EPS" in miss.detail
    beat = number_surprise("Zoom Q2 adj. EPS $1.53 vs. $1.38 est; revenue $1.22B vs. $1.20B est")
    assert beat.direction == 1
    # a small EPS beat with a miss on same-store sales is a miss for a retailer
    comps = number_surprise("Target Q2 EPS $2.05 vs $2.03 est; comparable sales -1.9% vs -1.5% est")
    assert comps.direction == -1
    g = guidance_surprise("ON Semiconductor Sees Q4 Revenue $1.40B-$1.50B vs $1.61B Est")
    assert g.direction == -1 and g.priority == 2
    # the forecast numbers aren't read as this quarter's results
    assert number_surprise("ON Semiconductor Sees Q4 Revenue $1.40B-$1.50B vs $1.61B Est") is None


@pytest.mark.parametrize("title", [
    "Why Nvidia stock jumped today",
    "Shares of Boeing were down 4% in midday trading - here's what drove the decline",
    "Coinbase stock has doubled since April as spot bitcoin ETF flows surge",
    "Mid-Day Gainers: NVDA, AMD, SMCI, PLTR Lead Tech Higher",
])
def test_stories_about_a_move_that_already_happened_are_neutral(title):
    r = read(title, ("NVDA", "Nvidia"), ("BA", "Boeing"), ("COIN", "Coinbase"))
    assert all(x.direction == 0 for x in r.values())
    assert "recap" in next(iter(r.values())).flags


def test_a_move_next_to_real_news_keeps_the_news():
    r = read("PayPal misses on branded checkout growth; shares fall in premarket", ("PYPL", "PayPal"))
    assert r["PYPL"].direction == -1 and "moved" in r["PYPL"].flags
    r = read("AMD shares rally as AI optimism builds", ("AMD", "AMD"))
    assert r["AMD"].direction == 0 and "share move" in r["AMD"].neutral_reason


def test_opinion_and_law_firm_adverts_are_not_news():
    r = read("Is Nvidia stock a buy now?", ("NVDA", "Nvidia"))
    assert r["NVDA"].direction == 0 and "opinion" in r["NVDA"].flags
    r = read("INVESTOR ALERT: Bronstein, Gewirtz & Grossman LLC Investigating Lululemon Athletica Inc. (LULU) on "
             "Behalf of Investors", ("LULU", "Lululemon"))
    assert r["LULU"].direction == 0 and "law_firm_ad" in r["LULU"].flags


def test_denials_and_negations_flip_the_event():
    r = read("Mondelez says it has no current intention to make an offer for Hershey",
             ("MDLZ", "Mondelez"), ("HSY", "Hershey"))
    assert r["HSY"].direction == -1 and r["MDLZ"].direction == 0
    r = read("Coinbase CEO says company is not in talks to acquire Circle Internet",
             ("COIN", "Coinbase"), ("CRCL", "Circle Internet"))
    assert r["CRCL"].direction == -1 and r["COIN"].direction == 0
    r = read("AstraZeneca's Dato-DXd fails to improve overall survival in lung cancer trial", ("AZN", "AstraZeneca"))
    assert r["AZN"].direction == -1


def test_winners_losers_and_bystanders():
    r = read("Synchrony wins Walmart credit card program, replacing Capital One",
             ("SYF", "Synchrony"), ("WMT", "Walmart"), ("COF", "Capital One"))
    assert r["SYF"].direction == 1 and r["COF"].direction == -1 and r["WMT"].direction == 0
    r = read("Nvidia partner Super Micro may face Nasdaq delisting, sources say",
             ("NVDA", "Nvidia"), ("SMCI", "Super Micro"))
    assert r["SMCI"].direction == -1 and r["NVDA"].direction == 0
    assert "unconfirmed" in r["SMCI"].flags
    r = read("EU fines Apple 1.8 billion euros in music streaming case brought by Spotify",
             ("AAPL", "Apple"), ("SPOT", "Spotify"))
    assert r["AAPL"].direction == -1 and r["SPOT"].direction == 0
    r = read("UPS wins U.S. Postal Service air cargo contract, replacing FedEx", ("UPS", "UPS"), ("FDX", "FedEx"))
    assert r["UPS"].direction == 1 and r["FDX"].direction == -1


def test_routine_news_has_no_direction():
    for title in ("Realty Income raises monthly dividend for 132nd time",
                  "Johnson & Johnson declares regular quarterly dividend of $1.30 per share",
                  "Apple to hold its annual shareholder meeting on Tuesday"):
        r = read(title, ("X", title.split()[0]))
        assert r["X"].direction == 0, title


def test_country_news():
    assert macro_reading("Reserve Bank of India unexpectedly cuts repo rate by 50 bps").direction == 1
    assert macro_reading("Canada's economy unexpectedly shrinks in second quarter").direction == -1
    assert macro_reading("Germany approves 500 billion euro infrastructure fund, loosens debt brake").direction == 1


def test_spoken_acronyms_keep_positions():
    text = "the f d a has approved eli lilly's new drug"
    folded = ascii_fold(text)
    assert len(folded) == len(text) and "fda" in folded
    r = read(text, ("LLY", "eli lilly"))
    assert r["LLY"].direction == 1


def _eval_module():
    spec = importlib.util.spec_from_file_location("eval_detection", ROOT / "scripts" / "eval_detection.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("which,min_accuracy", [("dev", 95), ("set2", 94), ("set3", 89)])
def test_detection_quality_does_not_slip(which, min_accuracy, tmp_path):
    """The labelled headline sets (offline, word-list sentiment). Guards against a rule change breaking others."""
    ev = _eval_module()
    table = ev.make_table(tmp_path)  # tmp_path, not a TemporaryDirectory: Windows can't delete the open database
    sentiment = SentimentService(tmp_path / "models")
    sentiment.load("lexicon")
    res = ev.evaluate(ev.load_items(which), ev.make_engine(table, sentiment, True), table)
    assert res["accuracy_pct"] >= min_accuracy, res["mistakes"][:10]
    assert res["opposite_direction"] <= 1
    assert res["auto_trades_wrong"] <= 1
    assert res["wrong_company_hits"] <= 2


def test_but_outweighs_and_despite_discounts():
    r = read("Walmart beats and raises, but shares slip as margin outlook disappoints", ("WMT", "Walmart"))
    assert r["WMT"].direction == -1
    r = read("Super Micro shares drop as revenue forecast falls short despite profit beat", ("SMCI", "Super Micro"))
    assert r["SMCI"].direction == -1


def test_customers_contracts_and_rivals():
    r = read("Walmart To Shift Bulk Of Online Parcel Volume From FedEx To UPS Next Year",
             ("WMT", "Walmart"), ("FDX", "FedEx"), ("UPS", "UPS"))
    assert r["FDX"].direction == -1 and r["UPS"].direction == 1
    r = read("Northrop Grumman beats Boeing to win Navy's next-generation F/A-XX fighter contract",
             ("NOC", "Northrop Grumman"), ("BA", "Boeing"))
    assert r["NOC"].direction == 1 and r["BA"].direction == -1
    r = read("Army scraps $1.2 billion AeroVironment loitering-munition order", ("AVAV", "AeroVironment"))
    assert r["AVAV"].direction == -1
    r = read("Lilly's obesity pill beats Novo's Rybelsus in head-to-head diabetes trial",
             ("LLY", "Lilly"), ("NVO", "Novo"))
    assert r["LLY"].direction == 1 and r["NVO"].direction == -1
    r = read("OpenAI launches web browser in direct challenge to Google Chrome", ("GOOGL", "Google"))
    assert r["GOOGL"].direction == -1


def test_index_changes():
    r = read("Strategy snubbed again as S&P 500 adds Affirm in quarterly rebalance",
             ("MSTR", "Strategy"), ("AFRM", "Affirm"))
    assert r["MSTR"].direction == -1 and r["AFRM"].direction == 1
    r = read("Broadcom to replace 3M in the Dow Jones Industrial Average", ("AVGO", "Broadcom"), ("MMM", "3M"))
    assert r["AVGO"].direction == 1 and r["MMM"].direction == -1
    r = read("Enphase Energy to exit S&P 500, shift to SmallCap 600 index", ("ENPH", "Enphase Energy"))
    assert r["ENPH"].direction == -1


def test_activists_short_sellers_and_holders():
    r = read("Starboard Value Takes ~6% Stake In Pinterest, Plans To Push For Strategic Review",
             ("PINS", "Pinterest"))
    assert r["PINS"].direction == 1
    r = read("Hertz Shares Jump After Bill Ackman's Pershing Square Reveals Nearly 20% Stake", ("HTZ", "Hertz"))
    assert r["HTZ"].direction == 1
    r = read("Muddy Waters shorts AppLovin, alleges it violates app-store terms", ("APP", "AppLovin"))
    assert r["APP"].direction == -1
    r = read("Citron Research covers Carvana short, says it is now long the stock", ("CVNA", "Carvana"))
    assert r["CVNA"].direction == 1
    r = read("SoftBank Sells $4.8B Of T-Mobile US Shares Via Overnight Block Trade", ("TMUS", "T-Mobile"))
    assert r["TMUS"].direction == -1


def test_things_that_look_like_news_but_are_not():
    r = read("Evercore ISI reiterates Amazon as top pick into third-quarter earnings",
             ("EVR", "Evercore"), ("AMZN", "Amazon"), actors={0: ("analyst", "target")})
    assert r["AMZN"].direction == 0 and r["EVR"].direction == 0
    r = read("Netflix begins trading on split-adjusted basis following 10-for-1 stock split", ("NFLX", "Netflix"))
    assert r["NFLX"].direction == 0
    r = read("Tesla Recalls 1.2 Million Vehicles Over Rearview Camera Delay; Fix Delivered Via Over-The-Air "
             "Software Update", ("TSLA", "Tesla"))
    assert r["TSLA"].direction == 0
    r = read("Lam Research Announces Pricing Of $1.5 Billion Senior Notes Offering", ("LRCX", "Lam Research"))
    assert r["LRCX"].direction == 0
    # "takes over as CEO" doesn't make the hiring company a loser
    r = read("Chipotle poaches Wingstop's chief executive. He takes over as CEO in March.",
             ("CMG", "Chipotle"), ("WING", "Wingstop"))
    assert r["CMG"].direction == 1 and r["WING"].direction == -1


def test_legal_outcomes_and_denied_deals():
    r = read("Arm loses licensing fight as jury sides with Qualcomm", ("ARM", "Arm"), ("QCOM", "Qualcomm"))
    assert r["ARM"].direction == -1 and r["QCOM"].direction == 1
    r = read("Judge spares Google a Chrome breakup in search monopoly case", ("GOOGL", "Google"))
    assert r["GOOGL"].direction == 1
    r = read("Warner Bros. Discovery Says It Has Not Received Any Proposal From Comcast, Denying Media Report",
             ("WBD", "Warner Bros. Discovery"), ("CMCSA", "Comcast"))
    assert r["WBD"].direction == -1 and r["CMCSA"].direction == 0
    r = read("Comcast tops Paramount Skydance offer for Warner Bros Discovery with $32-a-share bid",
             ("CMCSA", "Comcast"), ("PSKY", "Paramount Skydance"), ("WBD", "Warner Bros Discovery"))
    assert r["WBD"].direction == 1 and r["CMCSA"].direction == 0 and r["PSKY"].direction == 0


@pytest.mark.parametrize("title,companies,expected", [
    # advice is not a takeover
    ("UBS tells clients it's too early to buy Nvidia", [("NVDA", "Nvidia")], {"NVDA": 0}),
    ("Morgan Stanley warns clients against rushing to buy Apple", [("AAPL", "Apple")], {"AAPL": 0}),
    ("Lowe's to buy Floor & Decor for $10.5 billion", [("LOW", "Lowe's"), ("FND", "Floor & Decor")],
     {"FND": 1, "LOW": 0}),
    # numbers belong to the company that reported them
    ("Apple supplier Qualcomm Q3 EPS $2.80 vs $2.50 est", [("AAPL", "Apple"), ("QCOM", "Qualcomm")],
     {"QCOM": 1, "AAPL": 0}),
    ("Walmart rival Target Q2 comps -1.9% vs -1.5% est", [("WMT", "Walmart"), ("TGT", "Target")],
     {"TGT": -1, "WMT": 0}),
    # units and losses
    ("Coca-Cola earned $1.05 per share, above analysts' average estimate of 98 cents", [("KO", "Coca-Cola")],
     {"KO": 1}),
    ("Snowflake sees Q3 revenue $950M-$1.05B vs $1.0B est", [("SNOW", "Snowflake")], {"SNOW": 0}),
    ("Rivian Q2 loss per share $0.97 vs $1.20 expected", [("RIVN", "Rivian")], {"RIVN": 1}),
    ("Rivian lost $0.97 a share, versus the $1.20 loss analysts expected", [("RIVN", "Rivian")], {"RIVN": 1}),
    # negated or undone bad news
    ("Intel says it will not cut its dividend", [("INTC", "Intel")], {"INTC": 0}),
    ("FDA removes clinical hold on Intellia gene-editing study", [("NTLA", "Intellia")], {"NTLA": 1}),
    ("Hertz emerges from Chapter 11 protection", [("HTZ", "Hertz")], {"HTZ": 0}),
    ("Novo Nordisk stops kidney trial early due to efficacy", [("NVO", "Novo Nordisk")], {"NVO": 1}),
    # only the real subject gets the news
    ("Samsung, Apple's main rival, cuts profit forecast", [("AAPL", "Apple")], {"AAPL": 0}),
    ("Unlike Walmart, Target cuts forecast", [("WMT", "Walmart"), ("TGT", "Target")], {"WMT": 0, "TGT": -1}),
    ("Walgreens, CVS gain as Rite Aid files for bankruptcy", [("WBA", "Walgreens"), ("CVS", "CVS")],
     {"WBA": 0, "CVS": 0}),
    ("Super Micro shares drop as revenue forecast falls short despite profit beat", [("SMCI", "Super Micro")],
     {"SMCI": -1}),
    ("Court blocks Trump tariffs, boosting Apple and Nike", [("AAPL", "Apple"), ("NKE", "Nike")],
     {"AAPL": 0, "NKE": 0}),
])
def test_review_findings_stay_fixed(title, companies, expected):
    actors = {i: ("analyst", "target", "guidance") for i, (sym, _n) in enumerate(companies) if sym in {"GS", "MS"}}
    r = read(title, *companies, actors=actors)
    assert {sym: r[sym].direction for sym in expected} == expected, {s: (r[s].direction, r[s].label) for s in r}


def test_bank_forecasts_are_not_company_guidance():
    for title in ("Goldman Sachs raises gold forecast to $4,000", "Goldman Sachs lowers US GDP forecast on tariffs"):
        r = read(title, ("GS", "Goldman Sachs"), actors={0: ("analyst", "target", "guidance")})
        assert r["GS"].direction == 0, title
