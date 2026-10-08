"""The event rules: which news event a headline describes, and which company it is good or bad for."""

from __future__ import annotations

import importlib.util
import tempfile
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


@pytest.mark.parametrize("which,min_accuracy", [("dev", 95), ("set2", 94)])
def test_detection_quality_does_not_slip(which, min_accuracy):
    """The labelled headline sets (offline, word-list sentiment). Guards against a rule change breaking others."""
    ev = _eval_module()
    with tempfile.TemporaryDirectory() as tmp:
        table = ev.make_table(Path(tmp))
        sentiment = SentimentService(Path(tmp) / "models")
        sentiment.load("lexicon")
        res = ev.evaluate(ev.load_items(which), ev.make_engine(table, sentiment, True), table)
    assert res["accuracy_pct"] >= min_accuracy, res["mistakes"][:10]
    assert res["opposite_direction"] <= 1
    assert res["auto_trades_wrong"] <= 1
    assert res["wrong_company_hits"] <= 2
