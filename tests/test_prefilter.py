from __future__ import annotations

import pytest

from newstrader.ai.dedupe import StoryDeduper, fingerprint, similarity
from newstrader.ai.prefilter import prefilter
from newstrader.ai.tickers import clean_company_name
from newstrader.db import Database, utcnow

from .helpers import make_tickers


@pytest.fixture
def tickers(tmp_path):
    return make_tickers(Database(tmp_path / "t.db"))


def syms(res):
    return [c.symbol for c in res.candidates]


@pytest.mark.parametrize("name,clean", [
    ("Apple Inc. Common Stock", "apple"),
    ("Alphabet Inc. Class C Capital Stock", "alphabet"),
    ("JPMorgan Chase & Co. Common Stock", "jpmorgan chase"),
    ("Ford Motor Company Common Stock", "ford motor"),
    ("NVIDIA Corporation Common Stock", "nvidia"),
    ("The Allstate Corporation Common Stock", "allstate"),
])
def test_clean_company_name(name, clean):
    assert clean_company_name(name) == clean


def test_otc_excluded(tickers):
    assert tickers.get("OTCX") is None
    assert tickers.get("AAPL") is not None


def test_cashtag_and_exchange_patterns(tickers):
    assert "AAPL" in syms(prefilter("Big day for $AAPL holders", tickers))
    assert "F" in syms(prefilter("Ford Motor Company (NYSE: F) recalls trucks", tickers))
    assert "F" in syms(prefilter("Shares of $F rallied", tickers))


def test_company_names_and_aliases(tickers):
    assert syms(prefilter("Nvidia unveils a new chip", tickers)) == ["NVDA"]
    assert "AAPL" in syms(prefilter("Apple's iPhone sales slump in China", tickers))
    assert "NVDA" in syms(prefilter("Jensen Huang says demand is insane", tickers))
    assert "TSLA" in syms(prefilter("Elon Musk teases robotaxi launch", tickers))
    assert "JPM" in syms(prefilter("JPMorgan Chase reports record profit", tickers))
    assert "MCD" in syms(prefilter("McDonald's raises prices again", tickers))
    goog = syms(prefilter("Alphabet faces new antitrust case", tickers))
    assert "GOOGL" in goog  # alias picks the main share class


def test_uppercase_tickers_but_not_common_words(tickers):
    assert "NVDA" in syms(prefilter("NVDA up 5% premarket", tickers))
    # "ALL" is a ticker (Allstate) but also a word - not matched from plain caps
    assert syms(prefilter("ALL eyes on the jobs report", tickers)) == []


def test_everyday_word_names_need_company_context(tickers):
    # "Target", "Ford", "Delta" are everyday words: only the company when the words around say so
    for text in ("We need to target inflation", "Analysts raise their price target for gold",
                 "Target price for Apple raised", "Ford the river at the shallow end", "Target date funds grow"):
        assert "TGT" not in syms(prefilter(text, tickers)) and "F" not in syms(prefilter(text, tickers)), text
    assert "TGT" in syms(prefilter("Target misses on earnings", tickers))
    assert "TGT" in syms(prefilter("Target shares slide after weak holiday forecast", tickers))
    assert "TGT" in syms(prefilter("Shares of Target fell", tickers))
    assert "F" in syms(prefilter("Ford recalls 300,000 F-150 trucks", tickers))
    assert "TGT" in syms(prefilter("$TGT misses on earnings", tickers))


def test_name_matching_details(tickers):
    assert syms(prefilter("Ford Motor raises outlook - Reuters", tickers)) == ["F"]  # " - " used to break matching
    assert syms(prefilter("the best apple pie recipe", tickers)) == []  # not capitalised, and "apple pie"
    assert syms(prefilter("Amazon rainforest fires spread", tickers)) == []
    assert syms(prefilter("Elon Musk's SpaceX launches Starship", tickers)) == []
    assert syms(prefilter("Musk says Tesla robotaxi launch delayed", tickers)) == ["TSLA"]
    assert syms(prefilter("tesla deliveries came in well below the street", tickers)) == ["TSLA"]  # lower-case speech


def test_shouty_headlines_dont_match_every_word(tickers):
    res = prefilter("BREAKING: MARKETS RALLY AS INVESTORS CHEER NEWS", tickers)
    assert syms(res) == []


def test_keywords_and_keyword_only_setting(tickers):
    res = prefilter("Fed signals a rate cut in December", tickers)
    assert res.hit and "rate cut" in res.keywords and not res.candidates
    assert not prefilter("Fed signals a rate cut in December", tickers, allow_keyword_only=False).hit


def test_irrelevant_text_filtered(tickers):
    res = prefilter("Local bakery wins award for best croissant in town", tickers)
    assert not res.hit


def test_source_symbols_pass_through(tickers):
    res = prefilter("Analyst note", tickers, source_symbols=["TSLA", "FAKE"])
    assert syms(res) == ["TSLA"]


def test_unknown_cashtag_ignored_when_table_loaded(tickers):
    assert syms(prefilter("$ZZZZ to the moon", tickers)) == []


# ---------------------------------------------------------------- story dedupe
def test_fingerprint_similarity():
    a = fingerprint("Nvidia to invest $5 billion in Intel")
    b = fingerprint("NVIDIA to invest $5 billion in Intel - report")
    c = fingerprint("Apple unveils new iPhone at September event")
    assert similarity(a, b) >= 0.8
    assert similarity(a, c) < 0.2


def test_deduper_window_and_url():
    d = StoryDeduper()
    now = utcnow()
    kw = dict(window_minutes=15, threshold=0.6)
    assert d.check_and_add(item_id=1, text="Nvidia to invest $5 billion in Intel", url="https://a.com/story/nvidia-intel-5b", source_id="a", at=now, **kw) is None
    assert d.check_and_add(item_id=2, text="NVIDIA to invest $5 billion in Intel", url="https://b.com/story/nvidia-intel", source_id="b", at=now, **kw) == 1
    assert d.check_and_add(item_id=3, text="totally different words here", url="https://a.com/story/nvidia-intel-5b", source_id="c", at=now, **kw) == 1
    from datetime import timedelta

    later = now + timedelta(minutes=30)
    assert d.check_and_add(item_id=4, text="Nvidia to invest $5 billion in Intel", url="https://d.com/story/nvidia-intel-9", source_id="d", at=later, **kw) is None


def test_generic_urls_dont_count_as_same_story():
    d = StoryDeduper()
    now = utcnow()
    kw = dict(window_minutes=15, threshold=0.6, at=now)
    assert d.check_and_add(item_id=1, text="Nvidia wins contract", url="https://example.com", source_id="a", **kw) is None
    assert d.check_and_add(item_id=2, text="Tesla recalls cars", url="https://example.com/", source_id="b", **kw) is None
    assert d.check_and_add(item_id=3, text="Other words entirely", url="https://site.com/news/nvidia-wins-contract-123",
                           source_id="c", **kw) is None
    assert d.check_and_add(item_id=4, text="Different headline", url="https://SITE.com/news/nvidia-wins-contract-123/",
                           source_id="d", **kw) == 3
