"""More news sources and international support: the preset catalogue, languages, translation, upgrades."""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime

import pytest

from newstrader.config import AppSettings, ConfigStore, SourceConfig
from newstrader.sources import presets
from newstrader.sources.base import NewsItem
from newstrader.sources.lang import detect_language
from newstrader.sources.rss import parse_feed

from .helpers import make_tickers


def test_preset_catalogue_is_valid():
    srcs = presets.default_sources()
    ids = [s["id"] for s in srcs]
    assert len(ids) == len(set(ids)) and len(srcs) >= 80
    settings = AppSettings()  # every preset passes validation
    assert len(settings.sources) == len(srcs)
    for s in settings.sources:
        assert s.region and s.category, s.id
        assert s.url.startswith("https://") or s.type == "alpaca_news", s.id
        if s.type == "stream":
            assert re.match(r"^https://www\.youtube\.com/(@[\w.-]+|c/\w+|user/\w+|channel/UC[\w-]{22})/live$", s.url), s.url
            assert s.translate == (s.language not in ("en", "")), s.id  # every foreign-language channel is translated
        if "news.google.com" in s.url:
            assert s.poll_seconds >= 300, s.id  # Google News rate-limits fast polling
    streams_on = [s for s in settings.sources if s.type == "stream" and s.enabled]
    assert len(streams_on) >= 3
    # the Trump / White House channels only go live for events, so they don't hold a slot when off air
    wh = next(s for s in settings.sources if s.id == "white-house")
    assert wh.enabled and wh.live_events and wh.category == "Live events"
    assert any(s.id == "trump-interviews" and s.enabled for s in settings.sources)
    regions = {s.region for s in settings.sources}
    assert {"US", "UK", "Europe", "Asia", "India", "Middle East", "Latin America"} <= regions


def test_new_presets_reach_existing_users_but_new_streams_stay_off(tmp_path):
    old = {"schema_version": 2, "sources": [s for s in AppSettings().model_dump(mode="json")["sources"]
                                             if s["id"] in ("bloomberg-tv", "cnbc-top")]}
    path = tmp_path / "config.json"
    path.write_text(json.dumps(old), encoding="utf-8")
    store = ConfigStore(path)
    ids = {s.id for s in store.settings.sources}
    assert {"dw-news", "nikkei-asia", "white-house", "trump-interviews"} <= ids
    assert not store.source("dw-news").enabled and not store.source("rsbn").enabled


def test_changed_preset_url_moves_users_who_never_edited_it(tmp_path, monkeypatch):
    from newstrader import config as cfg

    monkeypatch.setattr(cfg, "PRESET_URL_FIXES", {"yahoo-finance-news": ("https://old.example/rss",
                                                                         "https://new.example/rss")})
    data = AppSettings().model_dump(mode="json")
    for s in data["sources"]:
        if s["id"] == "yahoo-finance-news":
            s["url"] = "https://old.example/rss"
        if s["id"] == "cnbc-top":
            s["url"] = "https://my.own/feed"
    path = tmp_path / "config.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    store = ConfigStore(path)
    assert store.source("yahoo-finance-news").url == "https://new.example/rss"
    assert store.source("cnbc-top").url == "https://my.own/feed"
    assert json.loads(path.read_text(encoding="utf-8"))["sources"]  # saved straight away


@pytest.mark.parametrize("text,lang", [
    ("Apple beats estimates as iPhone sales surge in China", "en"),
    ("La Bolsa de Madrid cae tras los datos de inflación en la zona euro", "es"),
    ("Le CAC 40 recule après la décision de la BCE sur les taux", "fr"),
    ("Die Aktie von Siemens steigt nach starken Zahlen für das Quartal", "de"),
    ("Ibovespa fecha em alta com o avanço das ações da Petrobras", "pt"),
    ("トヨタ自動車、通期の営業利益見通しを上方修正", "ja"),
    ("삼성전자, 3분기 영업이익 시장 예상치 상회", "ko"),
    ("阿里巴巴第三季度营收超出预期", "zh"),
    ("Nestlé names new CEO as Société Générale shares rise", "en"),  # accents in names are still English
    ("NVDA", ""),
])
def test_language_detection(text, lang):
    assert detect_language(text) == lang


def test_feed_language_comes_from_the_source_or_is_detected():
    xml = b"""<?xml version="1.0"?><rss><channel>
      <item><title>Le CAC 40 recule apres la decision de la BCE sur les taux</title><guid>1</guid></item>
      <item><title>Apple shares rise after record quarter</title><guid>2</guid></item></channel></rss>"""
    auto = SourceConfig(id="f", type="rss", name="F", url="https://x/f.rss")
    items = parse_feed(xml, auto)
    assert [i.language for i in items] == ["fr", "en"]
    fixed = SourceConfig(id="f", type="rss", name="F", url="https://x/f.rss", language="fr")
    assert {i.language for i in parse_feed(xml, fixed)} == {"fr"}


async def test_local_engine_skips_foreign_text_but_claude_reads_it(ctx):
    from newstrader.ai.pipeline import Pipeline

    p = Pipeline(ctx, tickers=make_tickers(ctx.db))
    item = NewsItem(source_id="les-echos", source_type="rss", source_name="Les Echos", external_id="1",
                    title="Nvidia : le titre bondit après des résultats records selon les analystes de la banque",
                    published_at=datetime.now(UTC))
    res = await p.submit(item)
    assert res["status"] == "filtered" and "English" in res["reason"]
    row = ctx.db.query_one("SELECT language FROM news_items WHERE id = ?", (item.db_id,))
    assert row["language"] == "fr"
    ctx.config.update({"ai": {"engine": "claude"}})
    item2 = NewsItem(source_id="les-echos", source_type="rss", source_name="Les Echos", external_id="2",
                     title=item.title, published_at=datetime.now(UTC))
    res2 = await p.submit(item2)
    assert res2["status"] != "filtered" or "English" not in res2["reason"]


def test_claude_is_told_the_language():
    from newstrader.ai.prefilter import PrefilterResult
    from newstrader.ai.prompts import user_message

    item = NewsItem(source_id="x", source_type="rss", source_name="X", external_id="1", title="t", language="de")
    msg = user_message(item, PrefilterResult(False, [], []), datetime.now(UTC), True)
    assert "language: de" in msg
    item.language = "en"
    assert "language:" not in user_message(item, PrefilterResult(False, [], []), datetime.now(UTC), True)


def test_translated_stream_items_are_english():
    from newstrader.audio.stream_manager import Line, build_item

    src = SourceConfig(id="dw-deutsch", type="stream", name="DW Deutsch", url="https://www.youtube.com/c/dwdeutsch/live",
                       language="de", translate=True)
    item = build_item(src, [], [Line(1, 0.0, 1.0, "Interest rates rise in Germany.")])
    assert item.language == "en"
    src2 = SourceConfig(id="nhk", type="stream", name="NHK", url="https://www.youtube.com/@x/live", language="ja")
    assert build_item(src2, [], [Line(1, 0.0, 1.0, "...")]).language == "ja"
