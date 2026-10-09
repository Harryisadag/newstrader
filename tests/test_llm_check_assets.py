"""scripts/check_llm_assets.py --verify-pins, with canned GitHub and Hugging Face answers."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import httpx
import pytest

from newstrader.llm import catalog

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "check_llm_assets.py"


@pytest.fixture(scope="module")
def checker():
    spec = importlib.util.spec_from_file_location("check_llm_assets", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def release_json(**changes) -> dict:
    assets = []
    for b in catalog.SERVER_BUILDS.values():
        for a in b.assets:
            assets.append({"name": a.name, "size": a.size, "digest": f"sha256:{a.sha256}"})
    assets.append({"name": "llama-b11512-bin-something-else.zip", "size": 1, "digest": "sha256:00"})
    for a in assets:
        a.update(changes.get(a["name"], {}))
    return {"tag_name": catalog.LLAMA_CPP_BUILD, "assets": assets}


def tree_json(repo: str, revision: str, **changes) -> list:
    out = [{"type": "file", "path": "README.md", "size": 10}]
    for m in catalog.MODELS:
        if (m.repo, m.revision) == (repo, revision):
            out.append({"type": "file", "path": m.file, "size": m.size,
                        "lfs": {"oid": m.sha256, "size": m.size, "pointerSize": 135}, **changes.get(m.file, {})})
    return out


def test_matching_pins(checker):
    rows = checker.check_build_pins(release_json())
    assert len(rows) == sum(len(b.assets) for b in catalog.SERVER_BUILDS.values())
    assert all(r["ok"] for r in rows)
    for m in catalog.MODELS:
        assert checker.check_model_pin(m, tree_json(m.repo, m.revision))["ok"]


def test_mismatches_are_named(checker):
    first = next(iter(catalog.SERVER_BUILDS.values())).assets[0]
    rows = checker.check_build_pins(release_json(**{first.name: {"size": first.size + 1, "digest": "sha256:ab"}}))
    bad = [r for r in rows if not r["ok"]]
    assert len(bad) == 1 and "size" in bad[0]["problem"] and "SHA-256" in bad[0]["problem"]
    rows = checker.check_build_pins(release_json(**{first.name: {"digest": None}}))
    assert "no SHA-256" in next(r for r in rows if not r["ok"])["problem"]
    rows = checker.check_build_pins({"assets": []})
    assert not any(r["ok"] for r in rows) and "missing" in rows[0]["problem"]
    m = catalog.MODELS[0]
    row = checker.check_model_pin(m, tree_json(m.repo, m.revision, **{m.file: {"lfs": {"oid": "f" * 64,
                                                                                         "size": m.size}}}))
    assert not row["ok"] and "SHA-256" in row["problem"]
    assert "missing" in checker.check_model_pin(m, [])["problem"]


def _client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_verify_pins_end_to_end(checker):
    gated = catalog.MODELS[-1]
    seen = []

    def handler(request: httpx.Request):
        seen.append((request.method, request.url.host, request.url.path))
        if request.url.host == "api.github.com":
            return httpx.Response(200, json=release_json())
        if request.method == "HEAD":
            return httpx.Response(401 if gated.file in request.url.path else 302)
        _, _, _, owner, name, _, revision = request.url.path.split("/", 6)
        return httpx.Response(200, json=tree_json(f"{owner}/{name}", revision))

    with _client(handler) as c:
        rows = checker.verify_pins(c)
    bad = [r for r in rows if not r["ok"]]
    assert [r["name"] for r in bad] == [f"{gated.repo}/{gated.file}"] and "401" in bad[0]["problem"]
    assert len(rows) == sum(len(b.assets) for b in catalog.SERVER_BUILDS.values()) + len(catalog.MODELS)
    trees = [p for m, h, p in seen if h == "huggingface.co" and m == "GET"]
    assert len(trees) == len(set(trees)) == len({(m.repo, m.revision) for m in catalog.MODELS})
    report = "\n".join(checker.pins_report(rows))
    assert "MISMATCH" in report and "OK" in report


def test_verify_pins_reports_lookup_failures(checker):
    with _client(lambda request: httpx.Response(403, text="Forbidden\nmore")) as c:
        rows = checker.verify_pins(c)
    assert rows and not any(r["ok"] for r in rows)
    assert all("lookup failed" in r["problem"] and "\n" not in r["problem"] for r in rows)
