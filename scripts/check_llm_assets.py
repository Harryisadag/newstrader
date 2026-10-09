"""Lists what Pro AI (the optional local language model) would download, with exact sizes and SHA-256 checksums,
so the app can pin them, and checks the pins already in newstrader/llm/catalog.py. Run by
.github/workflows/check-llm-assets.yml (on demand, weekly and when catalog.py changes).

    python scripts/check_llm_assets.py                  # Markdown report
    python scripts/check_llm_assets.py --json out.json  # also machine-readable
    python scripts/check_llm_assets.py --verify-pins    # every pinned file still matches? (exit 1 if not)

Needs GITHUB_TOKEN for the GitHub API (set automatically in Actions). Only needs httpx.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from newstrader.llm import catalog  # noqa: E402  (only dataclasses - no app dependencies)

LLAMA_REPO = "ggml-org/llama.cpp"
# the server builds the app can use: Windows NVIDIA (CUDA 13 / 12), Windows any GPU (Vulkan), CPU, Apple Silicon
ASSET_PATTERNS = [r"bin-win-cuda-1[23]\.\d+-x64\.(zip|tar\.gz)$", r"^cudart-.*win-cuda-1[23]\.\d+-x64\.zip$",
                  r"bin-win-vulkan-x64\.(zip|tar\.gz)$", r"bin-win-cpu-x64\.(zip|tar\.gz)$",
                  r"bin-macos-(arm64|x64)\.(zip|tar\.gz)$", r"bin-ubuntu-(x64|vulkan-x64)\.(zip|tar\.gz)$"]
# model searches on Hugging Face (GGUF builds); the report lists the most downloaded repos and their files
MODEL_SEARCHES = ["Qwen3.5-4B", "Qwen3.5-9B", "gemma-4-12b", "gemma-4-E4B", "gemma-4-26B-A4B", "Qwen3.6-35B-A3B",
                  "Qwen3.8-27B", "Qwen3.6-27B", "gpt-oss-20b", "Qwen3.5-0.8B", "Qwen3-Embedding-0.6B"]
QUANTS = re.compile(r"(Q4_K_M|Q5_K_M|Q6_K|Q8_0|Q4_0|IQ4_XS|MXFP4|UD-Q4_K_XL|F16)\.gguf$", re.I)


def _github_headers() -> dict:
    headers = {"Accept": "application/vnd.github+json"}
    if os.environ.get("GITHUB_TOKEN"):
        headers["Authorization"] = f"Bearer {os.environ['GITHUB_TOKEN']}"
    return headers


def github_releases(client: httpx.Client) -> list[dict]:
    r = client.get(f"https://api.github.com/repos/{LLAMA_REPO}/releases", params={"per_page": 15},
                   headers=_github_headers())
    r.raise_for_status()
    return r.json()


# ------------------------------------------------------------------------------------------------ --verify-pins
def check_build_pins(release: dict) -> list[dict]:
    """One row per pinned server file: does the GitHub release still have it with the same size and SHA-256?"""
    published = {a["name"]: a for a in release.get("assets", [])}
    rows = []
    for build in catalog.SERVER_BUILDS.values():
        for a in build.assets:
            got = published.get(a.name)
            if got is None:
                rows.append(_row("server", a.name, a.size, a.sha256, None, None, "missing from the release"))
                continue
            digest = (got.get("digest") or "").removeprefix("sha256:").lower()
            rows.append(_row("server", a.name, a.size, a.sha256, got.get("size"), digest or None))
    return rows


def check_model_pin(model: catalog.ModelChoice, tree: list) -> dict:
    """Does the pinned Hugging Face revision still have the file with the same size and SHA-256 (the LFS oid)?"""
    entry = next((f for f in tree if isinstance(f, dict) and f.get("path") == model.file), None)
    if entry is None:
        return _row("model", f"{model.repo}/{model.file}", model.size, model.sha256, None, None,
                    "missing at the pinned revision")
    lfs = entry.get("lfs") or {}
    return _row("model", f"{model.repo}/{model.file}", model.size, model.sha256,
                lfs.get("size", entry.get("size")), (lfs.get("oid") or "").lower() or None)


def _row(kind: str, name: str, size: int, sha: str, got_size, got_sha, problem: str = "") -> dict:
    if not problem:
        problems = []
        if got_size != size:
            problems.append(f"size {got_size} != pinned {size}")
        if not got_sha:
            problems.append("no SHA-256 published")
        elif got_sha != sha.lower():
            problems.append(f"SHA-256 {got_sha} != pinned {sha}")
        problem = "; ".join(problems)
    return {"kind": kind, "name": name, "size": size, "sha256": sha, "got_size": got_size, "got_sha256": got_sha,
            "ok": not problem, "problem": problem}


def verify_pins(client: httpx.Client) -> list[dict]:
    rows = []
    try:
        r = client.get(f"https://api.github.com/repos/{LLAMA_REPO}/releases/tags/{catalog.LLAMA_CPP_BUILD}",
                       headers=_github_headers())
        r.raise_for_status()
        rows += check_build_pins(r.json())
    except Exception as exc:
        rows.append(_row("server", f"release {catalog.LLAMA_CPP_BUILD}", 0, "", None, None, _failed(exc)))
    trees: dict[tuple[str, str], list] = {}
    for m in catalog.MODELS:
        key = (m.repo, m.revision)
        try:
            if key not in trees:
                r = client.get(f"https://huggingface.co/api/models/{m.repo}/tree/{m.revision}")
                r.raise_for_status()
                trees[key] = r.json()
            row = check_model_pin(m, trees[key])
            if row["ok"]:  # the app downloads without a Hugging Face login, so a gated repo won't work
                head = client.head(catalog.hf_url(m), follow_redirects=False)
                if head.status_code not in (200, 301, 302, 303, 307, 308):
                    row.update(ok=False, problem=f"download answers HTTP {head.status_code} (gated or removed?)")
            rows.append(row)
        except Exception as exc:
            rows.append(_row("model", f"{m.repo}/{m.file}", m.size, m.sha256, None, None, _failed(exc)))
    return rows


def _failed(exc: Exception) -> str:
    return "lookup failed: " + (str(exc).splitlines() or [type(exc).__name__])[0]


def pins_report(rows: list[dict]) -> list[str]:
    bad = [r for r in rows if not r["ok"]]
    lines = [f"## Pinned Pro AI downloads (llama.cpp {catalog.LLAMA_CPP_BUILD})", "",
             f"{len(rows) - len(bad)} of {len(rows)} match." if bad else f"All {len(rows)} pinned files match.", "",
             "| | File | Size | Problem |", "|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {'OK' if r['ok'] else 'MISMATCH'} | {r['name']} | {r['size']:,} | {r['problem'] or '-'} |")
    return lines + [""]


def release_report(releases: list[dict]) -> tuple[list[str], list[dict]]:
    lines, out = ["## llama.cpp server builds", ""], []
    if releases:
        lines += ["All assets of the newest build: " + ", ".join(a["name"] for a in releases[0].get("assets", [])), ""]
    for rel in releases[:3]:
        lines.append(f"### {rel['tag_name']} ({'pre-release' if rel.get('prerelease') else 'release'}, "
                     f"{rel.get('published_at')})")
        lines += ["", "| Asset | Size | SHA-256 | URL |", "|---|---|---|---|"]
        for a in rel.get("assets", []):
            if any(re.search(p, a["name"]) for p in ASSET_PATTERNS):
                digest = (a.get("digest") or "").removeprefix("sha256:")
                lines.append(f"| {a['name']} | {a['size']:,} | {digest or '-'} | {a['browser_download_url']} |")
                out.append({"tag": rel["tag_name"], "prerelease": rel.get("prerelease"), "name": a["name"],
                            "size": a["size"], "sha256": digest, "url": a["browser_download_url"]})
        lines.append("")
    return lines, out


def model_report(client: httpx.Client) -> tuple[list[str], list[dict]]:
    lines, out = ["## GGUF model files on Hugging Face", ""], []
    for query in MODEL_SEARCHES:
        try:
            r = client.get("https://huggingface.co/api/models",
                           params={"search": query, "filter": "gguf", "sort": "downloads", "limit": 6})
            r.raise_for_status()
            repos = r.json()
        except Exception as exc:
            lines += [f"### {query}: search failed ({exc})", ""]
            continue
        lines += [f"### {query}", ""]
        for repo in repos:
            rid = repo["id"]
            try:
                info = client.get(f"https://huggingface.co/api/models/{rid}", params={"blobs": "true"}).json()
                tree = client.get(f"https://huggingface.co/api/models/{rid}/tree/main").json()
            except Exception as exc:
                lines.append(f"- {rid}: failed ({exc})")
                continue
            card = info.get("cardData") or {}
            lines.append(f"- **{rid}** - downloads {repo.get('downloads')}, gated: {info.get('gated')}, "
                         f"license: {card.get('license')}, base: {card.get('base_model')}, sha: {info.get('sha')}")
            for f in tree if isinstance(tree, list) else []:
                path = f.get("path", "")
                if f.get("type") == "file" and QUANTS.search(path):
                    lfs = f.get("lfs") or {}
                    lines.append(f"  - `{path}` {f.get('size', 0):,} B sha256 {lfs.get('oid', '-')}")
                    out.append({"search": query, "repo": rid, "revision": info.get("sha"), "file": path,
                                "size": f.get("size"), "sha256": lfs.get("oid"), "gated": info.get("gated"),
                                "license": card.get("license"), "downloads": repo.get("downloads")})
        lines.append("")
    return lines, out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    ap.add_argument("--verify-pins", action="store_true", help="check catalog.py's pins; exit 1 on any mismatch")
    args = ap.parse_args()
    with httpx.Client(timeout=30, follow_redirects=True, headers={"User-Agent": "NewsTrader asset check"}) as client:
        if args.verify_pins:
            rows = verify_pins(client)
            print("\n".join(pins_report(rows)))
            if args.json:
                with open(args.json, "w", encoding="utf-8") as fh:
                    json.dump({"pins": rows}, fh, indent=1)
            return 0 if rows and all(r["ok"] for r in rows) else 1
        rel_lines, assets = release_report(github_releases(client))
        mod_lines, models = model_report(client)
    print("\n".join(rel_lines + mod_lines))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump({"assets": assets, "models": models}, fh, indent=1)
    return 0 if assets and models else 1


if __name__ == "__main__":
    sys.exit(main())
