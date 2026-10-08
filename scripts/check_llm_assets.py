"""Lists what Pro AI (the optional local language model) would download, with exact sizes and SHA-256 checksums,
so the app can pin them. Run by .github/workflows/check-llm-assets.yml (on demand and weekly).

    python scripts/check_llm_assets.py                  # Markdown report
    python scripts/check_llm_assets.py --json out.json  # also machine-readable

Needs GITHUB_TOKEN for the GitHub API (set automatically in Actions). Standalone on purpose (only needs httpx).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

import httpx

LLAMA_REPO = "ggml-org/llama.cpp"
# the server builds the app can use: Windows NVIDIA (CUDA 13 / 12), Windows any GPU (Vulkan), CPU, Apple Silicon
ASSET_PATTERNS = [r"bin-win-cuda-1[23]\.\d+-x64\.(zip|tar\.gz)$", r"^cudart-.*win-cuda-1[23]\.\d+-x64\.zip$",
                  r"bin-win-vulkan-x64\.(zip|tar\.gz)$", r"bin-win-cpu-x64\.(zip|tar\.gz)$",
                  r"bin-macos-(arm64|x64)\.(zip|tar\.gz)$", r"bin-ubuntu-(x64|vulkan-x64)\.(zip|tar\.gz)$"]
# model searches on Hugging Face (GGUF builds); the report lists the most downloaded repos and their files
MODEL_SEARCHES = ["Qwen3.5-4B", "Qwen3.5-9B", "gemma-4-12b", "gemma-4-E4B", "gemma-4-26B-A4B", "Qwen3.6-35B-A3B",
                  "Qwen3.8-27B", "Qwen3.6-27B", "gpt-oss-20b", "Qwen3.5-0.8B", "Qwen3-Embedding-0.6B"]
QUANTS = re.compile(r"(Q4_K_M|Q5_K_M|Q6_K|Q8_0|Q4_0|IQ4_XS|MXFP4|UD-Q4_K_XL|F16)\.gguf$", re.I)


def github_releases(client: httpx.Client) -> list[dict]:
    headers = {"Accept": "application/vnd.github+json"}
    if os.environ.get("GITHUB_TOKEN"):
        headers["Authorization"] = f"Bearer {os.environ['GITHUB_TOKEN']}"
    r = client.get(f"https://api.github.com/repos/{LLAMA_REPO}/releases", params={"per_page": 15}, headers=headers)
    r.raise_for_status()
    return r.json()


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
    args = ap.parse_args()
    with httpx.Client(timeout=30, follow_redirects=True, headers={"User-Agent": "NewsTrader asset check"}) as client:
        rel_lines, assets = release_report(github_releases(client))
        mod_lines, models = model_report(client)
    print("\n".join(rel_lines + mod_lines))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump({"assets": assets, "models": models}, fh, indent=1)
    return 0 if assets and models else 1


if __name__ == "__main__":
    sys.exit(main())
