"""The Solana desk is PAPER ONLY. These tests scan its code, its API route and its screen for anything that could
hold a wallet's keys or build, sign or send a transaction, check that it only calls read-only addresses, and check
that no Solana signing library is installed. Each scanner first proves itself by catching planted examples."""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCANNED = [*sorted((ROOT / "newstrader" / "solana").glob("*.py")), ROOT / "newstrader" / "api" / "routes" / "solana.py",
           ROOT / "newstrader" / "web" / "js" / "solana.js"]
FORBIDDEN = {
    "sends a transaction": r"send_?(raw_?)?transaction",
    "signs a transaction": r"sign_?(all_?)?transactions?|partial_?sign|\.sign\(",
    "a keypair": r"key_?pair",
    "a private key": r"private[\s_-]?key",
    "a secret key": r"secret[\s_-]?key",
    "a seed phrase": r"seed[\s_-]?phrase|mnemonic|bip39",
    "a Solana signing library": (r"\bsolders\b|anchorpy|\bnacl\b|tweetnacl|@solana/web3|"
                                 r"\bsolana\.(keypair|transaction|rpc\.(async_)?api)\b"),
    "Jupiter's transaction builder (/swap)": r"/swap(?![/\w-])",  # /swap/v1/quote is a price quote; /swap" builds one
    "Jupiter's swap instructions": r"swap[-_]instructions",
    "Jupiter Ultra (order / execute)": r"ultra/v\d|/execute\b",
    "an order for a wallet (taker)": r"\btaker\b",
    "wallet bytes": r"from_bytes|from_secret|b58decode",
    "a transaction RPC call": r"\b(send|simulate)[A-Z]\w*|requestAirdrop",
}
BAD_IMPORTS = {"solders", "solana", "nacl", "base58", "anchorpy", "ed25519", "ecdsa", "cryptography", "Crypto",
               "bip_utils", "mnemonic", "spl", "solathon"}
BAD_PACKAGES = {"solana", "solders", "anchorpy", "pynacl", "base58", "solana-py", "spl-token", "ed25519", "bip-utils",
                "mnemonic", "solathon", "jupiter-python-sdk", "solana-agent-kit", "driftpy", "pyserum"}
ALLOWED_URLS = {
    "https://api.geckoterminal.com/api/v2/networks/solana",
    "https://lite-api.jup.ag/swap/v1/quote",
    "https://lite-api.jup.ag/price/v3",
    "https://lite-api.jup.ag/tokens/v2/search",
    "https://api.rugcheck.xyz/v1/tokens/{}/report",
    "https://api.mainnet-beta.solana.com",
}
ALLOWED_RPC = {"getAccountInfo", "getTokenLargestAccounts"}


def findings(text: str) -> list[str]:
    return [name for name, pattern in FORBIDDEN.items() if re.search(pattern, text, re.IGNORECASE if name not in (
        "a transaction RPC call",) else 0)]


def bad_imports(source: str) -> list[str]:
    out = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            out += [a.name for a in node.names if a.name.split(".")[0] in BAD_IMPORTS]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            if node.module.split(".")[0] in BAD_IMPORTS:
                out.append(node.module)
    return out


def bad_requirements(text: str) -> list[str]:
    out = []
    for line in text.splitlines():
        name = re.split(r"[\s\[<>=!~;@]", line.split("#", 1)[0].strip(), maxsplit=1)[0].lower().replace("_", "-")
        if name in BAD_PACKAGES:
            out.append(name)
    return out


# ---------------------------------------------------------------------------------------------- the scanners prove themselves
@pytest.mark.parametrize("planted,what", [
    ('await connection.sendTransaction(tx)', "sends a transaction"),
    ("client.send_raw_transaction(raw)", "sends a transaction"),
    ("wallet.signTransaction(tx)", "signs a transaction"),
    ("tx.partial_sign(kp)", "signs a transaction"),
    ("kp = Keypair.from_base58_string(x)", "a keypair"),
    ("PRIVATE_KEY = os.environ['X']", "a private key"),
    ("secret_key = load()", "a secret key"),
    ("Type your seed phrase here", "a seed phrase"),
    ("words = mnemonic.generate()", "a seed phrase"),
    ("from solders.keypair import Keypair", "a Solana signing library"),
    ("import nacl.signing", "a Solana signing library"),
    ("from solana.rpc.api import Client", "a Solana signing library"),
    ('r = client.post(f"{JUPITER}/swap", json=body)', "Jupiter's transaction builder (/swap)"),
    ('"https://lite-api.jup.ag/swap/v1/swap"', "Jupiter's transaction builder (/swap)"),
    ("https://lite-api.jup.ag/swap/v1/swap-instructions", "Jupiter's swap instructions"),
    ("https://lite-api.jup.ag/ultra/v1/execute", "Jupiter Ultra (order / execute)"),
    ('get("https://lite-api.jup.ag/ultra/v1/order", params={"taker": me})', "an order for a wallet (taker)"),
    ("Keypair.from_bytes(b)", "wallet bytes"),
    ('{"method": "sendTransaction"}', "a transaction RPC call"),
    ('{"method": "simulateTransaction"}', "a transaction RPC call"),
])
def test_the_scanner_catches_planted_examples(planted, what):
    assert what in findings(planted)


def test_the_scanner_leaves_read_only_code_alone():
    clean = ('JUPITER_QUOTE = "https://lite-api.jup.ag/swap/v1/quote"\n'
             'body = {"method": "getAccountInfo", "params": [mint, {"encoding": "jsonParsed"}]}\n'
             "# a pretend wallet of SOL: paper only, no keys\n"
             "self.data.rpc_url = settings.solana.rpc_url\n")
    assert findings(clean) == []


def test_the_import_and_requirement_checks_prove_themselves():
    assert bad_imports("import solders\nfrom solana.rpc.api import Client\nfrom nacl.signing import SigningKey\n"
                       "import base58") == ["solders", "solana.rpc.api", "nacl.signing", "base58"]
    assert bad_imports("from . import gates\nfrom ..db import iso\nimport httpx\nfrom newstrader.solana import data") == []
    planted = "httpx>=0.27\nsolders==0.21.0\nsolana >=0.30 ; python_version>'3.8'\nPyNaCl[x]\n# solders in a comment\n"
    assert bad_requirements(planted) == ["solders", "solana", "pynacl"]


# ---------------------------------------------------------------------------------------------- the real files
def test_the_desk_files_exist():
    names = {p.name for p in SCANNED}
    assert {"data.py", "gates.py", "desk.py", "service.py", "solana.py", "solana.js"} <= names
    assert all(p.exists() for p in SCANNED)


@pytest.mark.parametrize("path", SCANNED, ids=lambda p: p.name)
def test_no_wallet_or_transaction_code(path):
    text = path.read_text(encoding="utf-8")
    assert findings(text) == [], f"{path.name} contains something only a real-money desk would need"
    if path.suffix == ".py":
        assert bad_imports(text) == []


def test_only_read_only_addresses_are_called():
    urls = set()
    rpc = set()
    for path in SCANNED:
        text = path.read_text(encoding="utf-8")
        urls |= set(re.findall(r"https://[^\s\"'<>)]+", text))
        if path.suffix == ".py":
            rpc |= set(re.findall(r"\"(get[A-Z]\w+|send\w+|simulate\w+|request\w+)\"", text))
    assert urls <= ALLOWED_URLS, urls - ALLOWED_URLS
    assert rpc <= ALLOWED_RPC, rpc - ALLOWED_RPC


def test_no_solana_signing_library_is_installed():
    for name in ("requirements.txt", "requirements-dev.txt", "requirements-mac-gpu.txt"):
        path = ROOT / name
        if path.exists():
            assert bad_requirements(path.read_text(encoding="utf-8")) == [], name
    for package in ("solders", "solana"):
        try:
            __import__(package)
        except ImportError:
            continue
        pytest.fail(f"{package} is installed - the paper desk must never be able to sign anything")
