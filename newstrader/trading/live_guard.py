"""The lock on real-money trading.

Rules (enforced here, in one place):
  * The app always starts in PAPER mode. The live flag lives only in memory and is never saved.
  * Switching to live needs ALL of: separate live keys in .env, an explicit enable request, and the
    exact confirmation phrase typed by you.
  * Only `credentials_for_mode` hands out Alpaca credentials, and it gives out live keys only while
    the in-memory live flag is set.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from ..keys import Keys
from ..state import AppState

log = logging.getLogger(__name__)

CONFIRM_PHRASE = "I UNDERSTAND THIS USES REAL MONEY"


class LiveTradingLocked(Exception):
    """Raised when something tries to use live trading without unlocking it properly."""


@dataclass(frozen=True)
class AlpacaCredentials:
    api_key: str
    secret_key: str
    paper: bool


def arm_live_trading(state: AppState, keys: Keys, enable: bool, phrase: str) -> None:
    if enable is not True:
        raise LiveTradingLocked("Live trading was not explicitly enabled.")
    if (phrase or "").strip() != CONFIRM_PHRASE:
        raise LiveTradingLocked(f'Type the phrase exactly: "{CONFIRM_PHRASE}"')
    if not keys.has_alpaca_live:
        raise LiveTradingLocked("No live Alpaca keys. Add ALPACA_LIVE_API_KEY and ALPACA_LIVE_SECRET_KEY first.")
    if keys.alpaca_live_key == keys.alpaca_paper_key:
        raise LiveTradingLocked("Live and paper keys are identical - live keys must come from your live account.")
    state._set_live_armed(True)
    log.warning("LIVE TRADING ENABLED by user confirmation. Real money is now at risk until the app restarts.")


def disarm_live_trading(state: AppState) -> None:
    if state.live_armed:
        log.warning("Live trading disabled - back to PAPER mode.")
    state._set_live_armed(False)


def credentials_for_mode(state: AppState, keys: Keys) -> AlpacaCredentials | None:
    """Alpaca credentials for the current mode, or None if the needed keys are missing."""
    if state.live_armed:
        if not keys.has_alpaca_live:
            raise LiveTradingLocked("Live mode is armed but live keys are missing.")
        return AlpacaCredentials(keys.alpaca_live_key, keys.alpaca_live_secret, paper=False)
    if not keys.has_alpaca_paper:
        return None
    return AlpacaCredentials(keys.alpaca_paper_key, keys.alpaca_paper_secret, paper=True)


def assert_mode_allowed(state: AppState, paper: bool) -> None:
    """Last line of defence before an Alpaca trading client is created."""
    if not paper and not state.live_armed:
        raise LiveTradingLocked("Refusing to create a LIVE Alpaca client: live trading is not armed.")
