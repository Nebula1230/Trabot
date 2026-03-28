"""
LLM Sentiment Agent — production-ready bridge to the TauricResearch/TradingAgents
LLM pipeline (https://github.com/TauricResearch/TradingAgents).

Wraps ``TradingAgentsGraph.propagate()`` and exposes it as a standard v2
``BaseAgent``.  The LLM pipeline is synchronous and latency-bound (multiple
LLM calls per cycle), so the following production safeguards are applied:

  1. **Thread pool dispatch** — ``propagate()`` runs in a background
     ``ThreadPoolExecutor`` so the async event loop is never blocked.
  2. **Async timeout** — ``asyncio.wait_for`` caps the entire call tree
     (including retries) to ``timeout_seconds``.  On expiry the event loop
     unblocks immediately; the background thread completes or is orphaned when
     the process exits (registered ``atexit`` shutdown is graceful).
  3. **Retry with exponential back-off** — transient errors (network glitches,
     rate-limit 429s) trigger up to ``max_retries`` additional attempts with
     5 s / 10 s / 20 s waits.
  4. **Circuit breaker** — after ``cb_fail_threshold`` consecutive failures the
     breaker opens, blocking further calls for ``cb_cooldown_minutes``.  It
     transitions to HALF-OPEN automatically and closes again on the first
     successful probe.  Call ``LLMSentimentAgent.reset()`` to force-close.
  5. **Per-symbol throttle cache** — the full pipeline runs at most once per
     ``throttle_hours`` per symbol; stale entries are served until they expire.
  6. **Graceful degradation** — any unrecoverable error returns a neutral
     ``AgentOutput(dir_score=0, conf=0)`` so the TA tier operates normally.

Configuration (all under ``llm_agents:`` in config.yaml / TradingConfig):

  enabled:               false
  upstream_path:         /path/to/TauricResearch/TradingAgents  # required
  analysts:              [market, news]  # market|social|news|fundamentals
  throttle_hours:        4.0
  timeout_seconds:       120     # wall-clock cap per call including retries
  max_retries:           2       # retry attempts after first failure
  cb_fail_threshold:     3       # consecutive failures before opening CB
  cb_cooldown_minutes:   15.0    # CB cooldown before half-open probe
  llm_provider:          openai  # openai|anthropic|google|ollama|lmstudio|openrouter|xai
  deep_think_llm:        gpt-4o-mini
  quick_think_llm:       gpt-4o-mini
  backend_url:           ""      # leave blank to use provider default
  weight:                1.0

Score mapping (follows upstream signal_processing output):
  BUY         → +0.70
  OVERWEIGHT  → +0.50
  HOLD        →  0.00
  UNDERWEIGHT → -0.50
  SELL        → -0.70
"""

from __future__ import annotations

import asyncio
import atexit
import logging
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from typing import Any, Dict, List, Optional, Tuple

from ..core.agent_base import BaseAgent
from ..core.types import AgentOutput, TechnicalFeatures, Timeframe

logger = logging.getLogger("LLMSentimentAgent")

# ── Symbol translation ────────────────────────────────────────────────────────
# The upstream TradingAgentsGraph uses yfinance / Finnhub internally.
# Those require different ticker formats than the MT5 broker symbols used by v2.
# This table covers every instrument in the default config.demo.yaml symbol list.

# Forex: any 6-char alpha symbol (or with common broker suffixes) → append "=X"
# Indices: fixed upstream-to-yfinance mapping
# Commodities/Futures: futures codes
# Stocks: strip exchange prefix (NASDAQ:NVDA → NVDA)

_INDEX_MAP: Dict[str, str] = {
    # Equity indices
    "US30":    "^DJI",    # Dow Jones
    "US500":   "^GSPC",   # S&P 500
    "USTEC":   "^NDX",    # Nasdaq 100
    "NAS100":  "^NDX",
    "DJ30":    "^DJI",
    "SP500":   "^GSPC",
    "DAX":     "^GDAXI",  # Germany 40
    "GER40":   "^GDAXI",
    "GER30":   "^GDAXI",
    "UK100":   "^FTSE",   # FTSE 100
    "FTSE":    "^FTSE",
    "FRA40":   "^FCHI",   # CAC 40
    "CAC40":   "^FCHI",
    "JP225":   "^N225",   # Nikkei 225
    "AUS200":  "^AXJO",   # ASX 200
    # Commodities / Futures
    "XAUUSD":  "GC=F",    # Gold
    "GOLD":    "GC=F",
    "XAGUSD":  "SI=F",    # Silver
    "SILVER":  "SI=F",
    "USOIL":   "CL=F",    # WTI Crude
    "XTIUSD":  "CL=F",
    "UKOIL":   "BZ=F",    # Brent Crude
    "NGAS":    "NG=F",    # Natural Gas
    "XPTUSD":  "PL=F",    # Platinum
    "XPDUSD":  "PA=F",    # Palladium
    # Crypto (broad proxy ETFs — yfinance has BTC-USD directly)
    "BTCUSD":  "BTC-USD",
    "ETHUSD":  "ETH-USD",
}

# Analyst modules that only make sense for equities / companies
_EQUITY_ONLY_ANALYSTS = frozenset({"fundamentals", "social"})

# Pattern: a "standard forex pair" is exactly 6 alpha chars, no digits
_FOREX_RE = re.compile(r'^[A-Za-z]{6}$')


def _to_yfinance_ticker(broker_symbol: str) -> Tuple[str, str]:
    """
    Translate a v2 broker symbol to a yfinance-compatible ticker.

    Returns (yf_ticker, instrument_type) where instrument_type is one of:
      "stock", "forex", "index", "commodity", "crypto"

    Examples:
        EURUSD     → ("EURUSD=X",   "forex")
        EURUSDm    → ("EURUSD=X",   "forex")
        GBPJPY     → ("GBPJPY=X",   "forex")
        US30       → ("^DJI",       "index")
        XAUUSD     → ("GC=F",       "commodity")
        NASDAQ:NVDA→ ("NVDA",       "stock")
        NVDA       → ("NVDA",       "stock")
        BTCUSD     → ("BTC-USD",    "crypto")
    """
    sym = broker_symbol.strip().upper()

    # 1. Exchange-prefixed stock: "NASDAQ:NVDA" / "NYSE:AAPL" → strip prefix
    if ":" in sym:
        return sym.split(":", 1)[1], "stock"

    # 2. Known index / commodity
    if sym in _INDEX_MAP:
        itype = "commodity" if ("=F" in _INDEX_MAP[sym] or "-USD" in _INDEX_MAP[sym]) else "index"
        if "-USD" in _INDEX_MAP[sym]:
            itype = "crypto"
        return _INDEX_MAP[sym], itype

    # 3. Forex: 6-char alpha (e.g. EURUSD) or with broker suffixes (EURUSDm, EURUSD.)
    bare = sym.rstrip(".")
    # Strip single trailing alpha suffix if base is exactly 6 chars (e.g. EURUSDm → EURUSD)
    if len(bare) == 7 and bare[:6].isalpha() and bare[6].isalpha():
        bare = bare[:6]
    if _FOREX_RE.match(bare):
        return f"{bare}=X", "forex"

    # 4. Fallback: pass through as-is (bare stock ticker like AAPL, MSFT)
    return sym, "stock"


def _filter_analysts_for_instrument(
    analysts: List[str], instrument_type: str, symbol: str
) -> List[str]:
    """
    Remove analyst modules that would fail or produce meaningless output
    for non-equity instruments (forex, index, commodity, crypto).
    Logs a warning so the user knows which analysts were dropped.
    """
    if instrument_type == "stock":
        return analysts

    filtered = [a for a in analysts if a not in _EQUITY_ONLY_ANALYSTS]
    dropped   = [a for a in analysts if a in _EQUITY_ONLY_ANALYSTS]
    if dropped:
        logger.warning(
            f"[LLM] {symbol} ({instrument_type}): dropping analysts "
            f"{dropped} — they only work for stocks/equities. "
            f"Using: {filtered}"
        )
    return filtered if filtered else ["market"]


# ── Constants ─────────────────────────────────────────────────────────────────

# Full signal vocabulary from upstream signal_processing.py
_SCORE_MAP: Dict[str, float] = {
    "BUY":         +0.70,
    "OVERWEIGHT":  +0.50,
    "HOLD":         0.00,
    "UNDERWEIGHT": -0.50,
    "SELL":        -0.70,
}

# Confidence by signal strength (used for fresh calls; cached calls use stored conf)
_CONF_MAP: Dict[str, float] = {
    "BUY":         0.65,
    "OVERWEIGHT":  0.50,
    "HOLD":        0.30,
    "UNDERWEIGHT": 0.50,
    "SELL":        0.65,
}

# Keys owned by v2 that must NOT be forwarded to TradingAgentsGraph config
_V2_INTERNAL_KEYS = frozenset({
    "upstream_path", "weight", "throttle_hours",
    "timeout_seconds", "max_retries",
    "cb_fail_threshold", "cb_cooldown_minutes",
})


# ── Circuit breaker ───────────────────────────────────────────────────────────

class _CircuitBreaker:
    """
    Three-state circuit breaker: CLOSED → OPEN → HALF_OPEN → CLOSED.

    Thread-safe.  All state transitions are logged at INFO / WARNING level.
    """

    CLOSED    = "CLOSED"    # normal: requests pass through
    OPEN      = "OPEN"      # tripped: requests are rejected
    HALF_OPEN = "HALF_OPEN" # cooldown elapsed: one probe request allowed

    def __init__(self, fail_threshold: int, cooldown_secs: float) -> None:
        self._lock          = threading.Lock()
        self._state         = self.CLOSED
        self._failures      = 0
        self._opened_at     = 0.0
        self._fail_threshold = fail_threshold
        self._cooldown      = cooldown_secs

    @property
    def state(self) -> str:
        with self._lock:
            if self._state == self.OPEN:
                if time.monotonic() - self._opened_at >= self._cooldown:
                    self._state = self.HALF_OPEN
                    logger.info("CircuitBreaker → HALF_OPEN (probe allowed)")
            return self._state

    def allow_request(self) -> bool:
        return self.state in (self.CLOSED, self.HALF_OPEN)

    def record_success(self) -> None:
        with self._lock:
            if self._state != self.CLOSED:
                logger.info("CircuitBreaker → CLOSED (recovered)")
            self._state    = self.CLOSED
            self._failures = 0

    def record_failure(self) -> None:
        with self._lock:
            self._failures += 1
            if self._state == self.HALF_OPEN or self._failures >= self._fail_threshold:
                self._state     = self.OPEN
                self._opened_at = time.monotonic()
                cooldown_min    = self._cooldown / 60.0
                logger.warning(
                    f"CircuitBreaker → OPEN after {self._failures} consecutive "
                    f"failure(s). Cooldown: {cooldown_min:.1f} min."
                )

    def reset(self) -> None:
        with self._lock:
            self._state    = self.CLOSED
            self._failures = 0
            self._opened_at = 0.0
        logger.info("CircuitBreaker manually reset → CLOSED")


# ── Module-level singleton state ──────────────────────────────────────────────
# Stored in a plain object (not on the Pydantic class) so it survives model
# re-validation and is shared across all LLMSentimentAgent instances.

class _SharedState:
    """All shared mutable state for LLMSentimentAgent."""

    def __init__(self) -> None:
        # Protects graph init and cache writes
        self.lock      = threading.RLock()
        # Lazily-initialised TradingAgentsGraph (or None before first call)
        self.graph: Optional[Any] = None
        # Circuit breaker — defaults overridden by _ensure_graph on first use
        self.cb = _CircuitBreaker(fail_threshold=3, cooldown_secs=15 * 60.0)
        # Throttle cache: symbol → (monotonic_ts, dir_score, conf, rationale)
        self.cache: Dict[str, Tuple[float, float, float, str]] = {}
        # Two workers: one active call + one queued; LLM calls are I/O-bound
        self.executor  = ThreadPoolExecutor(max_workers=2, thread_name_prefix="llm_agent")

    def shutdown(self) -> None:
        self.executor.shutdown(wait=False)
        logger.info("LLMSentimentAgent executor shut down")


_shared = _SharedState()
atexit.register(_shared.shutdown)


# ── Agent class ───────────────────────────────────────────────────────────────

class LLMSentimentAgent(BaseAgent):
    """
    Optional macro-overlay agent backed by the TauricResearch/TradingAgents
    LLM pipeline.

    Acts on ``Timeframe.LONG`` — biases directional score before TA-tier fusion.
    Only instantiated when ``llm_agents.enabled: true``.  All failures are
    caught and result in a neutral output so the TA pipeline proceeds normally.
    """

    name:            str       = "LLMSentimentAgent"
    timeframe:       Timeframe = Timeframe.LONG

    # ── Pydantic fields (populated from TradingConfig via runner._build_registry) ──
    analysts:        List[str] = ["market", "news"]
    throttle_hours:  float     = 4.0
    timeout_seconds: int       = 120
    max_retries:     int       = 2
    llm_config:      dict      = {}

    model_config = {"arbitrary_types_allowed": True}

    # ── Required features ─────────────────────────────────────────────────────

    def get_required_features(self) -> list:
        return []  # LLM agent does not consume TA features

    # ── Graph lifecycle (thread-safe lazy init) ───────────────────────────────

    @classmethod
    def _ensure_graph(cls, llm_config: dict, analysts: list) -> bool:
        """
        Lazily initialise TradingAgentsGraph.

        Thread-safe via double-checked locking.  Updates the module-level
        circuit breaker thresholds from ``llm_config`` on first call so they
        reflect user YAML settings.

        Returns True when the graph is ready for use.
        """
        # Fast path — graph already initialised
        if _shared.graph is not None:
            return True

        with _shared.lock:
            # Double check under lock
            if _shared.graph is not None:
                return True

            # Sync CB thresholds from current config (done once)
            cb_fail     = int(llm_config.get("cb_fail_threshold", 3))
            cb_cooldown = float(llm_config.get("cb_cooldown_minutes", 15.0)) * 60.0
            _shared.cb._fail_threshold = cb_fail
            _shared.cb._cooldown       = cb_cooldown

            if not _shared.cb.allow_request():
                remaining = max(0.0, _shared.cb._cooldown - (time.monotonic() - _shared.cb._opened_at))
                logger.debug(
                    f"CircuitBreaker OPEN — skipping graph init "
                    f"(retry in {remaining/60:.1f} min)"
                )
                return False

            try:
                upstream = llm_config.get("upstream_path", "").strip()
                if upstream:
                    # Flush any stale shadowing local tradingagents/ package
                    for key in list(sys.modules.keys()):
                        if key == "tradingagents" or key.startswith("tradingagents."):
                            del sys.modules[key]
                    if upstream not in sys.path:
                        sys.path.insert(0, upstream)
                    logger.debug(f"Prepended upstream path: {upstream}")

                from tradingagents.graph.trading_graph import TradingAgentsGraph  # type: ignore
                from tradingagents.default_config import DEFAULT_CONFIG            # type: ignore

                # Build config: upstream DEFAULT_CONFIG baseline + user overrides.
                # Empty strings and v2-internal keys are excluded so provider
                # defaults (base_url, api_key lookup, etc.) remain intact.
                override = {
                    k: v for k, v in llm_config.items()
                    if k not in _V2_INTERNAL_KEYS and v != ""
                }
                cfg = {**DEFAULT_CONFIG, **override}

                _shared.graph = TradingAgentsGraph(
                    selected_analysts=list(analysts),
                    debug=False,
                    config=cfg,
                )
                _shared.cb.record_success()
                logger.info(
                    "TradingAgentsGraph ready — "
                    f"analysts={analysts}, provider={cfg.get('llm_provider')}, "
                    f"deep={cfg.get('deep_think_llm')}, quick={cfg.get('quick_think_llm')}"
                )
                return True

            except ImportError as exc:
                logger.error(
                    f"tradingagents package not importable ({exc}). "
                    "Set llm_agents.upstream_path to the cloned "
                    "TauricResearch/TradingAgents directory."
                )
                _shared.cb.record_failure()
                return False

            except Exception as exc:
                logger.error(
                    f"TradingAgentsGraph init failed: {exc}",
                    exc_info=True,
                )
                _shared.cb.record_failure()
                return False

    # ── Throttle cache helpers ────────────────────────────────────────────────

    @staticmethod
    def _get_cached(
        symbol: str, throttle_secs: float
    ) -> Optional[Tuple[float, float, str]]:
        """Return (dir_score, conf, rationale) if cache is fresh, else None."""
        entry = _shared.cache.get(symbol)
        if entry is None:
            return None
        ts, score, conf, rationale = entry
        if time.monotonic() - ts < throttle_secs:
            return score, conf, rationale
        return None

    @staticmethod
    def _set_cache(symbol: str, score: float, conf: float, rationale: str) -> None:
        _shared.cache[symbol] = (time.monotonic(), score, conf, rationale)

    # ── Synchronous LLM call with retry (dispatched to thread pool) ───────────

    @classmethod
    def _run_with_retry(
        cls,
        llm_config: dict,
        analysts:   list,
        symbol:     str,
        trade_date: str,
        max_retries: int,
    ) -> Tuple[float, float, str]:
        """
        Initialise graph (if needed) then call ``propagate()`` with
        exponential back-off retry.

        Designed to run inside ``_shared.executor`` — all I/O happens here,
        never on the async event loop.

        Returns (dir_score, conf, rationale).
        """
        # Ensure graph inside the thread (TradingAgentsGraph.__init__ may do I/O)
        if not cls._ensure_graph(llm_config, analysts):
            return 0.0, 0.0, "LLM graph unavailable"

        # ── Translate broker symbol to yfinance-compatible ticker ──────────────
        yf_ticker, instrument_type = _to_yfinance_ticker(symbol)
        effective_analysts = _filter_analysts_for_instrument(
            analysts, instrument_type, symbol
        )
        if yf_ticker != symbol:
            logger.info(
                f"[LLM] Symbol translation: {symbol} → {yf_ticker} ({instrument_type})"
            )

        last_exc: Optional[Exception] = None
        for attempt in range(max_retries + 1):
            t0 = time.monotonic()
            try:
                _graph_state, raw_decision = _shared.graph.propagate(yf_ticker, trade_date)
                elapsed  = time.monotonic() - t0
                decision = raw_decision.strip().upper()
                score    = _SCORE_MAP.get(decision, 0.0)
                conf     = _CONF_MAP.get(decision, 0.30)
                rationale = (
                    f"LLM[{decision}] {symbol}({yf_ticker})@{trade_date} "
                    f"({elapsed:.1f}s, attempt={attempt + 1}/{max_retries + 1})"
                )
                logger.info(
                    f"[LLM] {symbol} → {decision}  "
                    f"score={score:+.2f}  conf={conf:.2f}  elapsed={elapsed:.1f}s"
                )
                _shared.cb.record_success()
                return score, conf, rationale

            except Exception as exc:
                elapsed  = time.monotonic() - t0
                last_exc = exc
                _shared.cb.record_failure()

                if attempt < max_retries:
                    backoff = min(5 * (2 ** attempt), 30)   # 5 s, 10 s, 20 s (capped)
                    logger.warning(
                        f"[LLM] {symbol} attempt {attempt + 1}/{max_retries + 1} "
                        f"failed in {elapsed:.1f}s: {exc!r} — retrying in {backoff}s"
                    )
                    time.sleep(backoff)
                else:
                    logger.error(
                        f"[LLM] {symbol} all {max_retries + 1} attempts failed. "
                        f"Last error: {exc!r}"
                    )

        return 0.0, 0.0, f"LLM pipeline error after {max_retries + 1} attempts: {last_exc!r}"

    # ── BaseAgent interface ───────────────────────────────────────────────────

    async def analyze(
        self,
        features: TechnicalFeatures,
        context: Dict[str, Any] = None,
    ) -> AgentOutput:
        """
        Return a directional score from the LLM pipeline (or from cache).

        Never raises — all errors produce a neutral ``AgentOutput``.
        """
        symbol      = (context or {}).get("symbol", "UNKNOWN")
        throttle    = self.throttle_hours * 3600.0
        trade_date  = date.today().isoformat()
        timeout_sec = float(self.timeout_seconds)

        # 1. Circuit breaker gate (cheap check — no I/O)
        if not _shared.cb.allow_request():
            remaining = max(0.0, _shared.cb._cooldown - (time.monotonic() - _shared.cb._opened_at))
            return self._neutral(
                f"LLM circuit breaker OPEN (retry in {remaining / 60:.1f} min)",
                {"cb_state": _shared.cb.state},
            )

        # 2. Return cached result if still within throttle window
        cached = self._get_cached(symbol, throttle)
        if cached is not None:
            score, conf, rationale = cached
            return AgentOutput(
                timeframe=self.timeframe,
                dir_score=score,
                conf=conf,
                rationale=f"[cached] {rationale}",
                evidence={"source": "llm_cache", "symbol": symbol},
            )

        # 3. Dispatch I/O-bound work to thread pool with a hard wall-clock cap
        loop = asyncio.get_running_loop()
        coro = loop.run_in_executor(
            _shared.executor,
            self._run_with_retry,
            self.llm_config,
            self.analysts,
            symbol,
            trade_date,
            self.max_retries,
        )
        try:
            score, conf, rationale = await asyncio.wait_for(coro, timeout=timeout_sec)
        except asyncio.TimeoutError:
            msg = (
                f"LLM call for {symbol} timed out after {timeout_sec:.0f}s "
                "(background thread will finish eventually). "
                "Increase timeout_seconds or reduce the analyst list."
            )
            logger.warning(msg)
            return self._neutral(msg, {"source": "llm_timeout", "symbol": symbol})
        except Exception as exc:
            logger.error(f"[LLM] Unexpected error for {symbol}: {exc}", exc_info=True)
            return self._neutral(f"LLM unexpected error: {exc}", {"source": "llm_error"})

        # 4. Persist in throttle cache
        self._set_cache(symbol, score, conf, rationale)

        return AgentOutput(
            timeframe=self.timeframe,
            dir_score=score,
            conf=conf,
            rationale=rationale,
            evidence={
                "source":   "llm_pipeline",
                "symbol":   symbol,
                "decision": rationale,
            },
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _neutral(self, reason: str, evidence: dict) -> AgentOutput:
        return AgentOutput(
            timeframe=self.timeframe,
            dir_score=0.0,
            conf=0.0,
            rationale=reason,
            evidence={"source": "llm_unavailable", **evidence},
        )

    # ── Observability & lifecycle (module-level operations) ───────────────────

    @classmethod
    def get_health(cls) -> dict:
        """
        Return a health status snapshot (useful for monitoring endpoints).

        Example::

            from tradingagents_v2.agents import LLMSentimentAgent
            print(LLMSentimentAgent.get_health())
        """
        now = time.monotonic()
        cache_summary = {
            sym: {
                "age_minutes": round((now - ts) / 60.0, 1),
                "score":       score,
                "conf":        conf,
            }
            for sym, (ts, score, conf, _) in _shared.cache.items()
        }
        return {
            "graph_ready":       _shared.graph is not None,
            "cb_state":          _shared.cb.state,
            "cb_failures":       _shared.cb._failures,
            "cache_symbols":     list(cache_summary.keys()),
            "cache_detail":      cache_summary,
            "executor_workers":  _shared.executor._max_workers,
        }

    @classmethod
    def reset(cls) -> None:
        """
        Force-reset graph, circuit breaker, and cache.

        Call this after correcting API keys or upstream_path.  The next
        ``analyze()`` invocation will attempt fresh graph initialisation.
        """
        with _shared.lock:
            _shared.graph = None
            _shared.cache.clear()
            _shared.cb.reset()
        logger.info("LLMSentimentAgent fully reset — next call will re-initialise")

    @classmethod
    def evict_cache(cls, symbol: Optional[str] = None) -> None:
        """Evict one symbol from the throttle cache, or all if ``symbol`` is None."""
        with _shared.lock:
            if symbol:
                _shared.cache.pop(symbol, None)
                logger.debug(f"Cache evicted for {symbol}")
            else:
                _shared.cache.clear()
                logger.info("LLM throttle cache fully cleared")
