"""
RL Exit Policy — learned exit decision framework.

This module provides a pluggable exit policy that can be:
  1. A hand-tuned heuristic baseline (default) — uses a weighted score of
     signal strength, profit-R, time-held, and volatility to decide exits.
  2. A trained RL model (future) — drop in a PyTorch/ONNX model that maps
     the same observation vector to an exit probability.

Observation vector (7 features, all normalised to [-1, 1] or [0, 1]):
  0: dir_long       — D1 trend alignment with trade  [-1, 1]
  1: dir_mid        — mid-TF alignment               [-1, 1]
  2: dir_short      — short-TF alignment              [-1, 1]
  3: profit_r       — current P&L in R-multiples      [-3, 5] clipped
  4: hours_held     — time-in-trade / max_hours       [0, 1]
  5: atr_ratio      — current ATR / entry ATR         [0.3, 3.0]
  6: signal_conf    — fusion confidence               [0, 1]

Action space:
  - HOLD   (0): keep position open
  - EXIT   (1): close entire position
  - TIGHTEN(2): tighten SL to breakeven + buffer

The heuristic baseline computes:
  exit_score = w_signal * (1 - aligned_score) + w_time * time_frac + w_profit * max(0, -profit_r)
  If exit_score > threshold → EXIT
  If exit_score > tighten_threshold → TIGHTEN
  Else → HOLD
"""

import logging
from dataclasses import dataclass
from enum import IntEnum
from typing import Optional

import numpy as np


class ExitAction(IntEnum):
    HOLD = 0
    EXIT = 1
    TIGHTEN = 2


@dataclass
class ExitObservation:
    """Observation vector for the exit policy."""
    dir_long: float       # D1 alignment with trade direction [-1, 1]
    dir_mid: float        # mid-TF alignment [-1, 1]
    dir_short: float      # short-TF alignment [-1, 1]
    profit_r: float       # current profit in R-multiples
    hours_held: float     # hours the trade has been open
    max_hours: float      # max allowed hours
    atr_ratio: float      # current ATR / entry ATR
    signal_conf: float    # fusion confidence [0, 1]
    is_counter_trend: bool = False

    def to_array(self) -> np.ndarray:
        """Convert to normalised numpy array for model input."""
        time_frac = self.hours_held / max(self.max_hours, 1.0)
        return np.array([
            np.clip(self.dir_long, -1, 1),
            np.clip(self.dir_mid, -1, 1),
            np.clip(self.dir_short, -1, 1),
            np.clip(self.profit_r, -3, 5) / 5.0,  # normalise to ~[-0.6, 1]
            np.clip(time_frac, 0, 1),
            np.clip(self.atr_ratio, 0.3, 3.0) / 3.0,
            np.clip(self.signal_conf, 0, 1),
        ], dtype=np.float32)


class HeuristicExitPolicy:
    """
    Hand-tuned heuristic exit policy (baseline).

    Weights and thresholds can be tuned via config or replaced entirely
    by a trained RL model inheriting from BaseExitPolicy.
    """

    def __init__(self,
                 w_signal: float = 0.40,
                 w_time: float = 0.25,
                 w_profit: float = 0.35,
                 exit_threshold: float = 0.65,
                 tighten_threshold: float = 0.45):
        self.w_signal = w_signal
        self.w_time = w_time
        self.w_profit = w_profit
        self.exit_threshold = exit_threshold
        self.tighten_threshold = tighten_threshold
        self.logger = logging.getLogger("RLExitPolicy")

    def predict(self, obs: ExitObservation) -> ExitAction:
        """
        Given an observation, return the recommended exit action.

        Returns ExitAction.HOLD, EXIT, or TIGHTEN.
        """
        # Signal alignment: how much the signal supports the current trade
        # Positive = signal agrees with trade, negative = signal opposes
        aligned_score = (obs.dir_long * 0.5 + obs.dir_mid * 0.3 + obs.dir_short * 0.2)
        # Invert: high score when signal DISAGREES = closer to exit
        signal_distress = 1.0 - np.clip(aligned_score, -1, 1)  # [0, 2] → normalise
        signal_distress = signal_distress / 2.0  # [0, 1]

        # Time pressure: increases as trade approaches max duration
        time_frac = obs.hours_held / max(obs.max_hours, 1.0)
        time_pressure = np.clip(time_frac, 0, 1)

        # Profit distress: high when losing, zero when winning
        profit_distress = np.clip(-obs.profit_r, 0, 3) / 3.0  # [0, 1]

        # Volatility adjustment: high vol = more patience, low vol = less
        vol_adjust = 1.0
        if obs.atr_ratio > 1.5:
            vol_adjust = 0.85   # more patience in high vol
        elif obs.atr_ratio < 0.7:
            vol_adjust = 1.15   # less patience in low vol

        # Counter-trend trades get extra exit pressure
        ct_penalty = 0.10 if obs.is_counter_trend else 0.0

        exit_score = (
            self.w_signal * signal_distress
            + self.w_time * time_pressure
            + self.w_profit * profit_distress
            + ct_penalty
        ) * vol_adjust

        self.logger.debug(
            f"RL exit score={exit_score:.3f} "
            f"(signal={signal_distress:.2f} time={time_pressure:.2f} "
            f"profit={profit_distress:.2f} vol_adj={vol_adjust:.2f})"
        )

        if exit_score >= self.exit_threshold:
            return ExitAction.EXIT
        elif exit_score >= self.tighten_threshold:
            return ExitAction.TIGHTEN
        return ExitAction.HOLD


class RLExitPolicy:
    """
    Wrapper that loads either the heuristic baseline or a trained ONNX model.

    To use a trained model:
        1. Train on historical trade data using the ExitObservation format
        2. Export to ONNX
        3. Set `model_path` in config to the ONNX file
        4. The model should output 3 logits: [hold, exit, tighten]
    """

    def __init__(self, config: dict = None):
        config = config or {}
        self.logger = logging.getLogger("RLExitPolicy")
        self._model = None
        self._heuristic = HeuristicExitPolicy(
            w_signal=config.get("w_signal", 0.40),
            w_time=config.get("w_time", 0.25),
            w_profit=config.get("w_profit", 0.35),
            exit_threshold=config.get("exit_threshold", 0.65),
            tighten_threshold=config.get("tighten_threshold", 0.45),
        )
        self.enabled = config.get("enabled", False)

        # Try to load ONNX model if path provided
        model_path = config.get("model_path", "")
        if model_path:
            self._load_onnx_model(model_path)

    def _load_onnx_model(self, path: str):
        """Load a trained ONNX exit model."""
        try:
            import onnxruntime as ort
            self._model = ort.InferenceSession(path)
            self.logger.info(f"Loaded RL exit model from {path}")
        except ImportError:
            self.logger.warning("onnxruntime not installed — using heuristic baseline")
        except Exception as e:
            self.logger.warning(f"Failed to load RL model from {path}: {e} — using heuristic")

    def predict(self, obs: ExitObservation) -> ExitAction:
        """Get exit decision from trained model or heuristic fallback."""
        if not self.enabled:
            return ExitAction.HOLD

        if self._model is not None:
            try:
                arr = obs.to_array().reshape(1, -1)
                input_name = self._model.get_inputs()[0].name
                logits = self._model.run(None, {input_name: arr})[0][0]
                return ExitAction(int(np.argmax(logits)))
            except Exception as e:
                self.logger.debug(f"Model inference failed: {e}")

        return self._heuristic.predict(obs)
