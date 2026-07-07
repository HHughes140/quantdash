from .signals import SIGNAL_PRESETS, evaluate_signal
from .backtest import BacktestConfig, BacktestResult, run_backtest
from .metrics import compute_metrics
from .factors import factor_regression, rolling_factor_betas
from .attribution import (
    best_worst_windows,
    factor_contribution,
    position_contribution,
    universe_performance,
)

__all__ = [
    "SIGNAL_PRESETS", "evaluate_signal",
    "BacktestConfig", "BacktestResult", "run_backtest",
    "compute_metrics", "factor_regression", "rolling_factor_betas",
    "position_contribution", "universe_performance", "factor_contribution",
    "best_worst_windows",
]
