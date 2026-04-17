"""
Evaluation metrics for MSF-Net experiments.

Six metrics used in Tables 3, 4, 5:
  - MAE    (Mean Absolute Error)
  - RMSE   (Root Mean Squared Error)
  - MASE   (Mean Absolute Scaled Error) with seasonal naive baseline
  - sMAPE  (Symmetric Mean Absolute Percentage Error)
  - Winkler Score   (probabilistic calibration)
  - PCM    (Perishable Cost Metric — primary business-facing metric)
"""

from typing import Dict, Optional
import numpy as np


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute error."""
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root mean squared error."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mase(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_insample: np.ndarray,
    seasonal_period: int = 7,
) -> float:
    """
    Mean Absolute Scaled Error [Hyndman & Koehler, 2006].

    Scaled by the in-sample mean absolute seasonal naive error.
    MASE < 1.0 indicates the model beats the seasonal naive benchmark.
    """
    numerator = np.mean(np.abs(y_true - y_pred))
    seasonal_errors = np.abs(y_insample[seasonal_period:] - y_insample[:-seasonal_period])
    denominator = np.mean(seasonal_errors)
    if denominator == 0:
        return float("nan")
    return float(numerator / denominator)


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Symmetric Mean Absolute Percentage Error. Bounded in [0, 200].
    """
    denom = np.abs(y_true) + np.abs(y_pred)
    mask = denom > 0
    if not mask.any():
        return 0.0
    return float(200.0 * np.mean(np.abs(y_true[mask] - y_pred[mask]) / denom[mask]))


def winkler_score(
    y_true: np.ndarray,
    lower: np.ndarray,    # e.g. τ = 0.10 quantile forecast
    upper: np.ndarray,    # e.g. τ = 0.90 quantile forecast
    alpha: float = 0.20,  # Central (1 − α) prediction interval
) -> float:
    """
    Winkler interval score [Winkler, 1972].

    Jointly rewards sharpness (narrow intervals) and reliability (correct coverage).
    """
    width = (upper - lower).astype(float)
    below = y_true < lower
    above = y_true > upper
    score = width.copy()
    score[below] += (2.0 / alpha) * (lower[below] - y_true[below])
    score[above] += (2.0 / alpha) * (y_true[above] - upper[above])
    return float(np.mean(score))


def perishable_cost_metric(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    alpha_over: float = 1.47,
    alpha_under: float = 1.0,
) -> float:
    """
    Perishable Cost Metric (PCM).

    Translates forecast errors into an asymmetric inventory cost aligned with
    the operational economics of perishable retailing.

    Over-forecast errors (ŷ > y) incur α_over per unit — perishable write-off cost.
    Under-forecast errors (ŷ ≤ y) incur α_under per unit — stockout opportunity cost.
    """
    over_err = np.maximum(y_pred - y_true, 0.0)
    under_err = np.maximum(y_true - y_pred, 0.0)
    return float(np.mean(alpha_over * over_err + alpha_under * under_err))


def diebold_mariano_test(
    errors_a: np.ndarray,
    errors_b: np.ndarray,
    h: int = 1,
) -> Dict[str, float]:
    """
    Diebold-Mariano test [Diebold & Mariano, 1995] for comparing predictive accuracy.

    Null hypothesis: two forecasts have equal predictive accuracy.
    Returns the DM test statistic and its two-sided p-value.

    Args:
        errors_a: forecast errors of model A (y - ŷ_a)
        errors_b: forecast errors of model B (y - ŷ_b)
        h: forecast horizon
    """
    from scipy import stats

    d = errors_a ** 2 - errors_b ** 2             # Loss differential (squared loss)
    n = len(d)
    d_mean = np.mean(d)

    # Newey-West HAC variance estimator with h-1 lags
    gamma = [np.var(d)]
    for lag in range(1, h):
        cov = np.cov(d[:-lag], d[lag:])[0, 1]
        gamma.append(cov)
    var_d = (gamma[0] + 2 * sum(gamma[1:])) / n

    dm_stat = d_mean / np.sqrt(var_d)
    p_value = 2 * (1 - stats.norm.cdf(abs(dm_stat)))

    return {"DM_statistic": float(dm_stat), "p_value": float(p_value)}


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_insample: Optional[np.ndarray] = None,
    quantile_forecasts: Optional[Dict[str, np.ndarray]] = None,
    alpha_over: float = 1.47,
    alpha_under: float = 1.0,
    seasonal_period: int = 7,
) -> Dict[str, float]:
    """Compute all reported metrics in one call."""
    results = {
        "MAE":   mae(y_true, y_pred),
        "RMSE":  rmse(y_true, y_pred),
        "sMAPE": smape(y_true, y_pred),
        "PCM":   perishable_cost_metric(y_true, y_pred, alpha_over, alpha_under),
    }
    if y_insample is not None:
        results["MASE"] = mase(y_true, y_pred, y_insample, seasonal_period)
    if quantile_forecasts is not None:
        results["Winkler"] = winkler_score(
            y_true,
            quantile_forecasts["q_10"],
            quantile_forecasts["q_90"],
            alpha=0.20,
        )
    return results


if __name__ == "__main__":
    # Sanity check
    np.random.seed(42)
    y_true = np.random.randn(100) * 50 + 200
    y_pred = y_true + np.random.randn(100) * 10
    y_insample = np.random.randn(500) * 50 + 200
    quantile_forecasts = {
        "q_10": y_pred - 20,
        "q_50": y_pred,
        "q_90": y_pred + 20,
    }
    metrics = compute_all_metrics(y_true, y_pred, y_insample, quantile_forecasts)
    for name, val in metrics.items():
        print(f"{name:8s}: {val:.4f}")
