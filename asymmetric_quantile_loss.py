"""
Asymmetric Quantile Loss for MSF-Net.

Modifies the standard pinball loss [Koenker & Bassett, 1978] by differentially
scaling over- and under-forecast error directions with cost coefficients
α_over and α_under, calibrated empirically from wholesale price records.

Reduces to symmetric pinball loss when α_over = α_under = 1.
"""

from typing import Dict, List
import torch
import torch.nn as nn


class AsymmetricQuantileLoss(nn.Module):
    r"""
    For each quantile level τ:

        L_τ(y, ŷ_τ) = α_over  · τ · (ŷ_τ − y)       if ŷ_τ > y    (over-forecast)
                    = α_under · (1 − τ) · (y − ŷ_τ)  if ŷ_τ ≤ y    (under-forecast)

    Total loss averages over all quantile levels, forecast steps, and samples.
    """

    def __init__(
        self,
        quantile_levels: List[float],
        alpha_over: float = 1.47,
        alpha_under: float = 1.0,
    ):
        super().__init__()
        self.quantile_levels = quantile_levels
        self.alpha_over = alpha_over
        self.alpha_under = alpha_under

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],   # {'q_10': (B, H), 'q_50': ..., 'q_90': ...}
        targets: torch.Tensor,                   # (B, H)
    ) -> torch.Tensor:
        total_loss = 0.0
        for q in self.quantile_levels:
            key = f"q_{int(q * 100)}"
            pred = predictions[key]
            diff = pred - targets                # positive = over-forecast

            loss_over = self.alpha_over * q * diff.clamp(min=0)
            loss_under = self.alpha_under * (1 - q) * (-diff).clamp(min=0)

            total_loss = total_loss + (loss_over + loss_under).mean()

        return total_loss / len(self.quantile_levels)


class SymmetricPinballLoss(AsymmetricQuantileLoss):
    """Backward-compatible pinball loss (α_over = α_under = 1). Used in Variant D ablation."""

    def __init__(self, quantile_levels: List[float]):
        super().__init__(quantile_levels, alpha_over=1.0, alpha_under=1.0)


if __name__ == "__main__":
    # Sanity check
    loss_fn = AsymmetricQuantileLoss([0.10, 0.50, 0.90], alpha_over=1.47, alpha_under=1.0)
    B, H = 4, 7
    preds = {
        "q_10": torch.randn(B, H),
        "q_50": torch.randn(B, H),
        "q_90": torch.randn(B, H),
    }
    targets = torch.randn(B, H)
    loss = loss_fn(preds, targets)
    print(f"Loss: {loss.item():.4f}")
