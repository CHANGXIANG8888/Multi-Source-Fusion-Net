"""
MSF-Net: Multi-Source Fusion Network for Perishable Demand Forecasting.

Full architecture in a single self-contained file:
  - Path A: Demand-Temporal Encoder   (TCN + sparse temporal attention)
  - Path B: Weather-Supply Encoder    (Transformer encoder)
  - Path C: Promotion-Context Encoder (two-layer LSTM)
  - Adaptive Seasonal Gate
  - Cross-Path Attention Fusion
  - Quantile output head
"""

from typing import Dict, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm


# =============================================================================
# Path A: Demand-Temporal Encoder (TCN + Sparse Temporal Attention)
# =============================================================================

class DilatedCausalConvBlock(nn.Module):
    """One TCN block: two dilated causal convolutions + residual + GELU."""
    def __init__(self, d_in, d_out, kernel_size, dilation, dropout):
        super().__init__()
        pad = (kernel_size - 1) * dilation
        self.conv1 = weight_norm(nn.Conv1d(d_in, d_out, kernel_size, padding=pad, dilation=dilation))
        self.conv2 = weight_norm(nn.Conv1d(d_out, d_out, kernel_size, padding=pad, dilation=dilation))
        self.dropout = nn.Dropout(dropout)
        self.residual = nn.Conv1d(d_in, d_out, 1) if d_in != d_out else nn.Identity()

    def forward(self, x):
        T = x.size(-1)
        y = F.gelu(self.conv1(x)[..., :T])
        y = self.dropout(y)
        y = F.gelu(self.conv2(y)[..., :T])
        return y + self.residual(x)


class SparseTemporalAttention(nn.Module):
    """Top-p sparse attention; retains only top fraction p of keys per query."""
    def __init__(self, d_model, top_p=0.25):
        super().__init__()
        self.d_model = d_model
        self.top_p = top_p
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

    def forward(self, h):
        q, k, v = self.q_proj(h), self.k_proj(h), self.v_proj(h)
        scores = (q @ k.transpose(-2, -1)) / (self.d_model ** 0.5)
        T = scores.size(-1)
        k_keep = max(1, int(self.top_p * T))
        topk = torch.topk(scores, k_keep, dim=-1)
        mask = torch.full_like(scores, float('-inf'))
        mask.scatter_(-1, topk.indices, topk.values)
        attn = F.softmax(mask, dim=-1)
        return (attn @ v)[:, -1, :]


class DemandTemporalEncoder(nn.Module):
    def __init__(self, d_input, d_model, n_layers=6, kernel_size=3,
                 dilation_factors=None, sparse_top_p=0.25, dropout=0.1):
        super().__init__()
        if dilation_factors is None:
            dilation_factors = [2 ** i for i in range(n_layers)]
        blocks, d_in = [], d_input
        for dil in dilation_factors:
            blocks.append(DilatedCausalConvBlock(d_in, d_model, kernel_size, dil, dropout))
            d_in = d_model
        self.tcn = nn.Sequential(*blocks)
        self.attn = SparseTemporalAttention(d_model, sparse_top_p)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        # x: (B, T, d_input) -> (B, d_model)
        z = self.tcn(x.transpose(1, 2)).transpose(1, 2)
        return self.proj(z[:, -1, :] + self.attn(z))


# =============================================================================
# Path B: Weather-Supply Encoder (Transformer)
# =============================================================================

class WeatherSupplyEncoder(nn.Module):
    def __init__(self, d_input, d_model, n_layers=2, n_heads=4, d_ff=256,
                 max_len=30, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(d_input, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, activation='gelu')
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)

    def forward(self, x):
        z = self.input_proj(x) + self.pos_emb[:, :x.size(1), :]
        return self.encoder(z).mean(dim=1)


# =============================================================================
# Path C: Promotion-Context Encoder (LSTM)
# =============================================================================

class PromotionContextEncoder(nn.Module):
    def __init__(self, d_input, d_model, hidden_size=64, n_layers=2, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(d_input, hidden_size, n_layers,
                            dropout=dropout if n_layers > 1 else 0.0,
                            batch_first=True)
        self.proj = nn.Linear(hidden_size, d_model)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return self.proj(h_n[-1])


# =============================================================================
# Adaptive Seasonal Gate
# =============================================================================

class AdaptiveSeasonalGate(nn.Module):
    """Softmax over 3 paths conditioned on seasonal context vector s_t."""
    def __init__(self, context_dim, hidden_dim=64, n_paths=3):
        super().__init__()
        self.w1 = nn.Linear(context_dim, hidden_dim)
        self.w2 = nn.Linear(hidden_dim, n_paths)

    def forward(self, s_t):
        return F.softmax(self.w2(F.relu(self.w1(s_t))), dim=-1)


# =============================================================================
# Cross-Path Attention Fusion
# =============================================================================

class CrossPathAttentionFusion(nn.Module):
    """Multi-head cross-attention over the three gated path tokens."""
    def __init__(self, d_model, n_heads=4, d_ff=256, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout,
                                           batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model))

    def forward(self, h_a, h_b, h_c):
        tokens = torch.stack([h_a, h_b, h_c], dim=1)       # (B, 3, d_model)
        out, _ = self.attn(tokens, tokens, tokens)
        return self.ffn(self.norm(out.sum(dim=1)))


# =============================================================================
# Full MSF-Net
# =============================================================================

class MSFNet(nn.Module):
    def __init__(self, config: Dict):
        super().__init__()
        d_model = config["model"]["d_model"]
        dropout = config["model"]["dropout"]
        self.horizon = config["input"]["horizon"]
        self.quantiles = config["loss"]["quantile_levels"]

        self.path_a = DemandTemporalEncoder(
            d_input=config["input"]["feature_dims"]["d_a"],
            d_model=d_model,
            n_layers=config["model"]["path_a"]["n_layers"],
            kernel_size=config["model"]["path_a"]["kernel_size"],
            dilation_factors=config["model"]["path_a"]["dilation_factors"],
            sparse_top_p=config["model"]["path_a"]["sparse_top_p"],
            dropout=dropout,
        )
        self.path_b = WeatherSupplyEncoder(
            d_input=config["input"]["feature_dims"]["d_b"],
            d_model=d_model,
            n_layers=config["model"]["path_b"]["n_layers"],
            n_heads=config["model"]["path_b"]["n_heads"],
            d_ff=config["model"]["path_b"]["d_ff"],
            max_len=config["input"]["window_length"],
            dropout=dropout,
        )
        self.path_c = PromotionContextEncoder(
            d_input=config["input"]["feature_dims"]["d_c"],
            d_model=d_model,
            hidden_size=config["model"]["path_c"]["hidden_size"],
            n_layers=config["model"]["path_c"]["n_layers"],
            dropout=dropout,
        )
        self.gate = AdaptiveSeasonalGate(
            context_dim=config["model"]["gate"]["context_dim"],
            hidden_dim=config["model"]["gate"]["hidden_dim"],
        )
        self.fusion = CrossPathAttentionFusion(
            d_model=d_model,
            n_heads=config["model"]["fusion"]["n_heads"],
            d_ff=config["model"]["fusion"]["d_ff"],
            dropout=dropout,
        )
        self.output_head = nn.ModuleDict({
            f"q_{int(q*100)}": nn.Linear(d_model, self.horizon)
            for q in self.quantiles
        })

    def forward(self, x_a, x_b, x_c, s_t):
        """
        Args:
            x_a: (B, T, d_a) demand-temporal features
            x_b: (B, T, d_b) weather features
            x_c: (B, T, d_c) promotion-context features
            s_t: (B, context_dim) seasonal context vector at forecast origin
        Returns:
            dict {'q_10', 'q_50', 'q_90'}, each of shape (B, H)
        """
        h_a = self.path_a(x_a)
        h_b = self.path_b(x_b)
        h_c = self.path_c(x_c)

        w = self.gate(s_t)                     # (B, 3)
        h_a, h_b, h_c = w[:, 0:1] * h_a, w[:, 1:2] * h_b, w[:, 2:3] * h_c

        u = self.fusion(h_a, h_b, h_c)         # (B, d_model)
        return {name: proj(u) for name, proj in self.output_head.items()}

    def get_gate_weights(self, s_t):
        """Expose gate weights for interpretability analysis (Figure 2)."""
        return self.gate(s_t)


if __name__ == "__main__":
    # Sanity check: forward pass with dummy inputs
    import yaml
    config = yaml.safe_load(open("config.yaml"))
    model = MSFNet(config)
    B, T = 4, config["input"]["window_length"]
    x_a = torch.randn(B, T, config["input"]["feature_dims"]["d_a"])
    x_b = torch.randn(B, T, config["input"]["feature_dims"]["d_b"])
    x_c = torch.randn(B, T, config["input"]["feature_dims"]["d_c"])
    s_t = torch.randn(B, config["model"]["gate"]["context_dim"])
    out = model(x_a, x_b, x_c, s_t)
    for k, v in out.items():
        print(f"{k}: {tuple(v.shape)}")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {n_params:,}")
