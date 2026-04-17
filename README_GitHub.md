# MSF-Net: Multi-Source Fusion Network for Perishable Demand Forecasting

Official implementation of:

**A Multi-Source Deep Learning Framework for Cost-Aligned Demand Forecasting in Perishable Food Supply Chains**
*Submitted to Scientific Reports, 2026*

Jie Zhang, Yue Chang, Xiang Chang (corresponding author), Hong Kong, Shiou Yih Lee, Yujiao Wang

---

## Overview

MSF-Net is an end-to-end deep learning framework for probabilistic demand forecasting in perishable food supply chains. It jointly addresses three interrelated design requirements:

1. **Modality-specific temporal encoding** via three path-specific encoders:
   - Path A (demand-temporal): TCN with sparse temporal attention
   - Path B (weather-supply): Transformer encoder
   - Path C (promotion-context): two-layer LSTM
2. **Context-adaptive source fusion** via an Adaptive Seasonal Gate and a Cross-Path Attention Fusion module
3. **Cost-aligned training** via an Asymmetric Quantile Loss with α_over/α_under calibrated empirically from wholesale price records

On a three-year real-world Chinese supermarket vegetable dataset, MSF-Net reduces MAE by 11.7% and a Perishable Cost Metric by 43.3% relative to TFT, the strongest multi-source baseline, with all improvements statistically significant under the Diebold–Mariano test at p < 0.05.

---

## Repository Contents

```
Multi-Source-Fusion-Net/
├── README.md                         # This file
├── LICENSE                           # MIT License
├── requirements.txt                  # Python dependencies
├── msf_net.py                        # Full MSF-Net model implementation
├── asymmetric_quantile_loss.py       # Cost-aligned training objective
├── metrics.py                        # MAE, RMSE, MASE, sMAPE, Winkler, PCM
└── config.yaml                       # Hyperparameters for all experiments
```

---

## Data Sources

**Supermarket sales data** (Path A, Path C):
[Kaggle — yapwh1208/supermarket-sales-data](https://www.kaggle.com/datasets/yapwh1208/supermarket-sales-data)
878,503 point-of-sale transaction records from a large supermarket chain in China, covering 1 July 2020 to 30 June 2023.

**Weather data** (Path B):
[Open-Meteo Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api)
Daily meteorological variables for four major Chinese vegetable production regions:
- Shouguang, Shandong (36.87°N, 118.73°E)
- Kunming, Yunnan (25.05°N, 102.72°E)
- Guangzhou, Guangdong (23.13°N, 113.26°E)
- Wuhan, Hubei (30.59°N, 114.30°E)

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Import and use MSF-Net
python
>>> import yaml, torch
>>> from msf_net import MSFNet
>>> config = yaml.safe_load(open("config.yaml"))
>>> model = MSFNet(config)
>>> # Forward pass with dummy inputs (batch=4, T=30 days)
>>> x_a = torch.randn(4, 30, 13)   # Path A: demand features
>>> x_b = torch.randn(4, 30, 8)    # Path B: weather features
>>> x_c = torch.randn(4, 30, 4)    # Path C: promotion-context
>>> s_t = torch.randn(4, 16)       # Seasonal context vector
>>> forecasts = model(x_a, x_b, x_c, s_t)
>>> # forecasts = {'q_10': ..., 'q_50': ..., 'q_90': ...} each of shape (4, 7)
```

---

## Key Experimental Settings

All details are in `config.yaml`. Summary:

- Input window: T = 30 days
- Forecast horizon: H = 7 days
- Feature dimensions: d_a = 13, d_b = 8, d_c = 4
- Model dimension: d_model = 128
- Quantile levels: τ ∈ {0.10, 0.50, 0.90}
- Cost coefficients: α_over = 1.47 (cross-SKU mean), α_under = 1.0
- Training: AdamW, lr = 1e-3, cosine annealing, up to 200 epochs, early stopping on val MAE (patience 15)
- Random seeds: [42, 123, 456, 789, 1024] — 5 independent runs per model per SKU

---

## Main Results (H = 7)

| Model | MAE ↓ | PCM ↓ | Winkler ↓ |
|-------|-------|-------|-----------|
| SARIMA | 318.4 | 631.5 | — |
| Prophet | 294.7 | 587.2 | — |
| XGBoost | 247.3 | 496.8 | — |
| LightGBM | 233.6 | 468.3 | — |
| LSTM | 218.4 | 424.6 | 351.2 |
| TCN | 203.1 | 394.3 | 325.8 |
| N-BEATS | 196.8 | 381.7 | 312.4 |
| DeepAR | 188.5 | 363.4 | 298.6 |
| Autoformer | 182.3 | 347.8 | 287.3 |
| TFT | 174.6 | 322.1 | 274.5 |
| PatchTST | 170.2 | 312.6 | 267.8 |
| **MSF-Net (ours)** | **154.2** | **182.5** | **243.7** |

See Table 3 in the paper for complete results with standard deviations and significance tests.

---

## Citation

```bibtex
@article{zhang2026msfnet,
  title   = {A Multi-Source Deep Learning Framework for Cost-Aligned Demand Forecasting in Perishable Food Supply Chains},
  author  = {Zhang, Jie and Chang, Yue and Chang, Xiang and Kong, Hong and Lee, Shiou Yih and Wang, Yujiao},
  journal = {Scientific Reports},
  year    = {2026},
  note    = {Under review}
}
```

---

## License

MIT License — see [LICENSE](LICENSE).

## Contact

**Corresponding author:** Xiang Chang, INTI International University — 444653702@qq.com
For code issues, please open a GitHub issue.
