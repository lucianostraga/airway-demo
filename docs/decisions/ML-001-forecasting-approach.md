# ML-001: Forecasting Mathematical Approach

**Date**: 2026-01-10
**Status**: Accepted
**Deciders**: ML Engineering, Architecture Team

## Context

The ULD Forecasting system requires a mathematically sophisticated approach that:
- Handles multiple forecasting horizons (day-of-ops to 30-day)
- Incorporates external signals (weather, events)
- Provides calibrated uncertainty quantification
- Respects hierarchical network structure
- Handles cold-start for new routes

## Decision

Implement a **Hierarchical Probabilistic Ensemble** approach combining:
1. State-space models for structural decomposition
2. Gradient boosting for non-linear patterns
3. Bayesian hierarchical pooling for information sharing
4. Conformal prediction for calibrated uncertainty

Full specification: [ML-FRAMEWORK-uld-forecasting.md](../ML-FRAMEWORK-uld-forecasting.md)

## Mathematical Foundation

### Core Model

**Observation Equation**:
$$D_{s,t,u} = \mu_{s,t,u} + \gamma_{s,t,u} + \tau_{s,t,u} + \boldsymbol{\beta}_{s,u}^\top \mathbf{X}_{s,t} + \epsilon_{s,t,u}$$

Components:
- $\mu$: Local level (trend via random walk with drift)
- $\gamma$: Seasonal (trigonometric Fourier representation)
- $\tau$: Holiday/event effects
- $\beta^\top X$: External regressors (weather, events, flights)
- $\epsilon$: Observation noise

### Hierarchical Structure

```
Network (Level 0)
├── Hub Region (Level 1): ATL, DTW, MSP, SLC regions
│   ├── Station (Level 2): Individual airports
│   │   ├── Route (Level 3): O-D pairs
│   │   │   └── ULD Type (Level 4): AKE, AKH, PMC per route
```

Bayesian partial pooling:
$$\boldsymbol{\beta}_{station} \sim \mathcal{N}(\boldsymbol{\beta}_{tier}, \Sigma_{tier})$$
$$\boldsymbol{\beta}_{tier} \sim \mathcal{N}(\boldsymbol{\beta}_{global}, \Sigma_{global})$$

### Multi-Horizon Architecture

| Horizon | Primary Model | Ensemble Weight |
|---------|---------------|-----------------|
| Day-of-ops (h=0) | State-space + GBM residuals | Real-time signals dominant |
| 7-day | LightGBM + Prophet ensemble | Weather features weighted |
| 30-day | Structural TS + hierarchical | Seasonality dominant |

### Uncertainty Quantification

**Conformalized Quantile Regression (CQR)**:

Provides distribution-free coverage guarantee:
$$P(Y_{new} \in [\hat{L}, \hat{U}]) \geq 1 - \alpha$$

Nonconformity score:
$$s(x,y) = \max(\hat{q}_{\alpha/2}(x) - y, y - \hat{q}_{1-\alpha/2}(x))$$

### Anomaly/Regime Detection

- **BOCPD**: Bayesian Online Change Point Detection
- **CUSUM**: Cumulative sum for drift detection
- **Isolation Forest**: Multivariate anomaly detection

## Algorithm Selection

| Component | Algorithm | Rationale |
|-----------|-----------|-----------|
| Point forecast | LightGBM | Fast, handles missing data, feature importance |
| Decomposition | Prophet-style | Interpretable components, auto-seasonality |
| Uncertainty | Conformal + Quantile Regression | Distribution-free guarantees |
| Hierarchy | MinT reconciliation | Optimal linear reconciliation |
| Cold-start | Hierarchical priors + transfer | Borrow strength from similar routes |

## Evaluation Metrics

| Metric | Target | Formula |
|--------|--------|---------|
| MAPE | < 15% | $\frac{1}{n}\sum\frac{|y-\hat{y}|}{y}$ |
| Coverage | 80% | % actuals in prediction interval |
| CRPS | Minimize | Continuous Ranked Probability Score |
| Interval Score | Minimize | Sharpness + calibration |
| MASE | < 1.0 | vs naive seasonal baseline |

Statistical tests:
- Diebold-Mariano for model comparison
- Model Confidence Set for ensemble selection

## Consequences

**Benefits**:
- Mathematically rigorous, defensible to statisticians
- Calibrated uncertainty enables risk-aware decisions
- Hierarchical structure handles sparse data
- Flexible ensemble adapts to different horizons

**Challenges**:
- Computational cost for full Bayesian inference
- Requires careful tuning of conformalization
- MinT reconciliation adds complexity

**Mitigation**:
- Use variational inference for scalability
- Pre-compute conformal scores
- Cache reconciliation matrices

## Implementation

Source code: `src/forecasting/`

| Module | Purpose |
|--------|---------|
| `features.py` | Feature engineering, external signals |
| `models.py` | LightGBM, Prophet, ensemble |
| `uncertainty.py` | Conformal prediction |
| `hierarchy.py` | MinT reconciliation |
| `anomaly.py` | BOCPD, CUSUM detection |
| `evaluation.py` | Metrics, cross-validation |
| `cold_start.py` | Transfer learning, priors |

## References

- [Full Framework Specification](../ML-FRAMEWORK-uld-forecasting.md)
- Hyndman, R.J. & Athanasopoulos, G. (2021). Forecasting: Principles and Practice
- Romano, Y. et al. (2019). Conformalized Quantile Regression
- Wickramasuriya, S.L. et al. (2019). Optimal forecast reconciliation
