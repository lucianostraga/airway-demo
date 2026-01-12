# ULD Demand Forecasting: Mathematical Framework

**Author**: ML Engineering Team
**Date**: 2026-01-10
**Version**: 1.0
**Status**: Technical Specification

---

## Executive Summary

This document defines a mathematically rigorous, hierarchical probabilistic forecasting framework for Unit Load Device (ULD) demand prediction. The approach combines state-space models, gradient boosting ensembles, and Bayesian hierarchical structures to provide point forecasts with calibrated uncertainty quantification across multiple time horizons.

---

## 1. Problem Formulation

### 1.1 Formal Definition

Let $D_{s,t,u}$ denote the demand for ULD type $u \in \mathcal{U}$ at station $s \in \mathcal{S}$ on day $t \in \mathcal{T}$.

**Objective**: Learn a conditional distribution:

$$P(D_{s,t+h,u} | \mathcal{F}_t, \mathbf{X}_{s,t+h})$$

where:
- $h \in \{0, 1, 7, 30\}$ is the forecast horizon
- $\mathcal{F}_t = \{D_{s,\tau,u} : \tau \leq t, s \in \mathcal{S}, u \in \mathcal{U}\}$ is the filtration (historical information)
- $\mathbf{X}_{s,t+h}$ represents exogenous covariates (weather, events, flight schedules)

### 1.2 Hierarchical Structure

The network exhibits a natural hierarchy that must be respected:

```
Level 0: Network (total ULD demand across all stations)
    │
Level 1: Hub Region (ATL-region, DTW-region, MSP-region, SLC-region)
    │
Level 2: Station (individual airport)
    │
Level 3: Route (origin-destination pair)
    │
Level 4: ULD Type × Route (AKE, AKH, PMC at route level)
```

**Coherence Constraint**: Forecasts must satisfy:

$$\hat{D}_{network,t} = \sum_{r \in Regions} \hat{D}_{r,t} = \sum_{s \in Stations} \hat{D}_{s,t} = \sum_{(o,d) \in Routes} \hat{D}_{o,d,t}$$

---

## 2. Core Mathematical Framework

### 2.1 Structural Time Series Model (State-Space Formulation)

We decompose demand using a state-space model with the following components:

**Observation Equation**:
$$D_{s,t,u} = \mu_{s,t,u} + \gamma_{s,t,u} + \tau_{s,t,u} + \boldsymbol{\beta}_{s,u}^\top \mathbf{X}_{s,t} + \epsilon_{s,t,u}$$

where:
- $\mu_{s,t,u}$: Local level (trend)
- $\gamma_{s,t,u}$: Seasonal component
- $\tau_{s,t,u}$: Holiday/event effect
- $\boldsymbol{\beta}_{s,u}^\top \mathbf{X}_{s,t}$: Regression on exogenous variables
- $\epsilon_{s,t,u} \sim \mathcal{N}(0, \sigma^2_{\epsilon,s,u})$: Observation noise

**State Equations**:

*Local Level (Random Walk with Drift)*:
$$\mu_{s,t+1,u} = \mu_{s,t,u} + \delta_{s,u} + \eta_{s,t,u}, \quad \eta_{s,t,u} \sim \mathcal{N}(0, \sigma^2_\mu)$$

*Seasonal Component (Trigonometric Form)*:
$$\gamma_{s,t,u} = \sum_{j=1}^{J} \left[ \gamma_{j,s,t,u} \cos\left(\frac{2\pi j t}{m}\right) + \gamma^*_{j,s,t,u} \sin\left(\frac{2\pi j t}{m}\right) \right]$$

where $m=7$ for weekly seasonality, $m=365.25$ for annual seasonality.

The harmonics evolve as:
$$\begin{pmatrix} \gamma_{j,s,t+1,u} \\ \gamma^*_{j,s,t+1,u} \end{pmatrix} = \begin{pmatrix} \cos(\lambda_j) & \sin(\lambda_j) \\ -\sin(\lambda_j) & \cos(\lambda_j) \end{pmatrix} \begin{pmatrix} \gamma_{j,s,t,u} \\ \gamma^*_{j,s,t,u} \end{pmatrix} + \begin{pmatrix} \omega_{j,s,t,u} \\ \omega^*_{j,s,t,u} \end{pmatrix}$$

where $\lambda_j = 2\pi j / m$.

### 2.2 Bayesian Hierarchical Pooling

To share information across stations while respecting heterogeneity, we employ partial pooling:

**Station-Level Parameters**:
$$\boldsymbol{\beta}_{s,u} \sim \mathcal{N}(\boldsymbol{\beta}_{tier(s),u}, \Sigma_{tier})$$

**Tier-Level Parameters**:
$$\boldsymbol{\beta}_{tier,u} \sim \mathcal{N}(\boldsymbol{\beta}_{global,u}, \Sigma_{global})$$

**Global Hyperpriors**:
$$\boldsymbol{\beta}_{global,u} \sim \mathcal{N}(\mathbf{0}, \mathbf{I} \cdot 10)$$
$$\Sigma_{tier}, \Sigma_{global} \sim \text{LKJ}(2) \otimes \text{Half-Cauchy}(0, 2.5)$$

This hierarchical structure enables:
- **Cold-start handling**: New stations borrow strength from tier-level estimates
- **Regularization**: Prevents overfitting at low-volume stations
- **Interpretability**: Tier effects are explicitly modeled

### 2.3 Covariate Integration (Exogenous Regressors)

The exogenous feature vector $\mathbf{X}_{s,t}$ includes:

$$\mathbf{X}_{s,t} = \begin{bmatrix}
\text{Scheduled passengers}_{s,t} \\
\text{Weather severity index}_{s,t} \\
\text{Event impact score}_{s,t} \\
\text{Delay probability}_{s,t} \\
\text{Holiday indicator}_{s,t} \\
\mathbf{Fourier}(t, k) \\
\text{Lagged demand}_{s,t-1:t-k}
\end{bmatrix}$$

**Weather Severity Index** (composite):
$$W_{s,t} = w_1 \cdot \mathbb{1}[\text{IFR conditions}] + w_2 \cdot \mathbb{1}[\text{SIGMET active}] + w_3 \cdot \frac{\text{wind speed}}{\text{threshold}}$$

**Event Impact Score** (from PredictHQ or computed):
$$E_{s,t} = \sum_{e \in \text{Events}_{s,t}} \text{attendance}_e \cdot \text{category\_weight}_e \cdot \text{decay}(d_{e,s})$$

where $\text{decay}(d) = \exp(-d / \sigma_d)$ is a distance decay function from event location.

---

## 3. Multi-Horizon Forecasting Architecture

### 3.1 Horizon-Specific Models

Different horizons require different modeling emphasis:

| Horizon | Dominant Signals | Model Architecture |
|---------|-----------------|-------------------|
| Day-of-Ops (h=0) | Real-time disruptions, current inventory | State-space + gradient boosting residuals |
| 7-Day | Weather forecasts, flight schedule changes | Ensemble: LightGBM + State-space |
| 30-Day | Seasonality, events calendar, trends | Structural TS + Hierarchical smoothing |

### 3.2 Ensemble Combination

We combine forecasts using a stacked generalization approach:

**Base Learners**:
1. $\hat{f}_1$: Structural Time Series (Prophet-style decomposition)
2. $\hat{f}_2$: LightGBM with lag features
3. $\hat{f}_3$: Temporal Fusion Transformer (optional, for complex patterns)
4. $\hat{f}_4$: SARIMA(p,d,q)(P,D,Q)[7]

**Meta-Learner** (Quantile Regression):
$$\hat{Q}_{\alpha}(D_{s,t+h}) = g_\alpha\left(\hat{f}_1(\mathbf{X}), \hat{f}_2(\mathbf{X}), \hat{f}_3(\mathbf{X}), \hat{f}_4(\mathbf{X})\right)$$

where $g_\alpha$ is trained to predict the $\alpha$-quantile using the base forecasts as features.

**Optimal Weights** (learned via cross-validation):
$$\mathbf{w}^* = \arg\min_{\mathbf{w}} \sum_{t \in \mathcal{T}_{val}} \rho_\alpha\left(D_t - \sum_{k} w_k \hat{f}_k(\mathbf{X}_t)\right)$$

where $\rho_\alpha(\cdot)$ is the pinball loss for quantile $\alpha$.

---

## 4. Feature Engineering Framework

### 4.1 Temporal Feature Taxonomy

```python
class TemporalFeatureSet:
    """Mathematical specification of temporal features."""

    # Cyclical encoding (preserves periodicity)
    def cyclical_encode(self, t: int, period: int) -> tuple[float, float]:
        """
        Maps time t to unit circle:
        sin(2*pi*t/T), cos(2*pi*t/T)

        Preserves distance: day 7 is close to day 1 for weekly cycle.
        """
        theta = 2 * np.pi * t / period
        return np.sin(theta), np.cos(theta)

    # Fourier basis expansion
    def fourier_features(self, t: np.ndarray, period: int, K: int) -> np.ndarray:
        """
        Fourier series approximation for flexible seasonality:

        f(t) = sum_{k=1}^{K} [a_k * sin(2*pi*k*t/T) + b_k * cos(2*pi*k*t/T)]

        K controls smoothness: higher K = more flexible but risk overfitting
        """
        features = []
        for k in range(1, K + 1):
            features.append(np.sin(2 * np.pi * k * t / period))
            features.append(np.cos(2 * np.pi * k * t / period))
        return np.column_stack(features)
```

### 4.2 Lag Feature Selection (Information-Theoretic)

Select lag features using Conditional Mutual Information:

$$I(D_t; D_{t-k} | D_{t-1}, ..., D_{t-k+1}) = H(D_t | D_{t-1:t-k+1}) - H(D_t | D_{t-1:t-k})$$

**Algorithm**:
1. Start with empty lag set $\mathcal{L} = \{\}$
2. For each candidate lag $k$:
   - Compute $CMI(D_t; D_{t-k} | D_{\mathcal{L}})$
   - If $CMI > \theta$, add $k$ to $\mathcal{L}$
3. Repeat until no lags add significant information

**Standard Lags** (empirically validated):
- Short-term: $k \in \{1, 2, 3, 7\}$ (day-over-day, week-over-week)
- Medium-term: $k \in \{14, 21, 28\}$ (bi-weekly patterns)
- Long-term: $k \in \{364, 365, 371\}$ (year-over-year, same weekday)

### 4.3 Rolling Statistics with Exponential Decay

$$\text{EWMA}_{t,\alpha} = \alpha \cdot D_{t-1} + (1-\alpha) \cdot \text{EWMA}_{t-1,\alpha}$$

**Multi-scale aggregations**:
```python
rolling_features = {
    'demand_ewm_7':   ewm(span=7),     # alpha = 2/(7+1)
    'demand_ewm_28':  ewm(span=28),    # alpha = 2/(28+1)
    'demand_std_7':   rolling(7).std(),
    'demand_skew_28': rolling(28).skew(),  # Asymmetry detection
    'demand_kurt_28': rolling(28).kurt(),  # Tail behavior
}
```

### 4.4 Cross-Station Features (Network Effects)

**Hub Spillover Effect**:
$$\text{HubPressure}_{s,t} = \sum_{h \in \text{Hubs}} \frac{D_{h,t} - \bar{D}_h}{\sigma_h} \cdot \text{connectivity}(s, h)$$

**Network Imbalance**:
$$\text{Imbalance}_{s,t} = \text{Inflow}_{s,t} - \text{Outflow}_{s,t} = \sum_{r: dest=s} D_{r,t} - \sum_{r: orig=s} D_{r,t}$$

---

## 5. Uncertainty Quantification

### 5.1 Conformal Prediction Intervals

Rather than relying on distributional assumptions, we use conformal prediction for distribution-free coverage guarantees.

**Algorithm** (Split Conformal):

1. Split data: $\mathcal{D} = \mathcal{D}_{train} \cup \mathcal{D}_{calib}$
2. Train model on $\mathcal{D}_{train}$, get predictions $\hat{D}_i$ for $i \in \mathcal{D}_{calib}$
3. Compute nonconformity scores: $R_i = |D_i - \hat{D}_i|$
4. Find quantile: $q = \lceil (1-\alpha)(|\mathcal{D}_{calib}| + 1) \rceil$ smallest $R_i$
5. Prediction interval: $[\hat{D}_{new} - q, \hat{D}_{new} + q]$

**Theorem (Coverage Guarantee)**:
For exchangeable data, this procedure achieves:
$$P(D_{new} \in [\hat{D}_{new} - q, \hat{D}_{new} + q]) \geq 1 - \alpha$$

### 5.2 Conformalized Quantile Regression (CQR)

For heteroscedastic data (variance changes with covariates):

**Nonconformity Score**:
$$R_i = \max\left(\hat{Q}_{\alpha/2}(\mathbf{X}_i) - D_i, D_i - \hat{Q}_{1-\alpha/2}(\mathbf{X}_i)\right)$$

**Interval**:
$$\left[\hat{Q}_{\alpha/2}(\mathbf{X}_{new}) - q, \hat{Q}_{1-\alpha/2}(\mathbf{X}_{new}) + q\right]$$

This adapts interval width to local uncertainty while maintaining coverage.

### 5.3 Quantile Loss Function

For training quantile estimators:

$$\mathcal{L}_\alpha(y, \hat{y}) = \rho_\alpha(y - \hat{y}) = (y - \hat{y}) \cdot (\alpha - \mathbb{1}[y < \hat{y}])$$

**Properties**:
- $\alpha = 0.5$: Equivalent to MAE (median regression)
- $\alpha = 0.1$: 10th percentile (conservative lower bound)
- $\alpha = 0.9$: 90th percentile (conservative upper bound)

---

## 6. Anomaly Detection and Regime Change

### 6.1 Online Change Point Detection (BOCPD)

We use Bayesian Online Change Point Detection (Adams & MacKay, 2007):

**Run Length Distribution**:
Let $r_t$ be the run length (time since last change point). We maintain:

$$P(r_t | D_{1:t}) = \frac{P(r_t, D_{1:t})}{\sum_{r'} P(r' = r_t, D_{1:t})}$$

**Recursive Update**:
$$P(r_t = r_{t-1} + 1, D_{1:t}) = P(D_t | r_{t-1}, D_{1:t-1}) \cdot P(r_{t-1} | D_{1:t-1}) \cdot (1 - H)$$
$$P(r_t = 0, D_{1:t}) = P(D_t | \text{new segment}) \cdot \sum_{r_{t-1}} P(r_{t-1} | D_{1:t-1}) \cdot H$$

where $H$ is the hazard function (prior probability of change).

**Change Point Signal**:
$$\text{CPD}_t = P(r_t = 0 | D_{1:t})$$

Alert when $\text{CPD}_t > \tau_{alert}$.

### 6.2 Isolation Forest for Multivariate Anomalies

For detecting anomalies in feature space (not just demand):

**Anomaly Score**:
$$s(x, n) = 2^{-\frac{E[h(x)]}{c(n)}}$$

where:
- $h(x)$: Path length to isolate $x$ in random trees
- $c(n)$: Average path length in unsuccessful search in BST
- $E[h(x)]$: Expected path length across forest

**Interpretation**:
- $s \approx 1$: Definite anomaly
- $s \approx 0.5$: Normal point
- $s < 0.5$: Dense region (very normal)

### 6.3 CUSUM for Drift Detection

**Cumulative Sum Statistic**:
$$S_t^+ = \max(0, S_{t-1}^+ + (D_t - \mu_0 - k))$$
$$S_t^- = \max(0, S_{t-1}^- - (D_t - \mu_0 + k))$$

**Alert** when $S_t^+ > h$ or $S_t^- > h$.

Parameters:
- $\mu_0$: Target mean (historical baseline)
- $k$: Allowable slack (typically $0.5\sigma$)
- $h$: Decision interval (determines sensitivity vs. false alarm rate)

---

## 7. Hierarchical Reconciliation

### 7.1 Problem: Forecast Incoherence

Independent forecasts at different hierarchy levels will not sum correctly:
$$\hat{D}_{network} \neq \sum_{s} \hat{D}_s$$

### 7.2 Optimal Reconciliation (MinT)

**Matrix Formulation**:
Let $\mathbf{y}_t$ be the vector of all forecasts at all levels. The summing matrix $\mathbf{S}$ maps bottom-level to all levels:

$$\mathbf{y}_t = \mathbf{S} \mathbf{b}_t$$

where $\mathbf{b}_t$ are the bottom-level (route × ULD type) series.

**Minimum Trace Reconciliation**:
$$\tilde{\mathbf{y}} = \mathbf{S}(\mathbf{S}'\mathbf{W}^{-1}\mathbf{S})^{-1}\mathbf{S}'\mathbf{W}^{-1}\hat{\mathbf{y}}$$

where $\mathbf{W}$ is a weight matrix. Options:

| Method | $\mathbf{W}$ | Properties |
|--------|-------------|------------|
| OLS | $\mathbf{I}$ | Simple, unbiased |
| WLS-var | $\text{diag}(\hat{\mathbf{W}}_1)$ | Weights by forecast variance |
| MinT-shr | $\hat{\mathbf{W}}_h$ (shrunk covariance) | Accounts for correlation, best accuracy |

### 7.3 Probabilistic Coherent Forecasts

For uncertainty propagation through reconciliation, we use:

$$\tilde{\mathbf{y}} | \hat{\mathbf{y}} \sim \mathcal{N}\left(\mathbf{S}\mathbf{G}\hat{\mathbf{y}}, \mathbf{S}\mathbf{G}\mathbf{W}\mathbf{G}'\mathbf{S}'\right)$$

where $\mathbf{G} = (\mathbf{S}'\mathbf{W}^{-1}\mathbf{S})^{-1}\mathbf{S}'\mathbf{W}^{-1}$.

---

## 8. Cold-Start Problem for New Routes

### 8.1 Warm-Start Strategies

**Strategy 1: Hierarchical Prior**
New route $r_{new}$ inherits from its station's tier:

$$\boldsymbol{\theta}_{r_{new}} \sim \mathcal{N}\left(\boldsymbol{\theta}_{tier(r_{new})}, \Sigma_{prior}\right)$$

After observing $n$ days of data, the posterior:
$$\boldsymbol{\theta}_{r_{new}} | D_{1:n} \sim \mathcal{N}\left(\boldsymbol{\mu}_{posterior}, \boldsymbol{\Sigma}_{posterior}\right)$$

converges to data-driven estimates as $n \to \infty$.

**Strategy 2: Analogous Route Transfer**

Find $k$ most similar existing routes using feature similarity:

$$\text{sim}(r_{new}, r_j) = \exp\left(-\frac{\|\mathbf{z}_{r_{new}} - \mathbf{z}_{r_j}\|^2}{2\sigma^2}\right)$$

where $\mathbf{z}_r$ includes:
- Origin station tier
- Destination station tier
- Route distance (great circle)
- Aircraft type distribution
- Day-of-week pattern (business vs leisure)

**Transfer Forecast**:
$$\hat{D}_{r_{new},t} = \sum_{j=1}^{k} w_j \cdot \hat{D}_{r_j,t} \cdot \frac{\text{capacity}_{r_{new}}}{\text{capacity}_{r_j}}$$

where $w_j \propto \text{sim}(r_{new}, r_j)$.

### 8.2 Meta-Learning Approach (MAML-style)

For rapid adaptation to new routes:

**Objective**:
$$\min_{\boldsymbol{\theta}} \sum_{r \in \mathcal{R}_{train}} \mathcal{L}\left(f_{\boldsymbol{\theta}'_r}, \mathcal{D}_r^{test}\right)$$

where $\boldsymbol{\theta}'_r = \boldsymbol{\theta} - \alpha \nabla_{\boldsymbol{\theta}} \mathcal{L}(f_{\boldsymbol{\theta}}, \mathcal{D}_r^{train})$.

This learns initialization $\boldsymbol{\theta}$ that adapts quickly with few gradient steps on new route data.

---

## 9. Evaluation Methodology

### 9.1 Metric Portfolio

| Metric | Formula | Target | Why |
|--------|---------|--------|-----|
| **MAPE** | $\frac{100}{n}\sum\|\frac{D_t - \hat{D}_t}{D_t}\|$ | < 15% (7-day) | Standard accuracy |
| **RMSE** | $\sqrt{\frac{1}{n}\sum(D_t - \hat{D}_t)^2}$ | Context-dependent | Penalizes large errors |
| **MAE** | $\frac{1}{n}\sum\|D_t - \hat{D}_t\|$ | < 2 ULDs | Robust to outliers |
| **Bias** | $\frac{1}{n}\sum(\hat{D}_t - D_t)$ | $\pm$0.5 ULDs | Systematic error |
| **MASE** | $\frac{MAE}{MAE_{naive}}$ | < 1.0 | Skill vs. baseline |
| **Coverage** | $\frac{1}{n}\sum\mathbb{1}[D_t \in CI]$ | 80% for 80% CI | Calibration |
| **Interval Width** | $\frac{1}{n}\sum(U_t - L_t)$ | Minimize | Sharpness |

### 9.2 Skill Scores (vs. Baselines)

**Continuous Ranked Probability Skill Score (CRPSS)**:
$$\text{CRPSS} = 1 - \frac{\text{CRPS}_{model}}{\text{CRPS}_{climatology}}$$

where:
$$\text{CRPS}(F, y) = \int_{-\infty}^{\infty} (F(x) - \mathbb{1}[y \leq x])^2 dx$$

**Interpretation**:
- CRPSS > 0: Better than climatology
- CRPSS = 1: Perfect forecast
- CRPSS < 0: Worse than climatology (reject model)

### 9.3 Temporal Cross-Validation Protocol

```
Training Window (Expanding)          Forecast Horizon
├──────────────────────────┤        ├────────┤
[=====Train=====][Gap][===Test===]
[========Train========][Gap][===Test===]
[===========Train===========][Gap][===Test===]
```

**Protocol**:
1. **Minimum training**: 365 days (capture full seasonality)
2. **Gap**: 1 day (prevent leakage from lag features)
3. **Test window**: Matches forecast horizon (7 days for 7-day forecast)
4. **Step**: 7 days (weekly retraining simulation)
5. **Folds**: Minimum 52 (one year of weekly evaluations)

### 9.4 Statistical Significance Testing

**Diebold-Mariano Test** for comparing forecasts:

$$DM = \frac{\bar{d}}{\sqrt{\hat{V}(\bar{d})/T}} \xrightarrow{d} \mathcal{N}(0,1)$$

where $d_t = L(\epsilon_{1,t}) - L(\epsilon_{2,t})$ is the loss differential.

**Model Confidence Set (MCS)** for selecting best models:
- Tests all pairwise comparisons
- Returns set of models not significantly worse than the best
- Controls family-wise error rate

---

## 10. Algorithm Selection by Component

### 10.1 Recommended Model Portfolio

| Component | Primary | Alternative | Rationale |
|-----------|---------|-------------|-----------|
| **Point Forecast** | LightGBM | XGBoost | Fast, handles categoricals, good with tabular features |
| **Trend/Seasonality** | Prophet | TBATS | Interpretable decomposition, handles holidays |
| **Short-horizon** | State-space (DLM) | ARIMA | Real-time updating, principled uncertainty |
| **Uncertainty** | CQR | Gaussian Process | Distribution-free coverage guarantees |
| **Anomaly** | Isolation Forest + BOCPD | LOF + PELT | Complementary (feature-space + time-series) |
| **Hierarchy** | MinT (shrunk) | Bottom-up | Optimal reconciliation with correlations |
| **Cold-start** | Bayesian hierarchical | kNN transfer | Principled pooling with uncertainty |

### 10.2 Ensemble Architecture

```
Input Features (X_t)
        │
        ├──────────────────┬──────────────────┬──────────────────┐
        ▼                  ▼                  ▼                  ▼
   ┌─────────┐       ┌──────────┐       ┌─────────┐       ┌─────────┐
   │ LightGBM │       │  Prophet  │       │  ARIMA  │       │   TFT   │
   │ (XGBoost)│       │ (trend+  │       │ (short- │       │(optional│
   │          │       │ seasonal)│       │  term)  │       │deep TS) │
   └────┬─────┘       └────┬─────┘       └────┬────┘       └────┬────┘
        │                  │                  │                  │
        └────────┬─────────┴─────────┬────────┴────────┬─────────┘
                 ▼                                     ▼
         ┌──────────────┐                    ┌──────────────────┐
         │  Meta-Learner │                    │ Quantile Outputs │
         │ (Linear/Ridge)│                    │  (10%, 50%, 90%) │
         └──────┬───────┘                    └────────┬─────────┘
                │                                     │
                ▼                                     ▼
         Point Forecast                     Prediction Intervals
                │                                     │
                └──────────────┬──────────────────────┘
                               ▼
                    ┌───────────────────┐
                    │   Hierarchical    │
                    │  Reconciliation   │
                    │    (MinT)         │
                    └─────────┬─────────┘
                              │
                              ▼
                    Coherent Forecasts
                    (Network → Route)
```

---

## 11. Implementation Specifications

### 11.1 LightGBM Configuration

```python
lgb_params = {
    'objective': 'quantile',           # Or 'regression' for point
    'alpha': 0.5,                       # Quantile level
    'metric': 'quantile',
    'boosting_type': 'gbdt',
    'num_leaves': 31,                   # Complexity control
    'learning_rate': 0.05,
    'feature_fraction': 0.8,            # Column subsampling
    'bagging_fraction': 0.8,            # Row subsampling
    'bagging_freq': 5,
    'min_child_samples': 20,            # Regularization
    'lambda_l1': 0.1,                   # L1 regularization
    'lambda_l2': 0.1,                   # L2 regularization
    'verbosity': -1,
    'n_estimators': 1000,
    'early_stopping_rounds': 50,
}
```

### 11.2 Prophet Configuration

```python
prophet_model = Prophet(
    growth='linear',                    # Or 'logistic' with capacity
    seasonality_mode='multiplicative',  # Demand typically multiplicative
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False,            # Usually too noisy
    holidays=holiday_df,
    interval_width=0.80,                # 80% prediction interval
    changepoint_prior_scale=0.05,       # Flexibility for trend changes
    seasonality_prior_scale=10,         # Strength of seasonality
)

# Add custom regressors
prophet_model.add_regressor('scheduled_passengers', standardize=True)
prophet_model.add_regressor('weather_severity', standardize=True)
prophet_model.add_regressor('event_impact', standardize=True)
```

### 11.3 Conformal Prediction Implementation

```python
class ConformQR:
    """Conformalized Quantile Regression for calibrated intervals."""

    def __init__(self, base_model, alpha: float = 0.1):
        self.alpha = alpha
        self.base_model = base_model
        self.q_lo = None
        self.q_hi = None

    def fit(self, X_train, y_train, X_calib, y_calib):
        # Fit base model for lower and upper quantiles
        self.model_lo = clone(self.base_model)
        self.model_hi = clone(self.base_model)

        self.model_lo.set_params(alpha=self.alpha/2)
        self.model_hi.set_params(alpha=1-self.alpha/2)

        self.model_lo.fit(X_train, y_train)
        self.model_hi.fit(X_train, y_train)

        # Calibration: compute nonconformity scores
        pred_lo = self.model_lo.predict(X_calib)
        pred_hi = self.model_hi.predict(X_calib)

        scores = np.maximum(pred_lo - y_calib, y_calib - pred_hi)

        # Quantile of scores (finite sample correction)
        n = len(scores)
        self.q = np.quantile(scores, np.ceil((n+1)*(1-self.alpha))/n)

    def predict(self, X):
        """Returns (point, lower, upper) predictions."""
        lo = self.model_lo.predict(X) - self.q
        hi = self.model_hi.predict(X) + self.q
        point = (lo + hi) / 2
        return point, lo, hi
```

---

## 12. Production Monitoring

### 12.1 Forecast Monitoring Metrics

```python
@dataclass
class ForecastMonitor:
    """Track forecast quality in production."""

    # Calibration: Are prediction intervals reliable?
    def compute_coverage(self, actuals, lower, upper) -> float:
        return np.mean((actuals >= lower) & (actuals <= upper))

    # Sharpness: Are intervals tight?
    def compute_sharpness(self, lower, upper) -> float:
        return np.mean(upper - lower)

    # Bias: Systematic over/under prediction?
    def compute_bias(self, actuals, predictions) -> float:
        return np.mean(predictions - actuals)

    # Probabilistic calibration (PIT histogram)
    def pit_values(self, actuals, cdf_predictions) -> np.ndarray:
        """
        Probability Integral Transform.
        If calibrated, PIT values are Uniform[0,1].
        """
        return np.array([cdf(a) for a, cdf in zip(actuals, cdf_predictions)])
```

### 12.2 Drift Detection Dashboard

| Metric | Green | Yellow | Red | Action |
|--------|-------|--------|-----|--------|
| Coverage (80% CI) | 75-85% | 70-75% or 85-90% | <70% or >90% | Recalibrate intervals |
| MAPE (7-day) | <15% | 15-20% | >20% | Investigate feature drift |
| Bias | ±0.5 ULD | ±1.0 ULD | >±1.0 ULD | Retrain model |
| CUSUM signal | Normal | Warning | Alert | Check for regime change |

---

## 13. Mathematical Notation Summary

| Symbol | Meaning |
|--------|---------|
| $D_{s,t,u}$ | Demand for ULD type $u$ at station $s$ on day $t$ |
| $\mathcal{F}_t$ | Information available at time $t$ (filtration) |
| $h$ | Forecast horizon (days ahead) |
| $\mathbf{X}_{s,t}$ | Feature vector for station $s$ at time $t$ |
| $\mu_{s,t}$ | Local level (trend) component |
| $\gamma_{s,t}$ | Seasonal component |
| $\boldsymbol{\beta}$ | Regression coefficients |
| $\mathcal{L}_\alpha$ | Pinball loss for quantile $\alpha$ |
| $\rho_\alpha$ | Quantile loss function |
| $\mathbf{S}$ | Summing matrix for hierarchy |
| $r_t$ | Run length (time since last change point) |
| $\text{CRPS}$ | Continuous Ranked Probability Score |

---

## 14. References

1. **State-Space Models**: Durbin, J., & Koopman, S. J. (2012). *Time Series Analysis by State Space Methods*. Oxford University Press.

2. **Bayesian Hierarchical Models**: Gelman, A., et al. (2013). *Bayesian Data Analysis* (3rd ed.). CRC Press.

3. **Conformal Prediction**: Romano, Y., Patterson, E., & Candes, E. (2019). Conformalized Quantile Regression. *NeurIPS*.

4. **Hierarchical Forecasting**: Wickramasuriya, S. L., Athanasopoulos, G., & Hyndman, R. J. (2019). Optimal Forecast Reconciliation for Hierarchical and Grouped Time Series. *JASA*.

5. **BOCPD**: Adams, R. P., & MacKay, D. J. C. (2007). Bayesian Online Changepoint Detection. *arXiv:0710.3742*.

6. **Prophet**: Taylor, S. J., & Letham, B. (2018). Forecasting at Scale. *The American Statistician*.

7. **Gradient Boosting**: Ke, G., et al. (2017). LightGBM: A Highly Efficient Gradient Boosting Decision Tree. *NeurIPS*.

8. **Temporal Fusion Transformers**: Lim, B., et al. (2021). Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting. *IJOF*.

---

## Appendix A: Feature List Specification

### A.1 Complete Feature Catalog

```python
FEATURE_CATALOG = {
    # Temporal (auto-generated)
    'temporal': [
        'day_of_week',           # 0-6
        'day_of_month',          # 1-31
        'week_of_year',          # 1-52
        'month',                 # 1-12
        'quarter',               # 1-4
        'is_weekend',            # bool
        'is_month_start',        # bool
        'is_month_end',          # bool
        'day_sin', 'day_cos',    # Cyclical day of week
        'month_sin', 'month_cos', # Cyclical month
        'fourier_weekly_1_sin', 'fourier_weekly_1_cos',  # Weekly harmonics
        'fourier_weekly_2_sin', 'fourier_weekly_2_cos',
        'fourier_annual_1_sin', 'fourier_annual_1_cos',  # Annual harmonics
        'fourier_annual_2_sin', 'fourier_annual_2_cos',
    ],

    # Lag features
    'lag': [
        'demand_lag_1', 'demand_lag_2', 'demand_lag_3',
        'demand_lag_7', 'demand_lag_14', 'demand_lag_21', 'demand_lag_28',
        'demand_lag_364', 'demand_lag_365', 'demand_lag_371',  # YoY
    ],

    # Rolling statistics
    'rolling': [
        'demand_roll_mean_7', 'demand_roll_mean_14', 'demand_roll_mean_28',
        'demand_roll_std_7', 'demand_roll_std_14', 'demand_roll_std_28',
        'demand_roll_min_7', 'demand_roll_max_7',
        'demand_ewm_7', 'demand_ewm_28',
    ],

    # External signals
    'weather': [
        'weather_severity_index',
        'is_ifr_conditions',
        'is_sigmet_active',
        'wind_speed_normalized',
        'precip_probability',
        'temp_departure_from_normal',
    ],

    'events': [
        'event_impact_score',
        'event_attendance_sum',
        'event_count',
        'is_major_event',      # attendance > threshold
        'days_to_next_event',
        'days_from_last_event',
    ],

    'holidays': [
        'is_holiday',
        'is_holiday_eve',
        'is_holiday_observed',
        'days_to_holiday',
        'days_from_holiday',
        'holiday_type',        # Federal, religious, school
    ],

    # Flight schedule
    'schedule': [
        'scheduled_departures',
        'scheduled_arrivals',
        'scheduled_passengers',
        'scheduled_widebody_flights',
        'schedule_change_indicator',  # vs. 7 days ago
    ],

    # Network/hierarchy
    'network': [
        'station_tier',        # Hub/Focus/Spoke
        'hub_pressure',        # Imbalance at connected hubs
        'network_imbalance',   # Inflow - outflow
        'route_density',       # Flights per day on route
    ],

    # Historical patterns
    'patterns': [
        'same_weekday_last_month',
        'same_weekday_last_year',
        'monthly_seasonal_index',
        'weekly_seasonal_index',
    ],
}
```

---

## Appendix B: Baseline Models for Comparison

### B.1 Naive Baselines

```python
class NaiveBaselines:
    """Baselines that any model must beat."""

    @staticmethod
    def seasonal_naive(history: pd.Series, horizon: int, season: int = 7):
        """Forecast = same value from {season} days ago."""
        return history.iloc[-season:-season+horizon].values

    @staticmethod
    def moving_average(history: pd.Series, horizon: int, window: int = 7):
        """Forecast = average of last {window} values."""
        return np.full(horizon, history.iloc[-window:].mean())

    @staticmethod
    def drift(history: pd.Series, horizon: int):
        """Linear extrapolation from first to last observation."""
        n = len(history)
        slope = (history.iloc[-1] - history.iloc[0]) / n
        return history.iloc[-1] + slope * np.arange(1, horizon + 1)

    @staticmethod
    def ets_auto(history: pd.Series, horizon: int):
        """Exponential smoothing (automatic selection)."""
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        model = ExponentialSmoothing(history, seasonal='add', seasonal_periods=7)
        fit = model.fit()
        return fit.forecast(horizon)
```

### B.2 Minimum Acceptable Performance

| Horizon | Baseline | Expected MASE |
|---------|----------|---------------|
| Day-of-Ops | Last value | Must beat |
| 7-Day | Seasonal naive (same weekday last week) | MASE < 0.85 |
| 30-Day | Seasonal naive (same weekday last month) | MASE < 0.90 |

---

*Document Version: 1.0*
*Last Updated: 2026-01-10*
*Next Review: Upon data availability milestone*
