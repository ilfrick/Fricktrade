# Strategy Upgrade Plan: Toward Industry Best Practices

## Executive Summary

This plan outlines a phased approach to upgrade Fricktrade's trading strategies from
functional prototypes to production-grade quant strategies. The focus is on practical
improvements that can be implemented incrementally, prioritizing high-impact changes
that work within the existing infrastructure.

**Timeline**: 6-12 months for full implementation
**Priority**: Features > Models > Execution > Portfolio Optimization

---

## Phase 1: Feature Engineering (Weeks 1-8)

The single highest-ROI improvement. Better features improve every downstream component.

### 1.1 Market Microstructure Features

| Feature | Description | Data Source | Impact |
|---------|-------------|-------------|--------|
| Bid-ask spread | Real-time spread as % of mid | Alpaca quotes API | High |
| Quote imbalance | (bid_size - ask_size) / (bid_size + ask_size) | Alpaca quotes API | High |
| Trade imbalance | Buy vs sell volume (tick rule classification) | Alpaca trades API | High |
| VWAP deviation | Price distance from session VWAP | Computed from bars | Medium |
| Order flow toxicity | VPIN (Volume-synchronized PIN) | Computed from trades | High |

**Implementation**:
```
app/learning/features.py
├── add microstructure_features(quotes, trades) -> np.ndarray
├── add vwap_deviation(prices, volumes) -> float
└── add order_flow_toxicity(trades, window=50) -> float

app/data/quote_stream.py (new)
├── AlpacaQuoteSubscriber - websocket for real-time quotes
└── QuoteBuffer - rolling buffer for bid/ask/size history
```

### 1.2 Technical Indicator Expansion

Current: SMA, EMA, RSI, ADX (4 indicators)
Target: 25+ indicators across multiple timeframes

| Category | Indicators to Add |
|----------|------------------|
| Momentum | ROC, Williams %R, CCI, Stochastic, TRIX, Ultimate Oscillator |
| Volatility | ATR, Keltner Channels, Donchian Channels, Historical Vol, Garman-Klass Vol |
| Volume | OBV, A/D Line, MFI, VWAP bands, Volume Profile |
| Trend | Parabolic SAR, Supertrend, Ichimoku (tenkan, kijun, senkou) |
| Mean Reversion | Bollinger %B, Z-score, Hurst exponent |

**Implementation**:
```
app/learning/indicators.py (new)
├── compute_all_indicators(ohlcv, periods=[5,10,20,50]) -> dict[str, np.ndarray]
├── normalize_indicators(indicators) -> np.ndarray
└── select_features(indicators, importance_threshold=0.01) -> np.ndarray
```

### 1.3 Multi-Timeframe Features

Current: Single timeframe (5m)
Target: 1m, 5m, 15m, 1h, 1d synchronized features

**Implementation**:
```
app/learning/multi_timeframe.py (new)
├── resample_ohlcv(df_1m, target_tf) -> pd.DataFrame
├── align_timeframes(tf_dict) -> pd.DataFrame  # forward-fill alignment
└── build_mtf_observation(tf_dict, feature_config) -> np.ndarray
```

### 1.4 Alternative Data Integration

| Data Type | Source | Cost | Impact |
|-----------|--------|------|--------|
| News sentiment | Alpaca News API (existing) | Free | Medium |
| Social sentiment | Twitter/StockTwits API | $50-200/mo | Medium |
| Options flow | Unusual Whales / Market Chameleon | $50-100/mo | High |
| Dark pool prints | FINRA ADF data | Free (delayed) | Medium |
| Insider transactions | SEC EDGAR | Free | Low |
| Short interest | FINRA / Ortex | Free-$100/mo | Medium |

**Implementation**:
```
app/data/alternative/
├── sentiment.py - news/social sentiment scoring
├── options_flow.py - unusual options activity detection
├── dark_pool.py - large block trade detection
└── aggregator.py - combine all alt data into feature vector
```

### 1.5 Regime Detection Features

| Feature | Description | Method |
|---------|-------------|--------|
| Volatility regime | Low/Medium/High vol state | Rolling vol percentile |
| Trend regime | Trending/Mean-reverting/Random | Hurst exponent + ADX |
| Correlation regime | Risk-on/Risk-off | SPY-VIX correlation |
| Liquidity regime | Normal/Stressed | Spread + volume deviation |

**Implementation**:
```
app/learning/regime.py (new)
├── classify_volatility_regime(returns, window=60) -> int
├── classify_trend_regime(prices, window=100) -> int
├── detect_regime_change(regime_history) -> bool
└── RegimeHMM - Hidden Markov Model for regime inference
```

---

## Phase 2: Model Architecture Upgrades (Weeks 6-14)

### 2.1 Replace MLP with Temporal Models

Current: PPO with default MLP policy
Target: Custom feature extractors with temporal modeling

**Option A: Temporal Convolutional Network (TCN)**
- Dilated causal convolutions
- Handles variable-length sequences efficiently
- Proven in finance (see: DeepLOB, etc.)

**Option B: Transformer with Temporal Attention**
- Self-attention over time dimension
- Better at capturing long-range dependencies
- Higher compute cost

**Recommended**: Start with TCN (simpler, faster), add transformer later

**Implementation**:
```
app/learning/networks/
├── tcn.py
│   └── TCNFeatureExtractor(nn.Module)
│       ├── TemporalBlock(dilation, kernel_size)
│       └── forward(x) -> features
├── transformer.py
│   └── TemporalTransformer(nn.Module)
│       ├── PositionalEncoding
│       ├── MultiHeadAttention
│       └── forward(x) -> features
└── policy.py
    └── CustomActorCriticPolicy(ActorCriticPolicy)
        └── Uses TCN/Transformer as feature extractor
```

**SB3 Integration**:
```python
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class TCNExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=128):
        super().__init__(observation_space, features_dim)
        self.tcn = TCN(input_dim=observation_space.shape[0], output_dim=features_dim)

    def forward(self, observations):
        return self.tcn(observations)

policy_kwargs = dict(
    features_extractor_class=TCNExtractor,
    features_extractor_kwargs=dict(features_dim=128),
)
model = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs)
```

### 2.2 Ensemble Methods

Current: Single PPO model
Target: Ensemble of diverse models

| Component | Purpose |
|-----------|---------|
| PPO (existing) | Primary RL signal |
| Gradient Boosting (LightGBM) | Return prediction |
| LSTM Classifier | Direction classification |
| Random Forest | Feature importance + signal |

**Implementation**:
```
app/learning/ensemble/
├── base.py - BasePredictor interface
├── lgbm_predictor.py - LightGBM return forecaster
├── lstm_classifier.py - PyTorch LSTM direction classifier
├── rf_signal.py - Random Forest buy/sell classifier
└── ensemble.py
    └── EnsembleStrategy
        ├── predictors: list[BasePredictor]
        ├── weights: dict[str, float]  # learned or fixed
        └── combine(predictions) -> action
```

**Meta-Learning for Ensemble Weights**:
```python
class MetaLearner:
    """Online learning of ensemble weights based on recent performance."""
    def __init__(self, predictors, lookback=100):
        self.predictors = predictors
        self.performance_buffer = deque(maxlen=lookback)

    def update_weights(self, predictions, actual_return):
        # Track which predictor was most accurate
        errors = {name: (pred - actual_return)**2 for name, pred in predictions.items()}
        self.performance_buffer.append(errors)
        # Inverse error weighting
        avg_errors = {name: np.mean([e[name] for e in self.performance_buffer])
                      for name in self.predictors}
        inv_errors = {name: 1/(err + 1e-6) for name, err in avg_errors.items()}
        total = sum(inv_errors.values())
        return {name: w/total for name, w in inv_errors.items()}
```

### 2.3 Hyperparameter Optimization

Current: Manual hyperparameters
Target: Automated Bayesian optimization

**Implementation**:
```
app/learning/hpo/
├── objective.py - Optuna objective function (Sharpe ratio)
├── search_space.py - Define hyperparameter ranges
└── optimize.py
    └── run_hpo(n_trials=100, study_name="ppo_hpo")
```

**Example search space**:
```python
def objective(trial):
    params = {
        "learning_rate": trial.suggest_float("lr", 1e-5, 1e-3, log=True),
        "n_steps": trial.suggest_categorical("n_steps", [256, 512, 1024, 2048]),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128, 256]),
        "gamma": trial.suggest_float("gamma", 0.95, 0.999),
        "ent_coef": trial.suggest_float("ent_coef", 1e-4, 0.1, log=True),
        "clip_range": trial.suggest_float("clip_range", 0.1, 0.4),
        "tcn_channels": trial.suggest_categorical("tcn_channels", [32, 64, 128]),
        "tcn_layers": trial.suggest_int("tcn_layers", 2, 6),
    }
    model = train_model(params)
    sharpe = evaluate_model(model, val_data)
    return sharpe
```

---

## Phase 3: Execution Quality (Weeks 10-16)

### 3.1 Smart Order Routing

Current: Simple TWAP/VWAP with fixed slices
Target: Adaptive execution with market impact minimization

**Implementation**:
```
app/execution/
├── smart_router.py
│   └── SmartOrderRouter
│       ├── estimate_impact(symbol, qty, urgency) -> float
│       ├── select_algo(symbol, qty, market_state) -> str
│       └── route(order) -> list[ChildOrder]
├── impact_model.py
│   └── MarketImpactModel
│       ├── almgren_chriss(qty, vol, spread) -> float  # permanent impact
│       └── temporary_impact(qty, adv) -> float
└── algos/
    ├── twap.py - existing, enhanced
    ├── vwap.py - existing, enhanced
    ├── is_algo.py - Implementation Shortfall minimization
    └── adaptive_algo.py - ML-based adaptive execution
```

### 3.2 Transaction Cost Analysis (TCA)

**Implementation**:
```
app/execution/tca.py
├── compute_slippage(fills, decision_price) -> float
├── compute_market_impact(fills, pre_trade_mid) -> float
├── compute_timing_cost(fills, arrival_price) -> float
└── TCAReport
    ├── by_symbol: dict[str, TCAMetrics]
    ├── by_algo: dict[str, TCAMetrics]
    └── aggregate: TCAMetrics
```

### 3.3 Execution ML Model

Train a model to predict optimal execution parameters:

```python
class ExecutionPredictor:
    """Predicts optimal execution strategy based on market conditions."""

    def __init__(self):
        self.model = LGBMClassifier()  # or neural net

    def features(self, symbol, qty, market_state):
        return [
            qty / market_state["adv"],  # participation rate
            market_state["spread_bps"],
            market_state["volatility"],
            market_state["quote_imbalance"],
            market_state["time_of_day"],  # normalized 0-1
            market_state["days_to_expiry"],  # if options-related
        ]

    def predict(self, symbol, qty, market_state) -> dict:
        """Returns optimal execution parameters."""
        features = self.features(symbol, qty, market_state)
        algo = self.model.predict([features])[0]
        urgency = self.urgency_model.predict([features])[0]
        return {"algo": algo, "urgency": urgency, "max_participation": 0.1}
```

---

## Phase 4: Portfolio Optimization (Weeks 14-20)

### 4.1 Multi-Asset Position Sizing

Current: Single-symbol sizing based on Kelly/risk limits
Target: Portfolio-aware optimization

**Implementation**:
```
app/portfolio/
├── optimizer.py
│   └── PortfolioOptimizer
│       ├── mean_variance(expected_returns, cov_matrix, risk_aversion) -> weights
│       ├── risk_parity(cov_matrix) -> weights
│       ├── max_sharpe(expected_returns, cov_matrix) -> weights
│       └── black_litterman(market_weights, views, confidence) -> weights
├── risk_model.py
│   └── RiskModel
│       ├── compute_covariance(returns, method="ledoit_wolf") -> np.ndarray
│       ├── factor_risk(exposures, factor_cov) -> np.ndarray
│       └── idiosyncratic_risk(residuals) -> np.ndarray
└── constraints.py
    └── PortfolioConstraints
        ├── max_position_pct: float
        ├── sector_limits: dict[str, float]
        ├── factor_neutrality: list[str]  # e.g., ["market", "sector"]
        └── turnover_limit: float
```

### 4.2 Factor Neutralization

Ensure the portfolio is neutral to unwanted factor exposures:

```python
class FactorNeutralizer:
    """Adjusts positions to achieve factor neutrality."""

    def __init__(self, factors=["market", "size", "value", "momentum"]):
        self.factors = factors
        self.factor_returns = {}  # loaded from factor data
        self.factor_loadings = {}  # per-symbol betas

    def compute_exposure(self, positions: dict[str, float]) -> dict[str, float]:
        """Compute portfolio factor exposures."""
        exposures = {f: 0.0 for f in self.factors}
        for symbol, weight in positions.items():
            for factor in self.factors:
                exposures[factor] += weight * self.factor_loadings[symbol][factor]
        return exposures

    def neutralize(self, positions: dict[str, float], target_exposures: dict[str, float] = None):
        """Adjust positions to neutralize factor exposures."""
        target = target_exposures or {f: 0.0 for f in self.factors}
        # Solve optimization: min ||positions - original|| s.t. exposures == target
        # Using cvxpy or scipy.optimize
        ...
```

### 4.3 Dynamic Rebalancing

```python
class RebalanceEngine:
    """Determines when and how to rebalance the portfolio."""

    def __init__(self, threshold_pct=5.0, min_interval_minutes=30):
        self.threshold_pct = threshold_pct
        self.min_interval = timedelta(minutes=min_interval_minutes)
        self.last_rebalance = None

    def should_rebalance(self, current: dict, target: dict) -> bool:
        if self.last_rebalance and datetime.utcnow() - self.last_rebalance < self.min_interval:
            return False
        drift = self.compute_drift(current, target)
        return drift > self.threshold_pct

    def compute_trades(self, current: dict, target: dict, nav: float) -> list[dict]:
        """Compute trades needed to move from current to target weights."""
        trades = []
        for symbol in set(current.keys()) | set(target.keys()):
            current_weight = current.get(symbol, 0.0)
            target_weight = target.get(symbol, 0.0)
            delta_weight = target_weight - current_weight
            if abs(delta_weight) > 0.001:  # 0.1% threshold
                notional = delta_weight * nav
                trades.append({
                    "symbol": symbol,
                    "action": "buy" if delta_weight > 0 else "sell",
                    "notional": abs(notional),
                })
        return trades
```

---

## Phase 5: Advanced Regime Detection (Weeks 18-24)

### 5.1 Hidden Markov Model for Regimes

```python
class RegimeHMM:
    """Hidden Markov Model for market regime detection."""

    def __init__(self, n_regimes=3):
        self.n_regimes = n_regimes
        self.model = GaussianHMM(n_components=n_regimes, covariance_type="full")

    def fit(self, returns: np.ndarray):
        """Fit HMM to historical returns."""
        features = self._compute_features(returns)
        self.model.fit(features)

    def predict_regime(self, returns: np.ndarray) -> int:
        """Predict current regime."""
        features = self._compute_features(returns)
        return self.model.predict(features)[-1]

    def regime_probabilities(self, returns: np.ndarray) -> np.ndarray:
        """Get probability distribution over regimes."""
        features = self._compute_features(returns)
        return self.model.predict_proba(features)[-1]

    def _compute_features(self, returns):
        """Compute features for HMM: [return, volatility, skewness]."""
        vol = pd.Series(returns).rolling(20).std().values
        skew = pd.Series(returns).rolling(20).skew().values
        return np.column_stack([returns, vol, skew])[20:]  # drop NaN rows
```

### 5.2 Regime-Conditional Strategies

```python
class RegimeAwareStrategy:
    """Adapts strategy based on detected market regime."""

    def __init__(self, regime_detector, strategies: dict[int, Strategy]):
        self.regime_detector = regime_detector
        self.strategies = strategies  # regime_id -> Strategy

    def generate_signal(self, market_state: dict) -> dict:
        returns = np.array(market_state["prices"])[1:] / np.array(market_state["prices"])[:-1] - 1
        regime = self.regime_detector.predict_regime(returns)

        # Select strategy for current regime
        strategy = self.strategies.get(regime)
        if strategy is None:
            return {"action": "hold", "regime": regime}

        signal = strategy.generate_signal(market_state)
        signal["regime"] = regime
        return signal
```

### 5.3 Changepoint Detection

```python
class ChangepointDetector:
    """Detects structural breaks in time series."""

    def __init__(self, method="pelt", min_segment=20):
        self.method = method
        self.min_segment = min_segment

    def detect(self, series: np.ndarray) -> list[int]:
        """Returns indices of detected changepoints."""
        import ruptures as rpt

        if self.method == "pelt":
            algo = rpt.Pelt(model="rbf", min_size=self.min_segment)
        elif self.method == "binseg":
            algo = rpt.Binseg(model="rbf", min_size=self.min_segment)
        else:
            algo = rpt.Window(model="rbf", min_size=self.min_segment)

        algo.fit(series)
        return algo.predict(pen=10)

    def recent_changepoint(self, series: np.ndarray, lookback=100) -> bool:
        """Check if there was a recent changepoint."""
        cps = self.detect(series[-lookback:])
        return len(cps) > 1 and cps[-2] > lookback - 20
```

---

## Implementation Priority & Timeline

| Phase | Weeks | Priority | Expected Impact on Sharpe | Status |
|-------|-------|----------|---------------------------|--------|
| 1.1 Microstructure features | 1-3 | Critical | +0.2 to +0.4 | **DONE** |
| 1.2 Technical indicators | 2-4 | High | +0.1 to +0.2 | **DONE** (22 indicators) |
| 1.3 Multi-timeframe | 3-5 | High | +0.1 to +0.3 | **DONE** (5m→15m→1h) |
| 1.5 Regime detection | 4-6 | High | +0.1 to +0.2 | **DONE** (8 features) |
| 2.1 TCN/Transformer | 6-10 | Medium-High | +0.2 to +0.4 | **DONE** (TCN extractor) |
| 2.2 Ensemble | 8-12 | Medium | +0.1 to +0.3 | **DONE** (LightGBM, meta-learning) |
| 2.3 HPO | 10-12 | Medium | +0.1 to +0.2 | **DONE** (Optuna integration) |
| 3.1-3.3 Execution | 10-16 | Medium | +0.1 to +0.2 (cost reduction) | **DONE** (SmartRouter, TCA) |
| 4.1-4.3 Portfolio | 14-20 | Medium | +0.1 to +0.2 | Pending |
| 5.1-5.3 Advanced regime | 18-24 | Low-Medium | +0.05 to +0.15 | Pending |

**Cumulative expected Sharpe improvement: +0.8 to +1.5** (from current ~0.3-0.5 to target 1.0-2.0)

---

## Quick Wins (Week 1)

These can be implemented immediately with minimal effort:

1. **Add ATR-based position sizing** - normalize position size by volatility
2. **Add VWAP deviation feature** - simple to compute, high signal value
3. **Add bid-ask spread to features** - already available from Alpaca
4. **Expand indicator set** - add Bollinger Bands, Stochastic, OBV (use `ta` library)
5. **Multi-timeframe SMA** - add 15m and 1h SMA to 5m features

```bash
pip install ta  # Technical analysis library with 100+ indicators
pip install hmmlearn  # Hidden Markov Models
pip install ruptures  # Changepoint detection
pip install cvxpy  # Portfolio optimization
pip install optuna  # Hyperparameter optimization
pip install lightgbm  # Gradient boosting
```

---

## Metrics for Success

| Metric | Current (Est.) | Target (6mo) | Target (12mo) |
|--------|----------------|--------------|---------------|
| Sharpe Ratio | 0.3-0.5 | 1.0-1.5 | 1.5-2.0 |
| Win Rate | 45-50% | 52-55% | 55-60% |
| Profit Factor | 1.0-1.2 | 1.3-1.5 | 1.5-2.0 |
| Max Drawdown | 15-25% | 10-15% | 8-12% |
| Avg Trade Duration | 2-4 hours | 30min-2hr | Adaptive |
| Execution Slippage | Unknown | <5bps | <3bps |

---

## Risk Considerations

1. **Overfitting**: More features and complex models increase overfitting risk. Use strict train/val/test splits, walk-forward validation, and regularization.

2. **Data snooping**: Track all experiments in MLflow/W&B. Don't cherry-pick results.

3. **Regime changes**: Models trained on 2020-2024 data may not work in different regimes. Use regime-conditional strategies.

4. **Execution gap**: Backtest results rarely match live performance. Build realistic execution simulation.

5. **Complexity cost**: More complex systems have more failure modes. Maintain observability (existing Prometheus/Grafana is good).

---

## Implementation Progress (2026-02-05)

### Completed

**Phase 1: Feature Engineering**
- `app/learning/indicators.py` - 22 extended technical indicators:
  - Volatility: ATR, Bollinger Bands, historical volatility, Keltner channels
  - Momentum: Stochastic K/D, Williams %R, CCI, ROC, momentum
  - Volume: OBV, MFI, VWAP deviation
  - Trend: Donchian position, Parabolic SAR, Supertrend, Ichimoku (5 components)
  - Mean reversion: Hurst exponent
- `app/learning/regime.py` - 8 regime detection features:
  - Volatility regime (0/1/2), trend regime (0/1/2), liquidity regime (0/1/2)
  - Volatility percentile, trend strength, Hurst exponent
  - Volatility ratio, realized volatility
- `app/learning/multi_timeframe.py` - 9 MTF features:
  - 5m, 15m, 1h timeframe aggregation
  - Per-timeframe: return, volatility, volume ratio
- `app/learning/features.py` - Integrated all modules into observation builder

**Phase 2: Model Architecture**
- `app/learning/networks/tcn.py` - TCN feature extractor for SB3 PPO:
  - TemporalBlock with dilated causal convolutions
  - TCN module with exponential dilation
  - TCNExtractor compatible with Stable Baselines3
  - `create_tcn_policy_kwargs()` helper function
- `app/learning/train_rl.py` - TCN integration when `learning.tcn.enabled=true`
- `app/learning/ensemble/` - Ensemble methods:
  - `base.py`: BasePredictor, ReturnPredictor, DirectionPredictor, SignalPredictor
  - `lgbm_predictor.py`: LGBMReturnPredictor, LGBMDirectionPredictor, LGBMSignalPredictor
  - `ensemble.py`: EnsemblePredictor with meta-learning, StackingEnsemble
- `app/learning/hpo/` - Hyperparameter optimization:
  - `search_space.py`: Optuna search spaces for PPO, TCN, reward params
  - `optimize.py`: HPOObjective, run_hpo(), apply_best_params()

**Phase 3: Execution Quality**
- `app/execution/smart_router.py` - Smart order routing:
  - SmartOrderRouter with automatic algo selection
  - Almgren-Chriss market impact model
  - almgren_chriss_optimal_trajectory() for IS minimization
- `app/execution/tca.py` - Transaction cost analysis:
  - TCAAnalyzer with slippage, impact, timing cost metrics
  - TCAReport with by-symbol, by-algo, by-venue breakdowns
  - compute_vwap_slippage() utility

**Configuration (config/config.yaml)**
```yaml
learning:
  features:
    include_extended_indicators: true
    include_regime_features: true
    include_mtf_features: true
  tcn:
    enabled: true
    features_dim: 64
    num_layers: 3
    kernel_size: 3
    dropout: 0.1
  training:
    timesteps: 50000
```

**Total new observation size**: ~190 dimensions (up from ~150)

### Next Steps
1. Model training in progress (50,000 timesteps with TCN)
2. Monitor performance metrics in Grafana after training completes
3. Implement Phase 4 (Portfolio optimization) when baseline is stable
4. Run HPO to tune hyperparameters (requires `pip install optuna`)

---

## Conclusion

The plan prioritizes **feature engineering** because it provides the highest ROI with the lowest
risk of breaking existing infrastructure. Model architecture improvements come second, followed
by execution and portfolio optimization.

The existing infrastructure (orchestrator, multi-broker routing, risk management, monitoring)
is solid and can support these upgrades without major refactoring. The main work is in the
`app/learning/` and `app/strategies/` directories.

**Current status**: Phases 1, 2, and 3 are complete. Model training with 50k timesteps in progress.
Next: Phase 4 (portfolio optimization) and Phase 5 (advanced regime detection).
