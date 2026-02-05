# Fricktrade Improvement Roadmap

Generated: 2026-02-05

## Executive Summary

This roadmap identifies gaps, optimization opportunities, and practical improvements for the Fricktrade trading system. Prioritized by impact and implementation effort.

---

## Phase 1: Critical Performance (Week 1-2)

### 1.1 Feature Computation Caching - HIGHEST PRIORITY

**Problem**: Feature vectors (indicators, regime, multi-timeframe) are computed on every bar without caching intermediate results. With 7 strategies × 50 symbols = 350+ feature computations per cycle.

**Location**: `app/learning/features.py`, `app/learning/indicators.py`

**Solution**:
- Implement rolling cache with timestamp-based invalidation
- Cache features at bar level (only recompute on new bar)
- Use deque for O(1) incremental updates
- Cache regime state separately (lower frequency)

**Impact**: 40-60% reduction in main loop latency

---

### 1.2 Market Cache Staleness Fix - CRITICAL BUG

**Problem**: Race condition where stale data might be served. `max_age_multiplier=2` with 5m bars accepts 10-minute stale data.

**Location**: `app/data/market_cache.py:80-90`

**Solution**:
- Add explicit timestamp freshness check at trade time
- Compare cache timestamp to current UTC
- Log warning when data older than 2× interval
- Fall back to live fetch if stale

**Impact**: Prevents slippage from stale quotes

---

### 1.3 Latency Profiling in Trading Loop

**Problem**: No per-component latency tracking. Can't identify bottlenecks.

**Location**: `app/agents/trader.py:2361-2396`

**Solution**:
```python
with latency_tracker("market_data_fetch"):
    self._prepare_market_data(...)
with latency_tracker("feature_compute"):
    features = compute_features(...)
with latency_tracker("model_inference"):
    action = policy.predict(...)
```

**Impact**: Enable bottleneck identification, 20-30% latency reduction possible

---

## Phase 2: Execution Quality (Week 2-3)

### 2.1 Wire Almgren-Chriss into SmartOrderRouter

**Problem**: `almgren_chriss_optimal_trajectory()` is implemented but never called.

**Location**: `app/execution/smart_router.py:204-264`

**Solution**:
- Call A-C trajectory in `_select_algo()` for large orders (>5% ADV)
- Use A-C as baseline for risk-averse execution
- Add to TCA report as benchmark

**Impact**: 10-20 bps savings on block trades

---

### 2.2 Adaptive Execution Slicing

**Problem**: Static slice parameters. No adaptation for volatility, volume spikes, venue liquidity.

**Location**: `app/execution/smart_router.py:165-183`

**Solution**:
- Regime-based slice duration (high vol → shorter slices)
- Dynamic participation scaling based on volume velocity
- Time-of-day volume profile integration

**Impact**: 10-20 bps savings on large orders

---

### 2.3 Implementation Shortfall Minimization (IS-Min) Algo

**Problem**: Only TWAP, VWAP, POV, Market algos. No IS-minimization.

**Solution**:
- IS-Min algo minimizing `permanent_impact + volatility_penalty + urgency_cost`
- Optimal stopping based on execution progress
- Integration with RegimeHMM volatility

**Impact**: 5-15 bps on average order

---

## Phase 3: Risk Management (Week 3-4)

### 3.1 Sector Concentration Limits

**Problem**: Risk manager checks daily loss but not sector concentration. 7 tech stocks @ 2% = 14% undetected.

**Location**: `app/risk/manager.py`

**Solution**:
- Add sector concentration limits (max 20% per sector)
- Theme detection (meme stocks, crypto-proxies)
- Dynamic VaR with realized correlation

**Impact**: Prevents drawdowns from sector crashes

---

### 3.2 Velocity-Based Circuit Breaker

**Problem**: `should_circuit_break()` only checks absolute drawdown, no velocity.

**Location**: `app/risk/manager.py:102-115`

**Solution**:
- Velocity constraint: if drawdown > X% in < Y minutes, halt
- Hysteresis: require 50% recovery to re-enable
- Per-symbol circuit breaks

**Impact**: Reduced tail risk

---

### 3.3 Position Concentration Alert

**Problem**: No warning on correlated positions from stat arb pairs strategy.

**Solution**:
- Correlation-based risk aggregation
- Alert when effective positions exceed threshold
- Dynamic position limits based on correlation

---

## Phase 4: Model Improvements (Week 4-5)

### 4.1 Ensemble Confidence Calibration

**Problem**: Ensemble predictions have no calibration. May be miscalibrated under regime shifts.

**Location**: `app/learning/ensemble/ensemble.py:60-100`

**Solution**:
- Add conformal prediction for uncertainty quantification
- Track prediction confidence separately from accuracy
- Separate weighting for direction vs. magnitude

**Impact**: Better risk sizing, fewer false signals

---

### 4.2 TCN Sequence Length Adaptation

**Problem**: Fixed sequence length (20 bars) regardless of market conditions.

**Location**: `app/learning/networks/tcn.py`

**Solution**:
- Add temporal position embeddings
- Regime-conditional sequence length (low vol: 15, high vol: 30)
- Cache TCN outputs

**Impact**: 3-5% improvement in policy accuracy

---

### 4.3 Event-Aware Regime Detection

**Problem**: RegimeHMM doesn't flag market anomalies (halts, earnings, Fed).

**Location**: `app/learning/regime_hmm.py`

**Solution**:
- External event feed integration
- Market halt detection from broker feeds
- Event-triggered regime override

**Impact**: Avoid losses during unexpected events

---

## Phase 5: Data & Features (Week 5-6)

### 5.1 Multi-Asset Correlation Features

**Problem**: Features computed per-symbol in isolation. No cross-asset features.

**Location**: `app/learning/features.py`

**Solution**:
- Rolling correlation matrix (sector-level, SPY)
- Pair spread mean-reversion (Kalman filter)
- Beta-adjusted returns vs. SPY

**Impact**: 5-10% improvement in pair trading

---

### 5.2 Observation Encoding Optimization

**Problem**: Observation array rebuilt from scratch every step.

**Location**: `app/learning/env.py`, `app/learning/features.py:build_observation()`

**Solution**:
- Feature buffer tracking changed components
- Pre-allocate numpy arrays
- Numba JIT for hot paths (RSI, EMA, MACD)

**Impact**: 25-35% encoding speedup

---

## Quick Wins (<1 hour each)

| Task | Location | Fix |
|------|----------|-----|
| Fix max drawdown TODO in HPO | `hpo/optimize.py:263` | Add rolling max equity tracking |
| Wire ensemble confidence to orchestrator | `orchestrator.py:227` | Scale strategy weights by confidence |
| Add sector concentration check | `risk/manager.py` | Check sector exposure on each trade |
| Add conformal prediction interface | `ensemble/ensemble.py` | Return prediction intervals |

---

## Code Quality Issues

### ExecutionEngine Oversimplified
**Location**: `app/execution/executor.py`

Only 30 lines, handles "buy", "sell", "exit" with no limit orders, partial fills, or retry logic. Should leverage SmartOrderRouter.

### Order Queue Lock Contention
**Location**: `app/execution/order_queue.py:85`

Single lock with many readers. Consider read-write lock or lock-free queue.

---

## Summary Table

| Priority | Improvement | Impact | Effort |
|----------|-------------|--------|--------|
| 1 | Feature caching | 40-60% latency | Medium |
| 2 | Cache staleness fix | Prevent slippage | Low |
| 3 | Almgren-Chriss wiring | 10-20 bps | Low |
| 4 | Sector concentration | Risk control | Low |
| 5 | Latency profiling | Enable optimization | Low |
| 6 | Ensemble calibration | Better sizing | Medium |
| 7 | Adaptive execution | 10-20 bps | Medium |
| 8 | IS-Min algo | 5-15 bps | High |
| 9 | Event-aware regime | Avoid surprises | Medium |
| 10 | Multi-asset features | 5-10% pairs | Medium |

---

## Implementation Notes

- All improvements should maintain backward compatibility
- Add feature flags for new components
- Test in paper trading before live deployment
- Monitor latency and P&L impact after each change
