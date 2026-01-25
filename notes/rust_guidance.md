# Rust candidates guidance (Fricktrade)

Short answer: yes, but only for a few tight, CPU-bound, Python-level loops where data can stay in contiguous arrays and you can avoid pandas/NumPy conversions. Otherwise, NumPy/Numba/Polars will beat a Rust FFI hop.

Good Rust candidates if profiling shows them hot:
- Backtest inner loop in app/backtest/agent_engine.py (if still Python-loop heavy after vectorization).
- AI filter feature extraction in app/data/ai_filter.py (_build_training_data, _latest_features, _target_value) if you can pass raw arrays and keep them in Rust.
- Signal metrics in app/utils/signal_features.py (simple loops over large arrays).
- Market cache serialization in app/data/market_cache.py only if JSON encode/decode dominates and you are willing to replace the format.

Less compelling for Rust:
- Anything already in pandas/NumPy (you pay conversion overhead).
- GPU-bound pieces (RL, Keras, SB3) -- Rust will not help there.

Offer: run profiler to identify hotspots or prototype a Rust/PyO3 module for a hot loop.
