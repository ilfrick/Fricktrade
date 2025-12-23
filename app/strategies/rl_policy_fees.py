from __future__ import annotations

from app.strategies.rl_policy import RLPolicyStrategy


class FeeAwareRLPolicyStrategy(RLPolicyStrategy):
    def __init__(
        self,
        model_path: str,
        window_size: int = 50,
        device: str = "auto",
        feature_config: dict | None = None,
        broker_fees: dict | None = None,
        fee_guard: dict | None = None,
        risk_cfg: dict | None = None,
    ):
        super().__init__(model_path, window_size=window_size, device=device, feature_config=feature_config)
        self.broker_fees = broker_fees or {}
        self.fee_guard = fee_guard or {}
        self.risk_cfg = risk_cfg or {}

    def generate_signal(self, market_state: dict) -> dict:
        signal = super().generate_signal(market_state)
        action = signal.get("action")
        if action not in ("buy", "sell"):
            return signal
        if not self._passes_fee_guard(market_state):
            return {"action": "hold"}
        return signal

    def _passes_fee_guard(self, market_state: dict) -> bool:
        min_edge_pct = float(self.fee_guard.get("min_edge_pct", 0.02))
        edge_multiplier = float(self.fee_guard.get("edge_multiplier", 1.0))
        min_notional = float(self.fee_guard.get("min_notional", 50.0))
        est_notional = self._estimate_notional(market_state)
        if est_notional < min_notional:
            return False

        total_cost_pct = self._estimate_cost_pct(market_state, est_notional)
        edge_threshold = max(min_edge_pct, total_cost_pct * edge_multiplier)
        momentum = self._recent_momentum_pct(market_state)
        return abs(momentum) >= edge_threshold

    def _estimate_notional(self, market_state: dict) -> float:
        portfolio = market_state.get("portfolio", {}) or {}
        equity = float(portfolio.get("equity", 0.0) or 0.0)
        max_pos_pct = float(self.risk_cfg.get("max_position_size_pct", 0.0))
        if equity > 0.0 and max_pos_pct > 0.0:
            return equity * (max_pos_pct / 100.0)
        last_price = market_state.get("last_price")
        if last_price:
            return float(last_price)
        return float(self.fee_guard.get("min_notional", 50.0))

    def _estimate_cost_pct(self, market_state: dict, notional: float) -> float:
        if notional <= 0:
            return 0.0
        commission_pct = float(self.broker_fees.get("commission_pct", 0.0))
        per_trade_fee = float(self.broker_fees.get("per_trade_fee", 0.0))
        per_share_fee = float(self.broker_fees.get("per_share_fee", 0.0))
        min_fee = float(self.broker_fees.get("min_fee", 0.0))
        spread_pct = float(
            market_state.get("spread_pct", self.broker_fees.get("spread_pct", 0.0)) or 0.0
        )
        last_price = market_state.get("last_price")
        qty_est = notional / last_price if last_price else 0.0
        fee_value = max(min_fee, per_trade_fee + per_share_fee * qty_est)
        fee_pct = (fee_value / notional) * 100.0
        return commission_pct + fee_pct + spread_pct

    @staticmethod
    def _recent_momentum_pct(market_state: dict) -> float:
        prices = market_state.get("prices", []) or []
        if len(prices) < 2:
            return 0.0
        prev = float(prices[-2])
        curr = float(prices[-1])
        if prev == 0.0:
            return 0.0
        return (curr - prev) / prev * 100.0
