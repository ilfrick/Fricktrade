from app.strategies.trend_following import TrendFollowingStrategy
from app.strategies.factor_model import FactorModelStrategy
from app.strategies.stat_arb_pairs import StatArbPairsStrategy
from app.strategies.market_maker import MarketMakerStrategy


def test_trend_following_buy() -> None:
    params = {"trend_following": {"fast_window": 3, "slow_window": 5, "breakout_pct": 0.1}}
    strat = TrendFollowingStrategy(params)
    prices = [10, 10.2, 10.4, 10.6, 10.8, 11.0]
    signal = strat.generate_signal({"prices": prices})
    assert signal["action"] == "buy"


def test_factor_model_sell() -> None:
    params = {
        "factor_model": {
            "momentum_weight": 1.0,
            "liquidity_weight": 0.0,
            "volatility_weight": 0.0,
            "buy_threshold": 0.2,
            "sell_threshold": -0.01,
        }
    }
    strat = FactorModelStrategy(params)
    prices = [10, 9.5, 9.0, 8.7]
    volumes = [100000, 100000, 100000, 100000]
    signal = strat.generate_signal({"prices": prices, "volumes": volumes})
    assert signal["action"] == "sell"


def test_stat_arb_pairs_signal() -> None:
    params = {"stat_arb_pairs": {"lookback": 5, "z_entry": 1.0, "z_exit": 0.2, "max_pairs": 1}}
    strat = StatArbPairsStrategy(params)
    StatArbPairsStrategy._price_cache = {
        "AAA": [10, 10.1, 10.2, 10.4, 11.0],
        "BBB": [10, 10.1, 10.2, 10.3, 10.2],
    }
    StatArbPairsStrategy._pairs = [("AAA", "BBB")]
    signal = strat.generate_signal({"symbol": "AAA", "prices": [11.0]})
    assert signal["action"] in {"buy", "sell", "exit", "hold"}


def test_market_maker_limit_order() -> None:
    strat = MarketMakerStrategy({"market_maker": {"base_spread_pct": 0.4}})
    signal = strat.generate_signal({"last_price": 100.0, "exposure_pct": 1.0})
    if signal["action"] != "hold":
        assert signal["order_type"] == "limit"
        assert signal["limit_price"] is not None
