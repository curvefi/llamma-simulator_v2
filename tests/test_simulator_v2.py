import multiprocessing
import unittest

from simulator.amm.intitial_liquidity import ConstantInitialLiquidity
from simulator.amm.price_history_loader import VolatilityPriceHistoryLoader
from simulator.amm.simulator import SimulatorV2


def make_simulator():
    # Real V2 rescaling, with fixed local observations and no network loading.
    prices = []
    for i in range(60):
        price = 100 + 0.05 * i if i < 30 else 100 - 0.1 * (i - 30)
        prices.append([1700000040 + 60 * i, price, price + 0.01, price - 0.01, price, 1.0])
    loader = VolatilityPriceHistoryLoader.__new__(VolatilityPriceHistoryLoader)
    loader.prices = prices
    loader.period = 1 / 48
    loader.max_drawdown = 0.05
    loader.drawdown_index = 0.9
    simulator = SimulatorV2.__new__(SimulatorV2)
    simulator.initial_liquidity_class = ConstantInitialLiquidity
    simulator.price_history_loader = loader
    simulator.prices = prices
    simulator.oracle_prices = [100.0] * len(prices)
    simulator.external_fee = 0.0005
    simulator.log_enabled = simulator.verbose = False
    return simulator


def arguments(start):
    return dict(
        A=100,
        fee=0.003,
        position_start=start,
        position_period=0.5,
        initial_liquidity_range=4,
        dynamic_fee_multiplier=0.25,
        position_shift=0,
    )


class SimulatorV2Test(unittest.TestCase):
    def test_worker_entry_filters_excluded_window(self):
        simulator = make_simulator()
        kwargs = arguments(0)
        is_down, _ = simulator.price_history_loader.change_period(simulator.prices[:30])
        self.assertFalse(is_down)
        self.assertEqual(simulator.single_run_v2(**kwargs), 0)
        self.assertNotEqual(simulator.single_run(**kwargs), 0)
        self.assertEqual(simulator.single_run_v2_kw(kwargs), 0)

    def test_worker_entry_uses_transformed_prices(self):
        simulator = make_simulator()
        kwargs = arguments(0.5)
        original = simulator.prices[30:]
        is_down, transformed = simulator.price_history_loader.change_period(original)
        self.assertTrue(is_down)
        self.assertNotEqual([row[1:5] for row in original], [list(row[1:5]) for row in transformed])
        expected = simulator.single_run_v2(**kwargs)
        self.assertNotEqual(simulator.single_run(**kwargs), expected)
        self.assertEqual(simulator.single_run_v2_kw(kwargs), expected)

    def test_spawn_pool_matches_ordered_v2_results(self):
        simulator = make_simulator()
        tasks = [arguments(start) for start in (0.5, 0, 0.5, 0)]
        tasks[2].update(A=50, fee=0.001, dynamic_fee_multiplier=0)
        expected = [simulator.single_run_v2(**kwargs) for kwargs in tasks]
        self.assertNotEqual(expected, [simulator.single_run(**kwargs) for kwargs in tasks])
        with multiprocessing.get_context("spawn").Pool(2) as pool:
            actual = pool.map_async(simulator.single_run_v2_kw, tasks).get(timeout=30)
        self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
