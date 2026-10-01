import unittest
from unittest.mock import Mock

from simulator.amm.intitial_liquidity import ConstantInitialLiquidity
from simulator.amm.price_history_loader import GenericPriceHistoryLoader
from simulator.amm.price_oracle import EmaPriceOracle
from simulator.amm.simulator import Simulator


class OracleTimingTest(unittest.TestCase):
    def test_close_is_available_at_bar_end_including_gaps(self):
        bars = [[t, 999, 999, 1, c, 0] for t, c in [(0, 10), (60, 20), (240, 40), (300, 80)]]
        oracle = EmaPriceOracle(60)
        self.assertEqual(oracle.calculate_oracle_prices(bars), [None, 10, 15, 36.875])
        bars[2][4] = 400
        self.assertEqual(oracle.calculate_oracle_prices(bars)[:3], [None, 10, 15])

    def test_consumer_preserves_alignment_after_warmup(self):
        bars = [[t, 1, 1, 1, 1, 0] for t in (0, 60, 120)]
        loader = Mock()
        loader.load_prices.return_value = bars
        simulator = Simulator(ConstantInitialLiquidity, loader, EmaPriceOracle(60))
        self.assertEqual(simulator.prices, bars[1:])
        self.assertEqual(simulator.oracle_prices, [1, 1])
        loader.load_prices.return_value = bars[:1]
        with self.assertRaisesRegex(ValueError, "No complete causal"):
            Simulator(ConstantInitialLiquidity, loader, EmaPriceOracle(60))

    def test_overlapping_bars_are_rejected(self):
        bar = [0, 1, 1, 1, 1, 0]
        with self.assertRaisesRegex(ValueError, "non-overlapping"):
            EmaPriceOracle(60).calculate_oracle_prices([bar, bar])

    def test_reverse_history_has_no_duplicate_boundary(self):
        loader = GenericPriceHistoryLoader.__new__(GenericPriceHistoryLoader)
        loader.pair, loader.add_reverse = None, True
        loader.importer = Mock()
        loader.importer.load.return_value = [[t, 1, 1, 1, 1, 0] for t in (0, 60, 60, 120)]
        self.assertEqual([row[0] for row in loader.load_prices()], [0, 60, 120, 180, 240])


if __name__ == "__main__":
    unittest.main()
