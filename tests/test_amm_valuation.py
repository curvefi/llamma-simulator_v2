import unittest
from datetime import datetime
from unittest.mock import patch

from simulator.amm.intitial_liquidity import ConstantInitialLiquidity
from simulator.amm.lending_amm import LendingAMM
from simulator.amm.simulator import Simulator


class AmmValuationTest(unittest.TestCase):
    def test_empty_bands_need_no_price_calculation(self):
        amm = LendingAMM(100.0, 100, 0.003)
        with patch.object(amm, "p_top", side_effect=AssertionError("Empty band needs no price")):
            for band in (-500, 0, 499):
                self.assertEqual(amm.get_x_down(band), 0)
                self.assertEqual(amm.get_y_up(band), 0)

    def test_total_valuation_includes_manually_funded_bands(self):
        amm = LendingAMM(100.0, 100, 0.003)
        amm.deposit_nrange(1.0, 99.0, 4)
        amm.bands_x[-10] = 0.5
        amm.bands_y[20] = 0.5
        funded = sorted({-10, 20, *range(amm.min_band, amm.max_band + 1)})
        self.assertEqual(amm.get_all_x(), sum(amm.get_x_down(n) for n in funded))
        self.assertEqual(amm.get_all_y(), sum(amm.get_y_up(n) for n in funded))

    def test_date_formatting_is_only_needed_for_logging(self):
        simulator = Simulator.__new__(Simulator)
        simulator.initial_liquidity_class = ConstantInitialLiquidity
        simulator.external_fee = 0.0005
        simulator.log_enabled = False
        simulator.verbose = False
        candles = [[1700000040, 100.0, 101.0, 99.0, 100.5, 1.0]]
        args = (100, 0.003, candles, [100.0], 4)

        with patch("simulator.amm.simulator.datetime") as date:
            date.fromtimestamp.side_effect = AssertionError("Logging is disabled")
            expected = simulator.calculate_loss(*args)
            simulator.verbose = True
            self.assertEqual(simulator.calculate_loss(*args), expected)

        simulator.log_enabled = True
        with patch("simulator.amm.simulator.logger") as logger:
            self.assertEqual(simulator.calculate_loss(*args), expected)
        label = datetime.fromtimestamp(candles[0][0]).strftime("%Y/%m/%d %H:%M")
        self.assertTrue(logger.info.call_args_list[0].args[0].startswith(f"Current x total for {label}:"))


if __name__ == "__main__":
    unittest.main()
