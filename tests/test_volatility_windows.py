import unittest
from unittest.mock import patch

from simulator.amm.intitial_liquidity import ConstantInitialLiquidity
from simulator.amm.price_history_loader import VolatilityPriceHistoryLoader
from simulator.amm.price_oracle import EmaPriceOracle
from simulator.amm.simulator import SimulatorV2, get_loss_rate_v2
from simulator.settings import Pair


def loss_rate(position_start):
    candles = [
        [0, 1, 1, 1, 1, 0],
        [60, 1, 1, 1, 1, 0],
        [120, 1, 1, 1, 1, 0],
        [180, 1, 1, 0.9, 0.9, 0],
    ]
    with (
        patch("simulator.amm.price_history_loader.BinanceImporter.load", return_value=candles),
        patch("simulator.amm.simulator.random.random", return_value=position_start),
    ):
        return get_loss_rate_v2(
            ConstantInitialLiquidity,
            VolatilityPriceHistoryLoader(Pair.ETHUSDT),
            EmaPriceOracle(600),
            0.0005,
            100,
            0.002,
            4,
            samples=1,
            n_top_samples=1,
            min_loan_duration=1 / 1440,
            max_loan_duration=1 / 1440,
            use_threading=False,
        )


class VolatilityWindowTest(unittest.TestCase):
    def test_flat_window_has_no_loss(self):
        self.assertEqual(loss_rate(0), 0)

    def test_failed_replay_is_not_replaced_by_zero(self):
        with patch.object(SimulatorV2, "calculate_loss", side_effect=ValueError("invalid baseline")):
            with self.assertRaisesRegex(ValueError, "invalid baseline"):
                loss_rate(0.75)


if __name__ == "__main__":
    unittest.main()
