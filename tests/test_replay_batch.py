import unittest

import numpy as np

from simulator.amm.intitial_liquidity import ConstantInitialLiquidity
from simulator.amm.lending_amm import LendingAMM, OracleState
from simulator.amm.simulator import Simulator, replay_batch


class ReplayBatchTest(unittest.TestCase):
    def setUp(self):
        self.simulator = Simulator.__new__(Simulator)
        self.simulator.initial_liquidity_class = ConstantInitialLiquidity
        self.simulator.external_fee = 0.007
        self.simulator.log_enabled = self.simulator.verbose = False
        prices = np.array([1.0, 1.01, 0.99, 1.02, 1.0, 0.98])
        self.points = np.column_stack(
            (1700000000 + np.arange(6) * 60, prices, prices * 1.04, prices * 0.96, prices, np.zeros(6), prices)
        )
        self.records = np.array(
            [[100, 0.001, 0, 4, 4, 0.0005, 0.25, 0], [200, 0.002, 2, 6, 4, 0.002, 0.25, 0]], dtype=np.float64
        )

    def test_batch_matches_individual_windows_without_changing_settings(self):
        reference = Simulator.__new__(Simulator)
        reference.initial_liquidity_class = ConstantInitialLiquidity
        reference.log_enabled = reference.verbose = False
        expected = []
        for A, fee, start, end, bands, external_fee, multiplier, _ in self.records:
            reference.external_fee = float(external_fee)
            start, end = int(start), int(end)
            expected.append(
                reference.calculate_loss(
                    int(A),
                    float(fee),
                    self.points[start:end, :6],
                    self.points[start:end, 6],
                    int(bands),
                    float(multiplier),
                )
            )
        actual = replay_batch(self.simulator, self.points, self.records)
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(self.simulator.external_fee, 0.007)
        np.testing.assert_array_equal(replay_batch(self.simulator, self.points, self.records[::-1]), actual[::-1])

    def test_failed_batch_preserves_settings(self):
        self.points[0, 6] = -1.0
        with self.assertRaises(ValueError):
            replay_batch(self.simulator, self.points, self.records)
        self.assertEqual(self.simulator.external_fee, 0.007)

    def test_empty_batch_and_invalid_shapes(self):
        self.assertEqual(replay_batch(self.simulator, self.points, self.records[:0]).shape, (0,))
        for points, records in ((self.points[:, 0], self.records), (self.points, self.records[0])):
            with self.assertRaises(ValueError):
                replay_batch(self.simulator, points, records)
        self.assertEqual(self.simulator.external_fee, 0.007)

    def test_reset_clears_previous_position_and_reuses_storage(self):
        amm = LendingAMM(1.0, 100, 0.001, oracle_state=OracleState.initial(1.0, 0))
        amm.deposit_nrange(1.0, 1.0, 4)
        amm.trade_to_price(1.04)
        amm.bands_x[501] = 2.0
        bands_x, bands_y = amm.bands_x, amm.bands_y
        state = OracleState.initial(1.2, 60)
        amm.reset(1.2, 200, 0.002, oracle_state=state)
        fresh = LendingAMM(1.2, 200, 0.002, oracle_state=state)
        self.assertIs(amm.bands_x, bands_x)
        self.assertIs(amm.bands_y, bands_y)
        self.assertEqual(dict(amm.bands_x), dict(fresh.bands_x))
        self.assertEqual(dict(amm.bands_y), dict(fresh.bands_y))
        self.assertEqual(amm.oracle_state(), fresh.oracle_state())
        self.assertEqual(
            (amm.min_band, amm.max_band, amm.active_band), (fresh.min_band, fresh.max_band, fresh.active_band)
        )


if __name__ == "__main__":
    unittest.main()
