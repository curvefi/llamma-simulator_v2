import unittest
from decimal import Decimal, localcontext
from math import isfinite
from unittest.mock import Mock, patch

from simulator.amm.intitial_liquidity import ConstantInitialLiquidity
from simulator.amm.lending_amm import LendingAMM, fee_multiplier
from simulator.amm.simulator import Simulator


def position(fee, memory=0, distance=0):
    amm = LendingAMM(1, 50, fee, distance)
    amm.set_p_oracle(1, 0)
    amm.old_dfee = memory
    amm.min_band = amm.max_band = amm.active_band = 0
    amm.bands_x[0] = amm.bands_y[0] = 1
    return amm


def invariant(amm, n):
    # Independent high-precision solution of the band invariant.
    with localcontext() as context:
        context.prec = 60
        a = Decimal(amm.A)
        p = Decimal(str(amm.p_oracle))
        top = Decimal(str(amm.p_base)) * ((a - 1) / a) ** n
        x, y = Decimal(str(amm.bands_x[n])), Decimal(str(amm.bands_y[n]))
        b = top / p * (a - 1) * x + p * p / top * a * y
        y0 = (b + (b * b + 4 * p * a * x * y).sqrt()) / (2 * p * a)
        f, g = y0 * p * p / top * a, y0 * top / p * (a - 1)
        return f, g, (f + x) * (g + y)


class GrossInputFeeTest(unittest.TestCase):
    def test_partial_band_gross_input_and_output(self):
        for up in (True, False):
            for base, memory in ((0, 0), (0.1, 0), (0.003, 0.2)):
                with self.subTest(up=up, base=base, memory=memory), localcontext() as context:
                    context.prec = 60
                    amm = position(base, memory)
                    fee = Decimal(str(max(base, memory)))
                    f, g, inv = invariant(amm, 0)
                    current = (f + 1) / (g + 1)
                    boundary = inv / g**2 if up else f**2 / inv
                    target = (current + boundary) / 2
                    external = target / (1 - fee) if up else target * (1 - fee)
                    x, y = (inv * target).sqrt() - f, (inv / target).sqrt() - g
                    if up:
                        x = 1 + (x - 1) / (1 - fee)
                    else:
                        y = 1 + (y - 1) / (1 - fee)
                    dx, dy = amm.trade_to_price(float(external))
                    self.assertAlmostEqual(amm.bands_x[0], float(x), delta=1e-11)
                    self.assertAlmostEqual(amm.bands_y[0], float(y), delta=1e-11)
                    self.assertAlmostEqual(dx, float(x - 1), delta=1e-11)
                    self.assertAlmostEqual(dy, float(y - 1), delta=1e-11)

    def test_full_bands_retain_gross_input(self):
        for up in (True, False):
            for base, memory, distance in ((0, 0, 0), (0.1, 0, 0), (0.003, 0.02, 0.25)):
                with self.subTest(up=up, base=base, memory=memory, distance=distance):
                    amm = position(base, memory, distance)
                    amm.min_band, amm.max_band = -1, 1
                    for n in (-1, 0, 1):
                        amm.bands_x[n] = 0 if up else 1
                        amm.bands_y[n] = 1 if up else 0
                    amm.active_band = -1 if up else 1
                    expected = {}
                    for n in (-1, 0, 1):
                        f, g, inv = invariant(amm, n)
                        fee = Decimal(str(amm.dynamic_fee(n, 0)))
                        net = inv / (g if up else f) - (f if up else g)
                        expected[n] = float(net / (1 - fee))
                    dx, dy = amm.trade_to_price(4 if up else 0.2)
                    for n, gross in expected.items():
                        self.assertAlmostEqual((amm.bands_x if up else amm.bands_y)[n], gross, delta=1e-11)
                        self.assertEqual((amm.bands_y if up else amm.bands_x)[n], 0)
                    self.assertAlmostEqual(dx if up else dy, sum(expected.values()), delta=1e-11)
                    self.assertAlmostEqual(dy if up else dx, -3, delta=1e-11)

    def test_market_inside_fee_spread_cannot_reverse_trade(self):
        for fee, factor in ((0.1, 0.95), (0.1, 1.05), (0.003, 0.999), (0.003, 1.001)):
            amm = position(fee)
            before = dict(amm.bands_x), dict(amm.bands_y)
            self.assertEqual(amm.trade_to_price(amm.get_p() * factor), (0, 0))
            self.assertEqual((dict(amm.bands_x), dict(amm.bands_y)), before)

    def test_capped_fee_is_finite_and_blocks_ordinary_trades(self):
        self.assertTrue(isfinite(fee_multiplier(1)))
        self.assertAlmostEqual(fee_multiplier(1) / 1e18, 1)
        for market in (0.5, 2):
            amm = position(0.003, memory=1)
            self.assertEqual(amm.trade_to_price(market), (0, 0))

    def test_simulator_passes_only_external_execution_cost(self):
        bars = [[180, 1, 2, 0.5, 1, 0], [360, 1, 2, 0.5, 1, 0]]
        loader, oracle = Mock(), Mock()
        loader.load_prices.return_value = bars
        oracle.calculate_oracle_prices.return_value = [1, 1]
        execute = LendingAMM.trade_to_price
        calls = []

        def checked(amm, price):
            calls.append(price)
            self.assertIn(price, (2 * 0.9995, 0.5 * 1.0005))
            return execute(amm, price)

        with patch.object(LendingAMM, "trade_to_price", checked):
            Simulator(ConstantInitialLiquidity, loader, oracle, 0.0005).single_run(50, 0.01, 0, 1, 4, 0)
        self.assertEqual(set(calls), {2 * 0.9995, 0.5 * 1.0005})


if __name__ == "__main__":
    unittest.main()
