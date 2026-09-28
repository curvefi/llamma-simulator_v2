import unittest
from decimal import Decimal, localcontext
from unittest.mock import Mock, patch

from simulator.amm.intitial_liquidity import ConstantInitialLiquidity
from simulator.amm.lending_amm import LendingAMM, OracleState, initial_recovery_coefficient, oracle_states
from simulator.amm.simulator import Simulator, get_loss_rate


def simulator(candles, oracles):
    loader, oracle = Mock(), Mock()
    loader.load_prices.return_value = candles
    oracle.calculate_oracle_prices.return_value = oracles
    return Simulator(ConstantInitialLiquidity, loader, oracle)


def expected_opening_recovery(a, bands):
    # Independently value the first band's invariant at oracle=1, then convert
    # its remaining collateral to the lower boundary. Other bands are all y.
    with localcontext() as context:
        context.prec = 60
        a, amount = Decimal(a), Decimal(1) / bands
        q = (a - 1) / a
        upper = 1 + Decimal("0.0001") * q
        lower = upper * q
        y0 = amount / upper
        f, g = a * y0 / upper, (a - 1) * y0 * upper
        invariant = f * (g + amount)
        y_at_oracle = a * y0 * (1 - lower)
        x_at_oracle = invariant / (g + y_at_oracle) - f
        value = x_at_oracle + y_at_oracle * lower.sqrt()
        for band in range(2, bands + 1):
            top = upper * q ** (band - 1)
            value += amount * (top * top * q).sqrt()
        return float(value)


class OracleStateTest(unittest.TestCase):
    def test_actual_history_retains_memory_independent_of_candidate(self):
        observations = [(0, 100), (60, 101), (120, 99), (300, 98), (360, 140)]
        expected = list(oracle_states(observations))
        self.assertAlmostEqual(expected[1].old_dfee, 0.5 * (1 - (100 / 101) ** 3))
        self.assertEqual(expected[3].old_dfee, 0)
        for a, fee, distance in [(2, 0, 0), (10, 0.025, 0.25), (393, 0.0028, 1)]:
            amm = LendingAMM(1000, a, fee, distance, oracle_state=expected[0])
            amm.deposit_nrange(7, 90, 4)
            for i, (t, price) in enumerate(observations[1:], 1):
                amm.set_p_oracle(price, t)
                self.assertEqual(amm.oracle_state(), expected[i])
                restored = LendingAMM(2, a, fee, distance, oracle_state=expected[i])
                self.assertEqual(restored.oracle_state(), expected[i])

    def test_restart_does_not_apply_first_clamped_update_twice(self):
        for jump, effective in [(4, 1.25), (0.25, 0.8)]:
            candles = [[t, p, p, p, p, 0] for t, p in [(0, 1), (60, jump), (120, jump)]]
            sim = simulator(candles, [1, jump, jump])
            self.assertEqual(sim.oracle_states[1].p_oracle, effective)
            calls = []
            update = LendingAMM.set_p_oracle

            def checked(amm, price, timestamp):
                calls.append(timestamp)
                return update(amm, price, timestamp)

            with patch.object(LendingAMM, "set_p_oracle", checked):
                sim.single_run(100, 0.001, 1 / 3, 2 / 3, 4)
            self.assertEqual(calls, [120])

    def test_flat_history_no_artificial_fee_and_correct_bands(self):
        state = list(oracle_states((t, 1.0) for t in range(0, 300, 60)))[-1]
        self.assertEqual(state.old_dfee, 0)
        for a in (2, 10, 100, 393, 600, 1200, 10000):
            for bands in (1, 4, 50):
                base = a / (a - 1) + 0.0001
                amm = LendingAMM(base, a, 0.001, oracle_state=state)
                ConstantInitialLiquidity(1, bands).deposit(amm, 1)
                self.assertEqual((amm.min_band, amm.max_band), (1, bands))
                self.assertAlmostEqual(sum(amm.bands_y.values()), 1)
                expected = expected_opening_recovery(a, bands)
                self.assertAlmostEqual(amm.get_all_x(), expected, delta=1e-10)
                self.assertAlmostEqual(initial_recovery_coefficient(a, bands), expected, delta=1e-10)

    def test_placement_matches_phil_without_resetting_memory(self):
        state = list(oracle_states([(0, 1), (60, 1.1), (120, 0.8)]))[-1]
        self.assertGreater(state.old_dfee, 0)
        for a in (10, 393, 600, 1200, 10000):
            for bands in (1, 4, 50):
                for shift in (0, 0.05):
                    p0 = state.p_oracle * (1 - shift)
                    base = p0 * (a / (a - 1) + 0.0001)
                    # Phil deposits while the constructor oracle is p_base.
                    phil = LendingAMM(base, a, 0.001, oracle_state=OracleState.initial(base, 120))
                    phil.deposit_nrange(1, p0, bands)
                    current = LendingAMM(base, a, 0.001, oracle_state=state)
                    ConstantInitialLiquidity(p0, bands).deposit(current, 1)
                    self.assertEqual((current.min_band, current.max_band), (phil.min_band, phil.max_band))
                    self.assertEqual(dict(current.bands_y), dict(phil.bands_y))
                    for n in range(phil.min_band, phil.max_band + 1):
                        self.assertEqual(current.p_top(n), phil.p_top(n))
                        self.assertEqual(current.p_bottom(n), phil.p_bottom(n))
                    self.assertEqual(current.oracle_state(), state)

    def test_score_preserves_terminal_recovery_objective(self):
        for a in (10, 100, 393, 600):
            q = (a - 1) / a
            upper = (1 / q + 0.0001) * q
            initial = expected_opening_recovery(a, 1)
            final = 0.7**3 / (upper**2 * q)
            # Start at the empty band's trading boundary so the first candle
            # leaves the collateral intact for the independent final formula.
            spot = 1 / upper**2
            candles = [[0, spot, spot, spot, spot, 0], [180, 0.7, 0.7, 0.7, 0.7, 0]]
            sim = simulator(candles, [1, 0.7])
            loss = sim.single_run(a, 0, 0, 1, 1, 0)
            self.assertAlmostEqual(loss, 1 - final / initial, places=12)
            self.assertAlmostEqual(1 - (1 - loss) * initial_recovery_coefficient(a, 1), 1 - final, places=12)

    def test_valuation_precedes_and_includes_first_trade(self):
        candles = [[0, 2, 2, 2, 2, 0]]
        sim = simulator(candles, [1])
        events = []
        value, trade = LendingAMM.get_all_x, LendingAMM.trade_to_price

        def checked_value(amm):
            result = value(amm)
            events.append(("value", result))
            return result

        def checked_trade(amm, price):
            result = trade(amm, price)
            events.append(("trade", result))
            return result

        with (
            patch.object(LendingAMM, "get_all_x", checked_value),
            patch.object(LendingAMM, "trade_to_price", checked_trade),
        ):
            loss = sim.single_run(10, 0, 0, 1, 4, 0)
        self.assertEqual(events[0][0], "value")
        self.assertTrue(any(e[0] == "trade" and e[1] != (0, 0) for e in events))
        self.assertEqual(events[-1][0], "value")
        self.assertEqual(loss, 1 - events[-1][1] / events[0][1])

    def test_failed_replay_is_not_replaced_by_zero(self):
        candles = [[0, 1, 1, 1, 1, 0], [60, 1, 1, 1, 1, 0]]
        sim = simulator(candles, [1, 1])
        with (
            patch("simulator.amm.simulator.Simulator", return_value=sim),
            patch.object(sim, "single_run", side_effect=ValueError("invalid baseline")),
        ):
            with self.assertRaisesRegex(ValueError, "invalid baseline"):
                get_loss_rate(
                    ConstantInitialLiquidity,
                    sim.price_history_loader,
                    sim.price_oracle,
                    0,
                    100,
                    0.001,
                    4,
                    samples=1,
                    use_threading=False,
                )


if __name__ == "__main__":
    unittest.main()
