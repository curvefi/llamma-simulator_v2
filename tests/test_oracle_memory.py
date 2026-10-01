import unittest
from unittest.mock import Mock, patch

from simulator.amm.intitial_liquidity import ConstantInitialLiquidity
from simulator.amm.lending_amm import LendingAMM, OracleState
from simulator.amm.simulator import Simulator

WAD = 10**18


def contract_limit(state, raw, timestamp):
    # Integer arithmetic from AMM.vy limit_p_o; no simulator helpers.
    old, memory, last = state
    dt = 120 - min(120, timestamp - last)
    if not dt:
        return raw, 0
    ratio = min(old, raw) * WAD // max(old, raw)
    if ratio < 8 * WAD // 10:
        raw = old * 5 // 4 if raw > old else old * 4 // 5
        ratio = 8 * WAD // 10
    fee = (WAD + memory - ratio * ratio // WAD * ratio // WAD) * dt // 120
    return raw, min(fee, WAD - 1)


def position():
    amm = LendingAMM(100 / 99 + 0.0001, 100, 0.003, oracle_state=OracleState.initial(1, 0))
    ConstantInitialLiquidity(1, 4).deposit(amm, 1)
    return amm


class OracleMemoryTest(unittest.TestCase):
    def test_observations_and_quotes_preserve_memory_and_quote_clock(self):
        amm = position()
        state = amm.oracle_state()
        for t, raw in ((10, 0.99), (20, 0.8), (30, 0.6)):
            amm.set_p_oracle(raw, t)
            self.assertEqual(amm.oracle_state(), state)
        before = vars(amm).copy()
        for t in (30, 60, 180):
            amm.dynamic_fee(1, t)
            amm._price_oracle_view(t)
        self.assertEqual(vars(amm), before)
        self.assertEqual(amm._price_oracle_view(None), amm._price_oracle_view(30))

    def test_extra_observations_do_not_change_exchange(self):
        sparse, frequent = position(), position()
        frequent.set_p_oracle(1.2, 10)
        frequent.set_p_oracle(0.95, 20)
        for amm in (sparse, frequent):
            amm.set_p_oracle(0.9, 30)
        self.assertEqual(sparse.trade_to_price(2), frequent.trade_to_price(2))
        self.assertEqual(sparse.oracle_state(), frequent.oracle_state())
        self.assertEqual(sparse.bands_x, frequent.bands_x)
        self.assertEqual(sparse.bands_y, frequent.bands_y)

    def test_integer_memory_parity_including_zero_exchange_and_expiry(self):
        amm = position()
        expected = (WAD, 0, 0)
        for timestamp, raw in ((30, 6 * WAD // 10), (30, 6 * WAD // 10), (60, WAD), (180, WAD)):
            amm.set_p_oracle(raw / WAD, timestamp)
            snapshot = contract_limit(expected, raw, timestamp)
            balances = dict(amm.bands_x), dict(amm.bands_y), amm.active_band
            self.assertEqual(amm.exchange_zero(), (0, 0))
            expected = (*snapshot, timestamp)
            self.assertAlmostEqual(amm.old_p_oracle, snapshot[0] / WAD, delta=1e-14)
            self.assertAlmostEqual(amm.old_dfee, snapshot[1] / WAD, delta=1e-14)
            self.assertEqual(amm.prev_p_oracle_time, timestamp)
            self.assertEqual((dict(amm.bands_x), dict(amm.bands_y), amm.active_band), balances)
            next_view = contract_limit(expected, raw, timestamp)
            self.assertAlmostEqual(amm.p_oracle, next_view[0] / WAD, delta=1e-14)
        self.assertEqual(amm.old_dfee, 0)

    def test_exchange_uses_one_snapshot_for_every_band(self):
        amm = position()
        amm.set_p_oracle(0.6, 30)
        snapshot = contract_limit((WAD, 0, 0), 6 * WAD // 10, 30)
        seen = []
        distance = amm._distance_fee

        def checked(price, n):
            seen.append(price)
            self.assertEqual(amm.oracle_state(), OracleState.initial(1, 0))
            return distance(price, n)

        with patch.object(amm, "_distance_fee", checked):
            dx, dy = amm.trade_to_price(2)
        self.assertGreater(dx, 0)
        self.assertLess(dy, 0)
        self.assertEqual(seen, [snapshot[0] / WAD] * 4)
        self.assertEqual(amm.old_p_oracle, 0.8)
        self.assertAlmostEqual(amm.old_dfee, 0.366)
        self.assertEqual(amm.prev_p_oracle_time, 30)
        self.assertAlmostEqual(amm.p_oracle, 0.64)
        self.assertAlmostEqual(amm._price_oracle_view(30)[1], 0.854)

    def test_low_opportunity_is_quoted_after_high_exchange(self):
        bars = [[30, 1, 2, 0.01, 1, 0]]
        loader, oracle = Mock(), Mock()
        loader.load_prices.return_value = bars
        oracle.calculate_oracle_prices.return_value = [0.6]
        sim = Simulator(ConstantInitialLiquidity, loader, oracle)
        exchange = LendingAMM.trade_to_price
        events = []

        def checked_trade(amm, price):
            result = exchange(amm, price)
            events.append(("exchange", amm.oracle_state()))
            return result

        quote = LendingAMM.dynamic_fee

        def checked_quote(amm, n, timestamp=None):
            events.append(("quote", amm.oracle_state()))
            return quote(amm, n, timestamp)

        with (
            patch.object(LendingAMM, "trade_to_price", checked_trade),
            patch.object(LendingAMM, "dynamic_fee", checked_quote),
        ):
            sim.calculate_loss(100, 0.003, bars, [0.6], 4, initial_state=OracleState.initial(1, 0))
        exchanges = [i for i, (event, _) in enumerate(events) if event == "exchange"]
        self.assertEqual(len(exchanges), 2)
        after_high = events[exchanges[0]][1]
        between = events[exchanges[0] + 1 : exchanges[1]]
        self.assertTrue(between)
        self.assertTrue(all(event == "quote" and state == after_high for event, state in between))

    def test_unfilled_trade_preserves_memory_and_active_band(self):
        amm = position()
        amm.fee = 0.5
        before = amm.oracle_state(), amm.active_band
        self.assertEqual(amm.trade_to_price(1.02), (0, 0))
        self.assertEqual((amm.oracle_state(), amm.active_band), before)


if __name__ == "__main__":
    unittest.main()
