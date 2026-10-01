import unittest

from simulator.amm.lending_amm import BandBalances


class TestBandBalances(unittest.TestCase):
    def test_missing_reads_and_clear_include_overflow_bands(self):
        balances = BandBalances()
        bands = (-10000, -501, -500, 0, 500, 501, 10000)
        for n in bands:
            self.assertEqual(balances[n], 0.0)
            balances[n] = 2.5
        self.assertEqual(dict(balances), dict.fromkeys(bands, 2.5))
        balances.clear()
        self.assertEqual(dict(balances), {})
        for n in bands:
            self.assertEqual(balances[n], 0.0)

    def test_fractional_band_keys_are_rejected(self):
        balances = BandBalances()
        for n in (1.5, -0.5):
            with self.assertRaises(TypeError):
                balances[n] = 1.0
            with self.assertRaises(TypeError):
                balances[n]
            with self.assertRaises(TypeError):
                balances.update({n: 1.0})
        self.assertEqual(dict(balances), {})


if __name__ == "__main__":
    unittest.main()
