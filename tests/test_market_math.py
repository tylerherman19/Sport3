import unittest

from model.ensemble_model import kelly_criterion


class KellyCriterionTests(unittest.TestCase):
    def test_uses_the_offered_price_not_a_fair_probability(self):
        # At +100, a 60% forecast has a full-Kelly stake of 20%; quarter Kelly
        # is therefore 5%.  This is independently checkable from the odds.
        self.assertEqual(kelly_criterion(0.60, 100), 0.05)

    def test_no_positive_stake_without_positive_expected_value(self):
        self.assertEqual(kelly_criterion(0.50, -120), 0.0)

    def test_invalid_or_missing_price_is_safe(self):
        self.assertEqual(kelly_criterion(0.60, None), 0.0)
        self.assertEqual(kelly_criterion(0.60, 0), 0.0)


if __name__ == "__main__":
    unittest.main()
