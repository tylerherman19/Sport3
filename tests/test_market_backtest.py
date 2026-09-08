import unittest

from scripts.backtest_nfl import market_edge_backtest


class MarketEdgeBacktestTests(unittest.TestCase):
    def test_uses_selected_side_price_and_fixed_cutoffs(self):
        records = [
            {"season": 2024, "prob": 0.60, "actual": 1.0,
             "market": {"home_prob": 0.50, "home_american": 100, "away_american": -120}},
            {"season": 2024, "prob": 0.40, "actual": 1.0,
             "market": {"home_prob": 0.50, "home_american": -120, "away_american": 100}},
        ]
        result = market_edge_backtest(records)
        five = next(row for row in result["cutoffs"] if row["min_edge"] == 0.05)
        self.assertEqual(five["bets"], 2)
        self.assertEqual(five["net_units"], 0.0)
        self.assertEqual(five["roi"], 0.0)


if __name__ == "__main__":
    unittest.main()
