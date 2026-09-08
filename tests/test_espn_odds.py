import unittest

from scripts.data_fetcher import fetch_nfl_espn_betting_odds


class EspnOddsTests(unittest.TestCase):
    def test_parses_and_devigs_public_moneyline(self):
        game = {
            "home_name": "Seattle Seahawks", "away_name": "New England Patriots",
            "espn_odds": [{
                "provider": {"displayName": "DraftKings"},
                "moneyline": {
                    "home": {"close": {"odds": "-150"}},
                    "away": {"close": {"odds": "+130"}},
                },
            }],
        }
        result = fetch_nfl_espn_betting_odds([game])
        odds = result["New England Patriots_at_Seattle Seahawks"]
        self.assertAlmostEqual(odds["home_prob"] + odds["away_prob"], 1.0, places=4)
        self.assertEqual(odds["source"], "ESPN / DraftKings")

    def test_skips_spread_only_games(self):
        self.assertEqual(fetch_nfl_espn_betting_odds([{"espn_odds": [{"spread": -3.5}]}]), {})


if __name__ == "__main__":
    unittest.main()
