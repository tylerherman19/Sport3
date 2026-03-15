"""
Injury Model — System 13 (Injury Prediction Drivers)
Computes per-team injury impact scores based on player position importance
and availability status. Returns ELO adjustments used in ensemble predictions.
Supports both NFL and NBA leagues.
"""

import logging

log = logging.getLogger(__name__)

# Position impact weights (ELO points lost per starter-equivalent absence)
POSITION_WEIGHTS = {
    "QB":  30.0,   # Quarterback — highest impact
    "LT":  10.0,   # Left tackle
    "OT":   9.0,   # Offensive tackle
    "WR":   8.0,   # Wide receiver
    "TE":   7.5,   # Tight end
    "RB":   7.0,   # Running back
    "OG":   6.0,   # Offensive guard
    "OC":   5.5,   # Center
    "DE":   7.0,   # Defensive end / edge rusher
    "DT":   6.0,   # Defensive tackle
    "OLB":  5.5,   # Outside linebacker
    "ILB":  5.0,   # Inside linebacker
    "MLB":  5.0,   # Middle linebacker
    "LB":   5.0,   # Linebacker (generic)
    "CB":   6.0,   # Cornerback
    "S":    5.0,   # Safety
    "SS":   5.0,
    "FS":   5.0,
    "K":    2.0,   # Kicker
    "P":    1.0,   # Punter
    "LS":   0.5,   # Long snapper
}

# Status multipliers — how much of the position weight to apply
STATUS_MULTIPLIERS = {
    "out":          1.00,
    "injured reserve": 1.00,
    "ir":           1.00,
    "doubtful":     0.75,
    "questionable": 0.40,
    "limited":      0.25,
    "probable":     0.10,
    "day-to-day":   0.30,
}

# Cap maximum ELO penalty per team (prevents extreme swings)
MAX_INJURY_ELO_PENALTY = 60.0
# Cap per-player contribution (no single non-QB player dominates more than QB)
MAX_PLAYER_CONTRIBUTION = 30.0


def _status_multiplier(status_str: str) -> float:
    """Return a [0, 1] severity multiplier for a given injury status string."""
    s = status_str.lower().strip()
    for key, mult in STATUS_MULTIPLIERS.items():
        if key in s:
            return mult
    return 0.0  # Unknown status → no impact assumed


def _position_weight(position_str: str) -> float:
    """Return the base ELO impact for a given position abbreviation."""
    pos = position_str.upper().strip()
    return POSITION_WEIGHTS.get(pos, 3.0)  # default 3 for unlisted positions


def compute_injury_impact(team_injuries: list) -> dict:
    """
    Compute an injury impact score for a single team.

    Parameters
    ----------
    team_injuries : list of dicts
        Each dict has keys: 'player', 'status', 'position'

    Returns
    -------
    dict with:
        'elo_penalty'     : float — ELO points subtracted from team strength
        'impact_score'    : float — normalised 0–1 severity index
        'key_players_out' : list  — players with Out/Doubtful status
        'total_players'   : int   — number of injured players considered
    """
    total_elo_penalty = 0.0
    key_players_out = []

    for player in team_injuries:
        pos = player.get("position", "")
        status = player.get("status", "")
        name = player.get("player", "Unknown")

        pos_weight = _position_weight(pos)
        status_mult = _status_multiplier(status)

        contribution = min(pos_weight * status_mult, MAX_PLAYER_CONTRIBUTION)
        total_elo_penalty += contribution

        if status_mult >= 0.75:  # Out or Doubtful
            key_players_out.append({
                "player": name,
                "position": pos,
                "status": status,
                "elo_impact": round(contribution, 1),
            })

    capped_penalty = min(total_elo_penalty, MAX_INJURY_ELO_PENALTY)
    # Normalised impact score: 0 = no injuries, 1 = max possible penalty
    impact_score = round(capped_penalty / MAX_INJURY_ELO_PENALTY, 4)

    return {
        "elo_penalty": round(capped_penalty, 2),
        "impact_score": impact_score,
        "key_players_out": key_players_out,
        "total_players": len(team_injuries),
    }


def compute_all_team_impacts(injuries: dict) -> dict:
    """
    Compute injury impacts for all teams.

    Parameters
    ----------
    injuries : dict mapping team abbreviation -> list of injury dicts

    Returns
    -------
    dict mapping team abbreviation -> injury impact dict
    """
    impacts = {}
    for team, team_injuries in injuries.items():
        impacts[team] = compute_injury_impact(team_injuries)
        if impacts[team]["elo_penalty"] > 0:
            log.debug(
                f"{team} injury penalty: {impacts[team]['elo_penalty']:.1f} ELO pts "
                f"({impacts[team]['total_players']} players)"
            )
    return impacts


# ─── NBA Injury System ────────────────────────────────────────────────────────

# NBA position tier weights (ELO points lost per absence)
NBA_POSITION_WEIGHTS = {
    # Guard positions
    "PG": 18.0,   # Point guard
    "SG": 15.0,   # Shooting guard
    "G":  15.0,   # Guard (generic)
    # Forward positions
    "SF": 15.0,   # Small forward
    "PF": 14.0,   # Power forward
    "F":  14.0,   # Forward (generic)
    # Center
    "C":  16.0,   # Center
    # Generic
    "G/F": 14.0,
    "F/C": 15.0,
    "F/G": 14.0,
}

# NBA player tier weights (used when position alone isn't enough)
NBA_TIER_WEIGHTS = {
    "superstar":  50.0,
    "all-star":   30.0,
    "starter":    15.0,
    "rotation":    8.0,
}

# NBA status multipliers (same scale as NFL)
NBA_STATUS_MULTIPLIERS = {
    "out":             1.00,
    "injured reserve": 1.00,
    "ir":              1.00,
    "day-to-day":      0.50,
    "doubtful":        0.75,
    "questionable":    0.40,
    "probable":        0.10,
    "game time decision": 0.35,
}

# Maximum NBA injury penalty per team
NBA_MAX_INJURY_ELO_PENALTY = 80.0
NBA_MAX_PLAYER_CONTRIBUTION = 50.0

# Back-to-back rest risk penalty
NBA_BACK_TO_BACK_PENALTY = 5.0


def _nba_status_multiplier(status_str: str) -> float:
    s = status_str.lower().strip()
    for key, mult in NBA_STATUS_MULTIPLIERS.items():
        if key in s:
            return mult
    return 0.0


def _nba_position_weight(position_str: str) -> float:
    pos = position_str.upper().strip()
    return NBA_POSITION_WEIGHTS.get(pos, 8.0)  # default 8 for unlisted


def compute_nba_injury_impact(team_injuries: list) -> dict:
    """
    Compute NBA injury impact for a single team.

    Parameters
    ----------
    team_injuries : list of dicts
        Each dict has keys: 'player', 'status', 'position', optionally 'tier'

    Returns
    -------
    dict with 'elo_penalty', 'impact_score', 'key_players_out', 'total_players'
    """
    total_elo_penalty = 0.0
    key_players_out = []

    for player in team_injuries:
        pos = player.get("position", "")
        status = player.get("status", "")
        name = player.get("player", "Unknown")
        tier = player.get("tier", "").lower()

        # Use tier weight if provided, otherwise position weight
        if tier in NBA_TIER_WEIGHTS:
            base_weight = NBA_TIER_WEIGHTS[tier]
        else:
            base_weight = _nba_position_weight(pos)

        status_mult = _nba_status_multiplier(status)
        contribution = min(base_weight * status_mult, NBA_MAX_PLAYER_CONTRIBUTION)
        total_elo_penalty += contribution

        if status_mult >= 0.75:
            key_players_out.append({
                "player": name,
                "position": pos,
                "status": status,
                "elo_impact": round(contribution, 1),
            })

    capped_penalty = min(total_elo_penalty, NBA_MAX_INJURY_ELO_PENALTY)
    impact_score = round(capped_penalty / NBA_MAX_INJURY_ELO_PENALTY, 4)

    return {
        "elo_penalty": round(capped_penalty, 2),
        "impact_score": impact_score,
        "key_players_out": key_players_out,
        "total_players": len(team_injuries),
    }


def compute_all_nba_team_impacts(injuries: dict) -> dict:
    """Compute NBA injury impacts for all teams."""
    impacts = {}
    for team, team_injuries in injuries.items():
        impacts[team] = compute_nba_injury_impact(team_injuries)
        if impacts[team]["elo_penalty"] > 0:
            log.debug(
                f"NBA {team} injury penalty: {impacts[team]['elo_penalty']:.1f} ELO pts "
                f"({impacts[team]['total_players']} players)"
            )
    return impacts


def injury_elo_adjustment(home_team: str, away_team: str,
                           injury_impacts: dict) -> tuple:
    """
    Return (home_adj, away_adj) ELO adjustments from injury data.
    Both values are non-positive (penalties only).

    Parameters
    ----------
    home_team, away_team : str — team abbreviations
    injury_impacts       : dict from compute_all_team_impacts()

    Returns
    -------
    (home_adj, away_adj) as floats (≤ 0)
    """
    home_penalty = injury_impacts.get(home_team, {}).get("elo_penalty", 0.0)
    away_penalty = injury_impacts.get(away_team, {}).get("elo_penalty", 0.0)
    return -home_penalty, -away_penalty
