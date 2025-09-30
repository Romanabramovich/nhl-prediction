# features/build_snapshot.py
from datetime import datetime
from database import get_database_engine
import json
from sqlalchemy import text 

def compute_rest_days(conn, team_id: int, cutoff: datetime) -> dict:
    # last game before cutoff
    last_game = conn.execute(text(
        """
        SELECT date FROM team_stats
        WHERE team_id = :t AND date < :cutoff
        ORDER BY date DESC LIMIT 1
        """),{"t": team_id, "cutoff": cutoff}).fetchone()
    
    rest_days = None
    
    if last_game:
        last_date = last_game[0]
        # Convert date to timezone-aware datetime for comparison
        last_datetime = datetime.combine(last_date, datetime.min.time()).replace(tzinfo=cutoff.tzinfo)
        rest_days = (cutoff - last_datetime).days
        
    # back-to-back flag
    # can derive from rest_days and recent density
    return {"rest_days": rest_days, "b2b": 1 if rest_days == 1 else 0}

def compute_rolling_form(conn, team_id: int, cutoff: datetime, window: int = 10) -> dict:
    rows = conn.execute(text(
        """
        SELECT points, wins, losses, goals_for, goals_against
        FROM team_stats
        WHERE team_id = :t AND date < :cutoff
        ORDER BY date DESC LIMIT :window
        """
    ), {"t": team_id, "cutoff": cutoff, "window":window}).fetchall()
    
    if not rows or len(rows) < 2:
        return {
            "roll_pts": None,
            "roll_wins":None,
            "roll_loss":None,
            "roll_goals_for":None,
            "roll_goals_against":None
            }
    
    # For cumulative stats, calculate the difference between most recent and N games ago
    if len(rows) >= window:
        # Most recent game (index 0) vs game N games ago (index window-1)
        recent = rows[0]
        old = rows[window-1]
        
        roll_pts = recent[0] - old[0]
        roll_wins = recent[1] - old[1]
        roll_loss = recent[2] - old[2]
        roll_gf = recent[3] - old[3]
        roll_ga = recent[4] - old[4]
    else:
        # If we don't have enough games, use all available data
        recent = rows[0]
        old = rows[-1]
        
        roll_pts = recent[0] - old[0]
        roll_wins = recent[1] - old[1]
        roll_loss = recent[2] - old[2]
        roll_gf = recent[3] - old[3]
        roll_ga = recent[4] - old[4]
        
    return {
            "roll_pts": roll_pts,
            "roll_wins": roll_wins,
            "roll_loss": roll_loss,
            "roll_goals_for": roll_gf,
            "roll_goals_against": roll_ga
            }
    
def build_feature_snapshot(season: str, game_id: int, cutoff: datetime, engine = None):
    
    if engine is None:
        engine = get_database_engine()
    
    with engine.begin() as conn:        
        game = conn.execute(text(
            """
            SELECT home_team_id, away_team_id, game_date
            FROM games
            WHERE game_id = :gid
            """
        ), {"gid": game_id}).fetchone()
        
        if not game:
            raise ValueError("value error: unknown game_id")
    
        home, away, game_timestamp = game
        
        # cutoff must be before game start
        assert cutoff < game_timestamp, "cutoff must be before game start"
        
        # team seasonal aggregates as-of day prior
        as_of = cutoff.date()
        aggregate = conn.execute(text(
            """
            SELECT team_id, points, wins, losses, goals_for, goals_against
            FROM team_stats
            WHERE date = :as_of AND team_id in (:h, :a)
            """
        ), {"as_of": as_of, "h": home, "a": away}).fetchall()
        
        aggregate_map = {row[0]: row for row in aggregate}
        
        # derive team metrics
        home_rest = compute_rest_days(conn, home, cutoff)
        home_form = compute_rolling_form(conn, home, cutoff, 10)
        
        away_rest = compute_rest_days(conn, away, cutoff)
        away_form = compute_rolling_form(conn, away, cutoff, 10)
        
        # match-up difference (home minus away)
        feature = {
            "cutoff_utc": cutoff.isoformat(),
            "home_team_id": home,
            "away_team_id": away,
            "home_rest_days": home_rest["rest_days"],
            "away_rest_days": away_rest["rest_days"],
            "rest_diff": (home_rest["rest_days"] or 0) - (away_rest["rest_days"] or 0),
            "b2b_home": home_rest["b2b"],
            "b2b_away": away_rest["b2b"],
            "home_roll_pts": home_form["roll_pts"], "away_roll_pts": away_form["roll_pts"],
            "home_roll_w": home_form["roll_wins"], "away_roll_w": away_form["roll_wins"],
            "home_roll_l": home_form["roll_loss"], "away_roll_l": away_form["roll_loss"],
            "home_roll_gf": home_form["roll_goals_for"], "away_roll_gf": away_form["roll_goals_for"],
            "home_roll_ga": home_form["roll_goals_against"], "away_roll_ga": away_form["roll_goals_against"],
        }
        
        conn.execute(text(
            """
          INSERT INTO feature_snapshots (game_id, as_of_ts_utc, feature_json)
          VALUES (:gid, :ts, :json)
          ON CONFLICT (game_id, as_of_ts_utc) DO NOTHING
        """), {"gid": game_id, "ts": cutoff, "json": json.dumps(feature)})
        
        