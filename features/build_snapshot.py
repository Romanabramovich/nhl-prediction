# features/build_snapshot.py
from datetime import datetime, timedelta
from database import get_database_engine
import json
from sqlalchemy import text
from decimal import Decimal 

def convert_decimals_to_floats(obj):
    """Convert Decimal objects to floats for JSON serialization"""
    if isinstance(obj, dict):
        return {key: convert_decimals_to_floats(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_decimals_to_floats(item) for item in obj]
    elif isinstance(obj, Decimal):
        return float(obj)
    else:
        return obj

def get_current_contextual_stats(conn, team_id: int, cutoff: datetime) -> dict:
    # get current contextual stats that represent situational advantages and current state
    current_stats = conn.execute(text(
        """
        SELECT league_sequence, league_home_sequence, league_road_sequence, league_l10_sequence,
               conference_sequence, conference_home_sequence, conference_road_sequence, conference_l10_sequence,
               division_sequence, division_home_sequence, division_road_sequence, division_l10_sequence,
               wildcard_sequence, waivers_sequence,
               streak_code, streak_count,
               home_wins, home_losses, home_ot_losses, home_points, home_goal_for, home_goal_against, home_goal_diff,
               home_regulation_wins, home_regulation_plus_ot_wins,
               road_wins, road_losses, road_ot_losses, road_points, road_goals_for, road_goals_against, road_goal_diff,
               road_regulation_wins, road_regulation_plus_ot_wins,
               games_played, points, win_pct, goal_diff_pct
        FROM team_stats
        WHERE team_id = :t AND date < :cutoff
        ORDER BY date DESC LIMIT 1
        """
    ), {"t": team_id, "cutoff": cutoff}).fetchone()
    
    if not current_stats:
        return {
            "league_sequence": None, "league_home_sequence": None, "league_road_sequence": None, "league_l10_sequence": None,
            "conference_sequence": None, "conference_home_sequence": None, "conference_road_sequence": None, "conference_l10_sequence": None,
            "division_sequence": None, "division_home_sequence": None, "division_road_sequence": None, "division_l10_sequence": None,
            "wildcard_sequence": None, "waivers_sequence": None,
            "streak_code": None, "streak_count": None,
            "home_wins": None, "home_losses": None, "home_ot_losses": None, "home_points": None, 
            "home_goal_for": None, "home_goal_against": None, "home_goal_diff": None,
            "home_regulation_wins": None, "home_regulation_plus_ot_wins": None,
            "road_wins": None, "road_losses": None, "road_ot_losses": None, "road_points": None,
            "road_goals_for": None, "road_goals_against": None, "road_goal_diff": None,
            "road_regulation_wins": None, "road_regulation_plus_ot_wins": None,
            "games_played": None, "points": None, "win_pct": None, "goal_diff_pct": None
        }
    
    return {
        "league_sequence": current_stats[0], "league_home_sequence": current_stats[1], "league_road_sequence": current_stats[2], "league_l10_sequence": current_stats[3],
        "conference_sequence": current_stats[4], "conference_home_sequence": current_stats[5], "conference_road_sequence": current_stats[6], "conference_l10_sequence": current_stats[7],
        "division_sequence": current_stats[8], "division_home_sequence": current_stats[9], "division_road_sequence": current_stats[10], "division_l10_sequence": current_stats[11],
        "wildcard_sequence": current_stats[12], "waivers_sequence": current_stats[13],
        "streak_code": current_stats[14], "streak_count": current_stats[15],
        "home_wins": current_stats[16], "home_losses": current_stats[17], "home_ot_losses": current_stats[18], "home_points": current_stats[19],
        "home_goal_for": current_stats[20], "home_goal_against": current_stats[21], "home_goal_diff": current_stats[22],
        "home_regulation_wins": current_stats[23], "home_regulation_plus_ot_wins": current_stats[24],
        "road_wins": current_stats[25], "road_losses": current_stats[26], "road_ot_losses": current_stats[27], "road_points": current_stats[28],
        "road_goals_for": current_stats[29], "road_goals_against": current_stats[30], "road_goal_diff": current_stats[31],
        "road_regulation_wins": current_stats[32], "road_regulation_plus_ot_wins": current_stats[33],
        "games_played": current_stats[34], "points": current_stats[35], "win_pct": current_stats[36], "goal_diff_pct": current_stats[37]
    }

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
        # convert date to timezone-aware datetime for comparison
        last_datetime = datetime.combine(last_date, datetime.min.time()).replace(tzinfo=cutoff.tzinfo)
        rest_days = (cutoff - last_datetime).days
        
    # back-to-back flag
    # can derive from rest_days and recent density
    return {"rest_days": rest_days, "b2b": 1 if rest_days == 1 else 0}

def compute_rolling_form(conn, team_id: int, cutoff: datetime, window: int = 10) -> dict:
    # first, try to get L10 stats directly from team_stats table
    l10_stats = conn.execute(text(
        """
        SELECT l10_games_played, l10_wins, l10_losses, l10_ot_losses, l10_points, l10_goals_for, l10_goals_against, l10_goal_diff, l10_regulation_wins, l10_regulation_plus_ot_wins, league_l10_sequence, conference_l10_sequence, division_l10_sequence    
        FROM team_stats
        WHERE team_id = :t AND date < :cutoff
        ORDER BY date DESC LIMIT 1
        """
    ), {"t": team_id, "cutoff": cutoff}).fetchone()
    
    # if L10 stats are available and we have at least some games played, use them
    if l10_stats and l10_stats[0] is not None and l10_stats[0] > 0:  # l10_games_played > 0
        # Calculate goal differential percentage from L10 data
        l10_goal_diff_pct = l10_stats[7] / max(l10_stats[0], 1) if l10_stats[0] is not None and l10_stats[0] > 0 else 0
        
        return {
            "roll_wins": l10_stats[1],    
            "roll_loss": l10_stats[2],
            "roll_ot_loss": l10_stats[3],
            "roll_pts": l10_stats[4],      
            "roll_goals_for": l10_stats[5],
            "roll_goals_against": l10_stats[6],   
            "roll_goal_diff": l10_stats[7],
            "roll_goal_diff_pct": l10_goal_diff_pct,
            "roll_regulation_wins": l10_stats[8],
            "roll_regulation_plus_ot_wins": l10_stats[9],
            "roll_league_sequence": l10_stats[10],
            "roll_conference_sequence": l10_stats[11],
            "roll_division_sequence": l10_stats[12]
        }
    
    # fallback to manual rolling calculation if L10 stats not available
    # Try to get comprehensive stats first, fallback to basic if not available
    try:
        recent_stats = conn.execute(text(
            """
            SELECT points, wins, losses, COALESCE(ot_losses, 0), goals_for, goals_against, goal_diff, goal_diff_pct,
                   home_wins, home_losses, home_ot_losses, home_points, home_goal_for, home_goal_against, home_goal_diff,
                   road_wins, road_losses, road_ot_losses, road_points, road_goals_for, road_goals_against, road_goal_diff,
                   COALESCE(regulation_wins, 0), COALESCE(regulation_plus_ot_wins, 0), home_regulation_wins, home_regulation_plus_ot_wins,
                   road_regulation_wins, road_regulation_plus_ot_wins, 
                   COALESCE(league_sequence, 0), COALESCE(conference_sequence, 0), COALESCE(division_sequence, 0)
            FROM team_stats
            WHERE team_id = :t AND date < :cutoff
            ORDER BY date DESC LIMIT 1
            """
        ), {"t": team_id, "cutoff": cutoff}).fetchone()
    except Exception:
        # Fallback to basic columns if comprehensive ones don't exist
        recent_stats = conn.execute(text(
            """
            SELECT points, wins, losses, 0 as ot_losses, goals_for, goals_against, goal_diff, goal_diff_pct,
                   home_wins, home_losses, home_ot_losses, home_points, home_goal_for, home_goal_against, home_goal_diff,
                   road_wins, road_losses, road_ot_losses, road_points, road_goals_for, road_goals_against, road_goal_diff,
                   0 as regulation_wins, 0 as regulation_plus_ot_wins, home_regulation_wins, home_regulation_plus_ot_wins,
                   road_regulation_wins, road_regulation_plus_ot_wins, 
                   0 as league_sequence, 0 as conference_sequence, 0 as division_sequence
            FROM team_stats
            WHERE team_id = :t AND date < :cutoff
            ORDER BY date DESC LIMIT 1
            """
        ), {"t": team_id, "cutoff": cutoff}).fetchone()
    
    if not recent_stats:
        return {
            "roll_wins": None,
            "roll_loss": None,
            "roll_ot_loss": None,
            "roll_pts": None,
            "roll_goals_for": None,
            "roll_goals_against": None,
            "roll_goal_diff": None,
            "roll_goal_diff_pct": None,
            "roll_regulation_wins": None,
            "roll_regulation_plus_ot_wins": None,
            "roll_league_sequence": None,
            "roll_conference_sequence": None,
            "roll_division_sequence": None
        }
    
    # get stats from window games ago (approximate by going back window days)
    window_date = cutoff - timedelta(days=window * 2)  # Approximate: 2 days per game
    try:
        old_stats = conn.execute(text(
            """
            SELECT points, wins, losses, COALESCE(ot_losses, 0), goals_for, goals_against, goal_diff,
                   home_wins, home_losses, home_ot_losses, home_points, home_goal_for, home_goal_against, home_goal_diff,
                   road_wins, road_losses, road_ot_losses, road_points, road_goals_for, road_goals_against, road_goal_diff,
                   COALESCE(regulation_wins, 0), COALESCE(regulation_plus_ot_wins, 0), home_regulation_wins, home_regulation_plus_ot_wins,
                   road_regulation_wins, road_regulation_plus_ot_wins, 
                   COALESCE(league_sequence, 0), COALESCE(conference_sequence, 0), COALESCE(division_sequence, 0)
            FROM team_stats
            WHERE team_id = :t AND date <= :window_date
            ORDER BY date DESC LIMIT 1
            """
        ), {"t": team_id, "window_date": window_date}).fetchone()
    except Exception:
        # Fallback to basic columns if comprehensive ones don't exist
        old_stats = conn.execute(text(
            """
            SELECT points, wins, losses, 0 as ot_losses, goals_for, goals_against, goal_diff,
                   home_wins, home_losses, home_ot_losses, home_points, home_goal_for, home_goal_against, home_goal_diff,
                   road_wins, road_losses, road_ot_losses, road_points, road_goals_for, road_goals_against, road_goal_diff,
                   0 as regulation_wins, 0 as regulation_plus_ot_wins, home_regulation_wins, home_regulation_plus_ot_wins,
                   road_regulation_wins, road_regulation_plus_ot_wins, 
                   0 as league_sequence, 0 as conference_sequence, 0 as division_sequence
            FROM team_stats
            WHERE team_id = :t AND date <= :window_date
            ORDER BY date DESC LIMIT 1
            """
        ), {"t": team_id, "window_date": window_date}).fetchone()
    
    if not old_stats:
        # if no old stats, use current season start (all stats are rolling from season start)
        return {
            "roll_wins": recent_stats[1], 
            "roll_loss": recent_stats[2],
            "roll_ot_loss": recent_stats[3],
            "roll_pts": recent_stats[0],
            "roll_goals_for": recent_stats[4],
            "roll_goals_against": recent_stats[5],
            "roll_goal_diff": recent_stats[6],
            "roll_goal_diff_pct": recent_stats[7],
            "roll_regulation_wins": recent_stats[23],  # regulation_wins
            "roll_regulation_plus_ot_wins": recent_stats[24],  # regulation_plus_ot_wins
            "roll_league_sequence": recent_stats[28],  # league_sequence
            "roll_conference_sequence": recent_stats[29],  # conference_sequence
            "roll_division_sequence": recent_stats[30]  # division_sequence
        }
    
    # Calculate rolling window differences for all stats
    roll_pts = recent_stats[0] - old_stats[0]
    roll_wins = recent_stats[1] - old_stats[1]
    roll_loss = recent_stats[2] - old_stats[2]
    roll_ot_loss = recent_stats[3] - old_stats[3]
    roll_gf = recent_stats[4] - old_stats[4]
    roll_ga = recent_stats[5] - old_stats[5]
    roll_goal_diff = recent_stats[6] - old_stats[6]
    
    # Calculate goal differential percentage for the rolling window
    roll_games = max(abs(roll_wins) + abs(roll_loss) + abs(roll_ot_loss), 1)  # Approximate games played
    roll_goal_diff_pct = roll_goal_diff / roll_games if roll_games > 0 else 0
    
    # Calculate regulation wins rolling stats
    roll_regulation_wins = recent_stats[23] - old_stats[23]  # regulation_wins
    roll_regulation_plus_ot_wins = recent_stats[24] - old_stats[24]  # regulation_plus_ot_wins
    
    # Note: Sequences are current standings, not rolling - we'll use current values
    # In a real rolling window, these would be the current sequence values
    roll_league_sequence = recent_stats[28]  # Current league sequence
    roll_conference_sequence = recent_stats[29]  # Current conference sequence  
    roll_division_sequence = recent_stats[30]  # Current division sequence
        
    return {
        "roll_wins": roll_wins,
        "roll_loss": roll_loss,
        "roll_ot_loss": roll_ot_loss,
        "roll_pts": roll_pts,
        "roll_goals_for": roll_gf,
        "roll_goals_against": roll_ga,
        "roll_goal_diff": roll_goal_diff,
        "roll_goal_diff_pct": roll_goal_diff_pct,
        "roll_regulation_wins": roll_regulation_wins,
        "roll_regulation_plus_ot_wins": roll_regulation_plus_ot_wins,
        "roll_league_sequence": roll_league_sequence,
        "roll_conference_sequence": roll_conference_sequence,
        "roll_division_sequence": roll_division_sequence
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
        home_context = get_current_contextual_stats(conn, home, cutoff)
        
        away_rest = compute_rest_days(conn, away, cutoff)
        away_form = compute_rolling_form(conn, away, cutoff, 10)
        away_context = get_current_contextual_stats(conn, away, cutoff)
        
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
            
            # rolling form stats
            "home_roll_pts": home_form["roll_pts"], "away_roll_pts": away_form["roll_pts"],
            "home_roll_w": home_form["roll_wins"], "away_roll_w": away_form["roll_wins"],
            "home_roll_l": home_form["roll_loss"], "away_roll_l": away_form["roll_loss"],
            "home_roll_ot_l": home_form["roll_ot_loss"], "away_roll_ot_l": away_form["roll_ot_loss"],
            "home_roll_gf": home_form["roll_goals_for"], "away_roll_gf": away_form["roll_goals_for"],
            "home_roll_ga": home_form["roll_goals_against"], "away_roll_ga": away_form["roll_goals_against"],
            "home_roll_goal_diff": home_form["roll_goal_diff"], "away_roll_goal_diff": away_form["roll_goal_diff"],
            "home_roll_goal_diff_pct": home_form["roll_goal_diff_pct"], "away_roll_goal_diff_pct": away_form["roll_goal_diff_pct"],
            "home_roll_reg_wins": home_form["roll_regulation_wins"], "away_roll_reg_wins": away_form["roll_regulation_wins"],
            "home_roll_reg_ot_wins": home_form["roll_regulation_plus_ot_wins"], "away_roll_reg_ot_wins": away_form["roll_regulation_plus_ot_wins"],
            "home_roll_league_seq": home_form["roll_league_sequence"], "away_roll_league_seq": away_form["roll_league_sequence"],
            "home_roll_conf_seq": home_form["roll_conference_sequence"], "away_roll_conf_seq": away_form["roll_conference_sequence"],
            "home_roll_div_seq": home_form["roll_division_sequence"], "away_roll_div_seq": away_form["roll_division_sequence"],
            
            # contextual stats
            "home_league_seq": home_context["league_sequence"], "away_league_seq": away_context["league_sequence"],
            "home_conf_seq": home_context["conference_sequence"], "away_conf_seq": away_context["conference_sequence"],
            "home_div_seq": home_context["division_sequence"], "away_div_seq": away_context["division_sequence"],
            "home_league_home_seq": home_context["league_home_sequence"], "away_league_home_seq": away_context["league_home_sequence"],
            "home_league_road_seq": home_context["league_road_sequence"], "away_league_road_seq": away_context["league_road_sequence"],
            "home_conf_home_seq": home_context["conference_home_sequence"], "away_conf_home_seq": away_context["conference_home_sequence"],
            "home_conf_road_seq": home_context["conference_road_sequence"], "away_conf_road_seq": away_context["conference_road_sequence"],
            "home_div_home_seq": home_context["division_home_sequence"], "away_div_home_seq": away_context["division_home_sequence"],
            "home_div_road_seq": home_context["division_road_sequence"], "away_div_road_seq": away_context["division_road_sequence"],
            
            # Streak information
            "home_streak_code": home_context["streak_code"], "away_streak_code": away_context["streak_code"],
            "home_streak_count": home_context["streak_count"], "away_streak_count": away_context["streak_count"],
            
            # Home/road situational stats
            "home_home_wins": home_context["home_wins"], "away_home_wins": away_context["home_wins"],
            "home_home_losses": home_context["home_losses"], "away_home_losses": away_context["home_losses"],
            "home_home_ot_losses": home_context["home_ot_losses"], "away_home_ot_losses": away_context["home_ot_losses"],
            "home_home_points": home_context["home_points"], "away_home_points": away_context["home_points"],
            "home_home_gf": home_context["home_goal_for"], "away_home_gf": away_context["home_goal_for"],
            "home_home_ga": home_context["home_goal_against"], "away_home_ga": away_context["home_goal_against"],
            "home_home_goal_diff": home_context["home_goal_diff"], "away_home_goal_diff": away_context["home_goal_diff"],
            "home_home_reg_wins": home_context["home_regulation_wins"], "away_home_reg_wins": away_context["home_regulation_wins"],
            "home_home_reg_ot_wins": home_context["home_regulation_plus_ot_wins"], "away_home_reg_ot_wins": away_context["home_regulation_plus_ot_wins"],
            
            "home_road_wins": home_context["road_wins"], "away_road_wins": away_context["road_wins"],
            "home_road_losses": home_context["road_losses"], "away_road_losses": away_context["road_losses"],
            "home_road_ot_losses": home_context["road_ot_losses"], "away_road_ot_losses": away_context["road_ot_losses"],
            "home_road_points": home_context["road_points"], "away_road_points": away_context["road_points"],
            "home_road_gf": home_context["road_goals_for"], "away_road_gf": away_context["road_goals_for"],
            "home_road_ga": home_context["road_goals_against"], "away_road_ga": away_context["road_goals_against"],
            "home_road_goal_diff": home_context["road_goal_diff"], "away_road_goal_diff": away_context["road_goal_diff"],
            "home_road_reg_wins": home_context["road_regulation_wins"], "away_road_reg_wins": away_context["road_regulation_wins"],
            "home_road_reg_ot_wins": home_context["road_regulation_plus_ot_wins"], "away_road_reg_ot_wins": away_context["road_regulation_plus_ot_wins"],
            
            # overall season stats
            "home_games_played": home_context["games_played"], "away_games_played": away_context["games_played"],
            "home_points": home_context["points"], "away_points": away_context["points"],
            "home_win_pct": home_context["win_pct"], "away_win_pct": away_context["win_pct"],
            "home_goal_diff_pct": home_context["goal_diff_pct"], "away_goal_diff_pct": away_context["goal_diff_pct"],
        }
        
        conn.execute(text(
            """
          INSERT INTO feature_snapshots (game_id, as_of_ts_utc, feature_json)
          VALUES (:gid, :ts, :json)
          ON CONFLICT (game_id, as_of_ts_utc) DO NOTHING
        """), {"gid": game_id, "ts": cutoff, "json": json.dumps(convert_decimals_to_floats(feature))})
        
        