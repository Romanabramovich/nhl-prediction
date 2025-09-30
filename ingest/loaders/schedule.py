# ingest/loaders/schedule.py
from ingest.clients.nhl import NHLClient
from database import get_database_engine
from sqlalchemy import text
from datetime import datetime, date
from dateutil import rrule

def upsert_games(date: str, engine = None):
    # /v1/schedule/YYYY-MM-DD
    # retrieves schedule for specific date
    
    if engine is None:
        engine = get_database_engine()
        
    client = NHLClient()
    payload = client.get(f"/v1/schedule/{date}")
    rows = []
    
    # game type lambda map
    game_type_map = lambda gt:(
        "PRESEASON" if gt == 1 else
        "REGULAR" if gt == 2 else
        "POSTSEASON"
    )

    week = payload.get("gameWeek", [])[0] if payload.get("gameWeek") else None
    if week:
        for game in week.get("games", []):
            home_team_id    = game["homeTeam"]["id"]
            away_team_id    = game["awayTeam"]["id"]
            
            # skip games with special teams (All-Star teams, etc.) - team_id > 60
            if home_team_id > 60 or away_team_id > 60:
                continue
                
            game_id         = game["id"]
            season          = game["season"]
            home_team_name  = game["homeTeam"]["abbrev"]
            away_team_name  = game["awayTeam"]["abbrev"]
            game_type       = game["gameType"]
            game_status     = game["gameState"]
            venue           = game.get("venue", {}).get("default", "")
            start_iso       = game["startTimeUTC"]
            game_date       = datetime.fromisoformat(start_iso.replace("Z", "+00:00"))
            
            rows.append(
                {
                "game_id": game_id, 
                "season": season,
                "home_team_id": home_team_id,
                "home_team_name": home_team_name,
                "away_team_id": away_team_id,
                "away_team_name": away_team_name,
                "status" : game_status,
                "game_type": game_type_map(game_type), 
                "venue" : venue,
                "game_date": game_date
            })
    
    with engine.begin() as conn:
        for r in rows:
            if r["home_team_id"] < 60:
                conn.execute(text (
                    """
                    INSERT INTO games(
                        game_id, season, home_team_id, home_team_name, 
                        away_team_id, away_team_name, game_type, game_date, venue, status 
                    )
                    VALUES (
                        :game_id, :season, :home_team_id, :home_team_name, :away_team_id,
                        :away_team_name, :game_type, :game_date, :venue, :status 
                    )
                    ON CONFLICT (game_id) DO UPDATE
                    SET season = EXCLUDED.season,
                        home_team_id    = EXCLUDED.home_team_id,
                        home_team_name  = EXCLUDED.home_team_name,
                        away_team_id    = EXCLUDED.away_team_id,
                        away_team_name  = EXCLUDED.away_team_name,
                        game_type       = EXCLUDED.game_type,
                        status          = EXCLUDED.status,
                        venue           = EXCLUDED.venue,
                        game_date       = EXCLUDED.game_date
                    """
                ), r)

def upsert_games_for_season(season: str, engine = None):
    # load games for an entire season by iterating through the season dates
    # NHL seasons typically run from October to June
    
    if engine is None:
        engine = get_database_engine()
    
    # define season window (October 1 to June 30)
    start_date = date(int(season), 10, 1)
    end_date = date(int(season) + 1, 6, 30)
    
    print(f"loading schedule for season {season}-{int(season)+1} from {start_date} to {end_date}")
    
    # iterate through each day in the season
    for day in rrule.rrule(rrule.DAILY, dtstart=start_date, until=end_date):
        day_str = day.strftime("%Y-%m-%d")
        print(f"loading games for {day_str}")
        upsert_games(day_str, engine)
    