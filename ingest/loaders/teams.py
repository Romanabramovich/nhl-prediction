# ingest/loaders/standings.py
from ingest.clients.nhl import NHLClient
from database import get_database_engine
from ingest.loaders.standings import ABBREV_TO_ID
from sqlalchemy import text
from datetime import date
from dateutil import rrule


def upsert_standings(date: str, engine = None):
    # /v1/standings/YYYY-MM-DD
    
    if engine is None:
        engine = get_database_engine()
    
    client = NHLClient()
    payload = client.get(f"/v1/standings/{date}")
    rows = []
    
    for team in payload.get("standings"):
        team_id     = ABBREV_TO_ID[team["teamAbbrev"]["default"]]
        team_name   = team["teamName"]["default"]
        team_abbrev = team["teamAbbrev"]["default"]
        conference  = team["conferenceName"]
        division    = team["divisionName"]
          
        rows.append(
            {
                "team_id": team_id,
                "team_name": team_name, 
                "team_abbrev" : team_abbrev, 
                "conference": conference,
                "division": division
            }
        )
        
    with engine.begin() as conn:
        for r in rows:
            conn.execute(text(
                """
                INSERT INTO teams(
                    team_id, team_name, team_abbrev, conference, division
                )
                VALUES(
                    :team_id, :team_name, :team_abbrev, :conference, :division
                )
                ON CONFLICT (team_id) DO UPDATE
                SET team_name = EXCLUDED.team_name,
                    team_abbrev = EXCLUDED.team_abbrev,
                    conference = EXCLUDED.conference,
                    division = EXCLUDED.division
                """
            ), r)

def upsert_teams_for_season(season: str, engine = None):
    # load teams for an entire season by getting standings early in the season
    # this will capture all teams that participated in the season
    
    if engine is None:
        engine = get_database_engine()
    
    # get standings early in the season (around October 15) to ensure teams are loaded before daily stats
    early_date = date(int(season), 10, 15)
    date_str = early_date.strftime("%Y-%m-%d")
    
    print(f"loading teams for season {season}-{int(season)+1} using standings from {date_str}")
    upsert_standings(date_str, engine)