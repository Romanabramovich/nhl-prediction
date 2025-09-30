# ingest/loaders/standings.py
from ingest.clients.nhl import NHLClient
from database import get_database_engine
from sqlalchemy import text

ABBREV_TO_ID = {
    "NJD": 1,
    "NYI": 2,
    "NYR": 3,
    "PHI": 4,
    "PIT": 5,
    "BOS": 6,
    "BUF": 7,
    "MTL": 8,
    "OTT": 9,
    "TOR": 10,
    "ATL": 11, # atlanta thrashers
    "CAR": 12,
    "FLA": 13,
    "TBL": 14,
    "WSH": 15,
    "CHI": 16,
    "DET": 17,
    "NSH": 18,
    "STL": 19,
    "CGY": 20,
    "COL": 21,
    "EDM": 22,
    "VAN": 23,
    "ANA": 24,
    "DAL": 25,
    "LAK": 26,
    "SJS": 28,
    "CBJ": 29,
    "MIN": 30,
    "WPG": 52,
    "ARI": 53,
    "VGK": 54,
    "SEA": 55,
    "UTA": 59, #utah hockey club
}


def upsert_standings_on_date(date: str, engine = None):
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
        points      = team.get("points", 0)
        wins        = team.get("wins", 0)
        losses      = team.get("losses", 0)
        goals_for   = team.get("goalFor", 0)
        goals_against = team.get("goalAgainst", 0)
          
        rows.append(
            {
                "date": date,
                "team_id": team_id,
                "team_name": team_name, 
                "team_abbrev" : team_abbrev, 
                "conference": conference,
                "division": division, 
                "points": points,
                "wins": wins,
                "losses": losses, 
                "goals_for": goals_for, 
                "goals_against": goals_against
            }
        )
        
    with engine.begin() as conn:
        for r in rows:
            conn.execute(text(
                """
                INSERT INTO team_stats(
                    date, team_id, team_name, team_abbrev, conference, division, points, wins, losses, goals_for, goals_against
                )
                VALUES(
                    :date, :team_id, :team_name, :team_abbrev, :conference, :division, :points, :wins, :losses, :goals_for, :goals_against
                )
                ON CONFLICT (team_id, date) DO UPDATE
                SET team_name = EXCLUDED.team_name,
                    team_abbrev = EXCLUDED.team_abbrev,
                    conference = EXCLUDED.conference,
                    division = EXCLUDED.division,
                    points = EXCLUDED.points,
                    wins = EXCLUDED.wins,
                    losses = EXCLUDED.losses,
                    goals_for = EXCLUDED.goals_for,
                    goals_against = EXCLUDED.goals_against
                """
            ), r)

