# ingest/loaders/rosters.py
from ingest.clients.nhl import NHLClient
from database import get_database_engine
from sqlalchemy import text


def upsert_rosters(team: str, season: str, engine = None):
    # v1/roster/{ABBREV}/{SEASON}
    # /v1/roster/TOR/20232024
    
    if engine is None:
        engine = get_database_engine()
    
    client = NHLClient()
    
    season = season + str(int(season) + 1)
    payload = client.get(f"/v1/roster/{team}/{season}")
    rows = []
    
    #forwards, defensemen, goalies
    for p in payload.get("forwards", []):
        rows.append(
            {
                "team_id": int(team),
                "player_id": p["id"],
                "first_name": p["firstName"]["default"],
                "last_name": p["lastName"]["default"],
                "jersey": p["sweaterNumber"], 
                "height": p["heightInCentimeters"],
                "weight": p["weightInKilograms"],
                "dominant_hand": p["shootsCatches"],
                "position": p["positionCode"]
            }
        )
        
    for p in payload.get("defensemen", []):
        rows.append(
            {
                "team_id": int(team),
                "player_id": p["id"],
                "first_name": p["firstName"]["default"],
                "last_name": p["lastName"]["default"],
                "jersey": p["sweaterNumber"], 
                "height": p["heightInCentimeters"],
                "weight": p["weightInKilograms"],
                "dominant_hand": p["shootsCatches"],
                "position": p["positionCode"]
            }
        )
    
    for p in payload.get("goalies", []):
        rows.append(
            {
                "team_id": int(team),
                "player_id": p["id"],
                "first_name": p["firstName"]["default"],
                "last_name": p["lastName"]["default"],
                "jersey": p["sweaterNumber"], 
                "height": p["heightInCentimeters"],
                "weight": p["weightInKilograms"],
                "dominant_hand": p["shootsCatches"],
                "position": p["positionCode"]
            }
        )
    
    
    with engine.begin() as conn:
        for r in rows:
            conn.execute(text(
                """
                INSERT INTO rosters
                    (team_id, player_id, first_name, last_name, jersey, height, weight, dominant_hand, position)
                VALUES
                    (:team_id, :player_id, :first_name, :last_name, :jersey, :height, :weight, :dominant_hand, :position)
                ON CONFLICT DO NOTHING
                """
            ), r)