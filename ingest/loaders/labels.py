# ingest/loaders/labels.py
from ingest.clients.nhl import NHLClient
from database import get_database_engine
from ingest.loaders.standings import ABBREV_TO_ID

from sqlalchemy import text

def update_label_for_final_games(game_ids: list[int], engine = None):
    
    if engine is None:
        engine = get_database_engine()
    
    client = NHLClient()
    
    with engine.begin() as conn:
        for game_id in game_ids:
            payload = client.get(f"/v1/gamecenter/{game_id}/boxscore")
            
            state = payload.get("gameState")
            if state != "OFF":
                continue
                
            home_score = payload["homeTeam"]["score"]
            away_score = payload["awayTeam"]["score"]
            home_id = ABBREV_TO_ID[payload["homeTeam"]["abbrev"]]
            away_id = ABBREV_TO_ID[payload["awayTeam"]["abbrev"]]
            winner = home_id if home_score > away_score else away_id
            home_win = winner == home_id 
            
            conn.execute(text(
                """
                INSERT INTO labels 
                    (game_id, home_score, away_score, home_id, away_id, home_win)
                VALUES 
                    (:game_id, :home_score, :away_score, :home_id, :away_id, :home_win)
                ON CONFLICT (game_id) DO UPDATE
                    SET home_score = EXCLUDED.home_score,
                    away_score = EXCLUDED.away_score,
                    home_id = EXCLUDED.home_id,
                    away_id = EXCLUDED.away_id,
                    home_win = EXCLUDED.home_win
                """
            ), {"game_id":game_id, "home_score":home_score, "away_score": away_score, "home_id":home_id, "away_id":away_id, "home_win":home_win})
            