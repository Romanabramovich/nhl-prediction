# jobs/backfill.py
import argparse
import datetime as dt
from sqlalchemy import text
from dateutil import rrule
from ingest.loaders import schedule, standings, teams, rosters, labels
from database import test_connection, get_database_engine
from features.build_snapshot import build_feature_snapshot

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--season", required=True, help="season start year (2022 for 2022-2023)")
    return p.parse_args()

def upsert_games_in_season(season, engine):
    print(f"upserting games for season: {season} - {str(int(season) + 1)}")
    schedule.upsert_games_for_season(season, engine)
    
def upsert_teams_in_season(season, engine):
    print(f"upserting teams for season: {season} - {str(int(season) + 1)}")
    teams.upsert_teams_for_season(season, engine)
    
    
def ingest_daily_data(start_date, end_date, engine):
    print(f"ingesting daily team stats from {start_date} to {end_date}")
    
    for day in rrule.rrule(rrule.DAILY, dtstart=start_date, until=end_date):
        d = day.date()
        print(f"processing stats on: {d}")
        
        # load standings snapshot
        standings.upsert_standings_on_date(d, engine)
        
def build_snapshots_for_games(season, engine):
    print(f"building feature snapshots for season {season}")
    
    # Convert season format from "2015" to "20152016"
    season_format = f"{season}{int(season) + 1}"
    print(f"Looking for games with season: {season_format}")
    
    with engine.begin() as conn:
        games = conn.execute(text(
            """
            SELECT game_id, game_date
            FROM games
            WHERE season = :s
            """
        ), {"s": season_format}).fetchall()
        
    print(f"Found {len(games)} games to process for feature building")
    
    for game_id, game_date in games:
        game_ts = game_date
        
        # Set cutoff to 12 PM on the day before the game
        cutoff_utc = (game_ts - dt.timedelta(days=1)).replace(hour=12, minute=0, second=0, microsecond=0)
        
        print(f"Game {game_id}: game_time={game_ts}, cutoff={cutoff_utc}")

        if cutoff_utc >= game_ts:
            print(f"skipping game {game_id}: cutoff >= start time")
            continue

        print(f"building features for game {game_id} with cutoff {cutoff_utc}...")
        build_feature_snapshot(season, game_id, cutoff_utc, engine)
        
    
def fill_labels_for_completed_games(season, engine):
    print(f"filling labels for completed games in season: {season}")
    
    # Convert season format from "2015" to "20152016"
    season_format = f"{season}{int(season) + 1}"
    print(f"Looking for completed games with season: {season_format}")
    
    with engine.begin() as conn:
        games = conn.execute(text(
            """
            SELECT game_id
            FROM games
            WHERE season = :s AND status IN ('OFF', 'FINAL')
            """
        ), {"s": season_format}).fetchall()
        
        print(f"Found {len(games)} completed games (status='OFF' or 'FINAL') to process for labels")
        
        game_ids = [game[0] for game in games]
        labels.update_label_for_final_games(game_ids, engine)
    
def main():
    if not test_connection():
        return
    
    # connect db engine
    engine = get_database_engine()
    
    # read CLI args
    args = parse_args()
    
    season = args.season
    
    #define season window
    start_date = dt.date(int(season), 10, 1)
    end_date = dt.date(int(season) + 1, 6, 30)
    
    # upsert league schedule for the season
    upsert_games_in_season(season, engine)
    
    # upsert teams for the season
    upsert_teams_in_season(season, engine)
    
    # ingest daily standings
    ingest_daily_data(start_date, end_date, engine)
    
    # build feature snapshots for in-season data
    build_snapshots_for_games(season, engine)
    
    # fill labels
    fill_labels_for_completed_games(season, engine)
    
    print("backfill complete")
    
if __name__ == "__main__":
    main()