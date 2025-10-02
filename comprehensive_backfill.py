# comprehensive_backfill.py
import argparse
import datetime as dt
from sqlalchemy import text
from dateutil import rrule
from ingest.loaders import schedule, standings, teams, rosters, labels
from database import test_connection, get_database_engine
from features.build_snapshot import build_feature_snapshot

def parse_args():
    p = argparse.ArgumentParser(description="Comprehensive backfill with new feature extraction")
    p.add_argument("--season", required=True, help="season start year (2022 for 2022-2023)")
    p.add_argument("--skip-standings", action="store_true", help="skip standings backfill (use existing data)")
    p.add_argument("--skip-snapshots", action="store_true", help="skip snapshot building (use existing data)")
    p.add_argument("--skip-labels", action="store_true", help="skip label building")
    p.add_argument("--dry-run", action="store_true", help="show what would be processed without making changes")
    return p.parse_args()

def check_current_data_status(engine):
    # check what data we currently have
    print("CHECKING CURRENT DATA STATUS")
    print("=" * 40)
    
    with engine.connect() as conn:
        # Check team_stats
        result = conn.execute(text("SELECT COUNT(*) as count, MIN(date) as min_date, MAX(date) as max_date FROM team_stats")).fetchone()
        print(f"TEAM_STATS: {result[0]:,} records ({result[1]} to {result[2]})")
        
        # Check feature_snapshots
        result = conn.execute(text("SELECT COUNT(*) as count, MIN(as_of_ts_utc) as min_date, MAX(as_of_ts_utc) as max_date FROM feature_snapshots")).fetchone()
        print(f"FEATURE_SNAPSHOTS: {result[0]:,} records ({result[1]} to {result[2]})")
        
        # Check sample feature structure
        sample = conn.execute(text("SELECT feature_json FROM feature_snapshots LIMIT 1")).fetchone()
        if sample:
            features = sample[0]  # feature_json is already a dict, not a JSON string
            print(f"CURRENT FEATURES: {len(features)} features")
            print(f"Sample: {list(features.keys())[:5]}...")
        else:
            print("CURRENT FEATURES: No snapshots found")

def upsert_teams_in_season(season, engine):
    print(f"\n UPDATING TEAMS FOR SEASON {season}-{int(season)+1}")
    print("-" * 45)
    teams.upsert_teams_for_season(season, engine)

def upsert_games_in_season(season, engine):
    print(f"\nUPDATING SCHEDULE FOR SEASON {season}-{int(season)+1}")
    print("-" * 50)
    schedule.upsert_games_for_season(season, engine)

def ingest_daily_standings(start_date, end_date, engine, dry_run=False):
    # ingest standings data with new stats
    print(f"\nINGESTING COMPREHENSIVE STANDINGS DATA")
    print(f"   Period: {start_date} to {end_date}")
    print("-" * 45)
    
    if dry_run:
        print("DRY RUN: Would process the following dates:")
        for day in rrule.rrule(rrule.DAILY, dtstart=start_date, until=end_date):
            if day.weekday() < 5:  # only show weekdays  
                print(f"   • {day.date()}")
        print(f"   ... and {sum(1 for _ in rrule.rrule(rrule.DAILY, dtstart=start_date, until=end_date)) - 5} more dates")
        return
    
    total_days = 0
    for day in rrule.rrule(rrule.DAILY, dtstart=start_date, until=end_date):
        d = day.date()
        if total_days % 30 == 0:  # progress update every 30 days
            print(f"   Processing: {d} ({total_days} days completed)")
        
        standings.upsert_standings_on_date(d, engine)
        total_days += 1
    
    print(f"Completed standings ingestion: {total_days} days")

def build_comprehensive_snapshots(season, engine, dry_run=False):
    # build feature snapshots with new comprehensive features
    print(f"\nBUILDING FEATURE SNAPSHOTS")
    print(f"   Season: {season}-{int(season)+1}")
    print("-" * 45)
    
    # Convert season format from "2015" to "20152016"
    season_format = f"{season}{int(season) + 1}"
    
    with engine.connect() as conn:
        games = conn.execute(text(
            """
            SELECT game_id, game_date, home_team_id, away_team_id
            FROM games
            WHERE season = :s
            ORDER BY game_date
            """
        ), {"s": season_format}).fetchall()
    
    print(f"Found {len(games)} games to process")
    
    if dry_run:
        print("DRY RUN: Would process these games:")
        for i, (game_id, game_date, home_id, away_id) in enumerate(games[:10]):
            print(f"   • Game {game_id}: {game_date} (Home: {home_id}, Away: {away_id})")
        if len(games) > 10:
            print(f"   ... and {len(games) - 10} more games")
        return
    
    processed = 0
    for game_id, game_date, home_id, away_id in games:
        game_ts = game_date
        
        # Set cutoff to 12 PM on the day before the game
        cutoff_utc = (game_ts - dt.timedelta(days=1)).replace(hour=12, minute=0, second=0, microsecond=0)
        
        if cutoff_utc >= game_ts:
            print(f"   Skipping game {game_id}: cutoff >= start time")
            continue
        
        if processed % 100 == 0:  # Progress update every 100 games
            print(f"   Processing game {game_id} ({processed}/{len(games)} completed)")
        
        build_feature_snapshot(season, game_id, cutoff_utc, engine)
        processed += 1
    
    print(f"✅ Completed snapshot building: {processed} games processed")

def fill_labels_for_completed_games(season, engine, dry_run=False):
    # fill labels for completed games
    print(f"\nFILLING LABELS FOR COMPLETED GAMES")
    print(f"   Season: {season}-{int(season)+1}")
    print("-" * 35)
    
    season_format = f"{season}{int(season) + 1}"
    
    with engine.connect() as conn:
        games = conn.execute(text(
            """
            SELECT game_id, game_date, status
            FROM games
            WHERE season = :s AND status IN ('OFF', 'FINAL')
            ORDER BY game_date
            """
        ), {"s": season_format}).fetchall()
    
    print(f"Found {len(games)} completed games")
    
    if dry_run:
        print("DRY RUN: Would process these completed games:")
        for game_id, game_date, status in games[:5]:
            print(f"   • Game {game_id}: {game_date} (Status: {status})")
        if len(games) > 5:
            print(f"   ... and {len(games) - 5} more games")
        return
    
    game_ids = [game[0] for game in games]
    labels.update_label_for_final_games(game_ids, engine)
    print(f"Completed label filling: {len(game_ids)} games")

def main():
    if not test_connection():
        print("Database connection failed")
        return
    
    # Connect to database
    engine = get_database_engine()
    
    # Parse arguments
    args = parse_args()
    season = args.season
    
    # Check current data status
    check_current_data_status(engine)
    
    # Define season window
    start_date = dt.date(int(season), 10, 1)
    end_date = dt.date(int(season) + 1, 6, 30)
    
    print(f"\nSTARTING COMPREHENSIVE BACKFILL")
    print(f"   Season: {season}-{int(season)+1}")
    print(f"   Period: {start_date} to {end_date}")
    print(f"   Dry run: {args.dry_run}")
    print("=" * 50)
    
    # Execute backfill steps
    if not args.skip_standings:
        upsert_teams_in_season(season, engine)
        upsert_games_in_season(season, engine)
        ingest_daily_standings(start_date, end_date, engine, args.dry_run)
    
    if not args.skip_snapshots:
        build_comprehensive_snapshots(season, engine, args.dry_run)
    
    if not args.skip_labels:
        fill_labels_for_completed_games(season, engine, args.dry_run)
    
    print(f"\n BACKFILL COMPLETED!")
    print("=" * 40)
    

if __name__ == "__main__":
    main()
