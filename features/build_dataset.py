# features/build_dataset.py
from database import get_database_engine
from sqlalchemy import text


def create_modeling_view(engine=None) -> None:
    """
    Create a SQL view `modeling_dataset` that joins feature_snapshots, games, and labels
    and filters to regular-season games only.

    Columns included are sufficient for training:
      - game_id, season, game_date
      - as_of_ts_utc
      - feature_json (JSON of pre-game features)
      - home_win (bool label)
    """
    if engine is None:
        engine = get_database_engine()

    with engine.begin() as conn:
        # Drop view if exists, then create
        conn.execute(text("DROP VIEW IF EXISTS modeling_dataset"))
        conn.execute(text(
            """
            CREATE VIEW modeling_dataset AS
            SELECT
              g.game_id,
              g.season,
              g.game_date,
              fs.as_of_ts_utc,
              fs.feature_json,
              l.home_win
            FROM feature_snapshots fs
            JOIN games g ON g.game_id = fs.game_id
            JOIN labels l ON l.game_id = fs.game_id
            WHERE g.game_type = 'REGULAR'
            """
        ))


if __name__ == "__main__":
    create_modeling_view()

