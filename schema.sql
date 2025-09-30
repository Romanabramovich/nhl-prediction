-- NHL Prediction Database Schema
-- Enable UUID extension for unique identifiers
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- CORE NHL DATA TABLES

-- teams table - stores team information
CREATE TABLE teams (
    team_id INTEGER PRIMARY KEY,
    team_name VARCHAR(100) NOT NULL,
    team_abbrev VARCHAR(10) UNIQUE NOT NULL,
    conference VARCHAR(50) NOT NULL,
    division VARCHAR(50) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- games table - stores game schedule and metadata
CREATE TABLE games (
    game_id BIGINT PRIMARY KEY,
    season VARCHAR(20) NOT NULL,
    home_team_id INTEGER NOT NULL REFERENCES teams(team_id),
    home_team_name VARCHAR(10) NOT NULL,
    away_team_id INTEGER NOT NULL REFERENCES teams(team_id),
    away_team_name VARCHAR(10) NOT NULL,
    status VARCHAR(20) NOT NULL DEFAULT 'SCHEDULED',
    game_type VARCHAR(20) NOT NULL CHECK (game_type IN ('PRESEASON', 'REGULAR', 'POSTSEASON')),
    venue VARCHAR(200),
    game_date TIMESTAMP WITH TIME ZONE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT different_teams CHECK (home_team_id != away_team_id)
);

-- team stats table - daily team performance metrics
CREATE TABLE team_stats (
    id SERIAL PRIMARY KEY,
    date DATE NOT NULL,
    team_id INTEGER NOT NULL REFERENCES teams(team_id),
    team_name VARCHAR(100) NOT NULL,
    team_abbrev VARCHAR(10) NOT NULL,
    conference VARCHAR(50) NOT NULL,
    division VARCHAR(50) NOT NULL,
    points INTEGER NOT NULL DEFAULT 0,
    wins INTEGER NOT NULL DEFAULT 0,
    losses INTEGER NOT NULL DEFAULT 0,
    goals_for INTEGER NOT NULL DEFAULT 0,
    goals_against INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(team_id, date)
);

-- rosters table - player information and team assignments
CREATE TABLE rosters (
    id SERIAL PRIMARY KEY,
    team_id INTEGER NOT NULL REFERENCES teams(team_id),
    player_id BIGINT NOT NULL,
    first_name VARCHAR(100) NOT NULL,
    last_name VARCHAR(100) NOT NULL,
    jersey INTEGER,
    height INTEGER, -- in centimeters
    weight INTEGER, -- in kilograms
    dominant_hand VARCHAR(1) CHECK (dominant_hand IN ('L', 'R')),
    position VARCHAR(5) NOT NULL CHECK (position IN ('C', 'LW', 'RW', 'D', 'G')),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(team_id, player_id)
);

-- labels table - game outcomes for training
CREATE TABLE labels (
    game_id BIGINT PRIMARY KEY REFERENCES games(game_id),
    home_score INTEGER NOT NULL,
    away_score INTEGER NOT NULL,
    home_id INTEGER NOT NULL REFERENCES teams(team_id),
    away_id INTEGER NOT NULL REFERENCES teams(team_id),
    home_win BOOLEAN NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);


-- ML FEATURE STORAGE
-- feature snapshots table - stores computed features for each game
CREATE TABLE feature_snapshots (
    id SERIAL PRIMARY KEY,
    game_id BIGINT NOT NULL REFERENCES games(game_id),
    as_of_ts_utc TIMESTAMP WITH TIME ZONE NOT NULL,
    feature_json JSONB NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(game_id, as_of_ts_utc)
);


-- INDEXES FOR PERFORMANCE
-- Games table indexes
CREATE INDEX idx_games_season ON games(season);
CREATE INDEX idx_games_date ON games(game_date);
CREATE INDEX idx_games_home_team ON games(home_team_id);
CREATE INDEX idx_games_away_team ON games(away_team_id);
CREATE INDEX idx_games_status ON games(status);
CREATE INDEX idx_games_type ON games(game_type);

-- Team stats indexes
CREATE INDEX idx_team_stats_date ON team_stats(date);
CREATE INDEX idx_team_stats_team_date ON team_stats(team_id, date);

-- Feature snapshots indexes
CREATE INDEX idx_feature_snapshots_game ON feature_snapshots(game_id);
CREATE INDEX idx_feature_snapshots_timestamp ON feature_snapshots(as_of_ts_utc);
CREATE INDEX idx_feature_snapshots_json ON feature_snapshots USING GIN(feature_json);

-- Labels indexes
CREATE INDEX idx_labels_home_win ON labels(home_win);
CREATE INDEX idx_labels_home_team ON labels(home_id);
CREATE INDEX idx_labels_away_team ON labels(away_id);


-- COMMENTS FOR DOCUMENTATION
COMMENT ON TABLE teams IS 'NHL team information including conference and division';
COMMENT ON TABLE games IS 'Game schedule and metadata for all NHL games';
COMMENT ON TABLE team_stats IS 'Daily team performance statistics for standings';
COMMENT ON TABLE rosters IS 'Player roster information including physical attributes';
COMMENT ON TABLE labels IS 'Game outcomes used as training labels for ML models';
COMMENT ON TABLE feature_snapshots IS 'Computed features for ML model training and prediction';

