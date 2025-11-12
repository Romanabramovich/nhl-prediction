# NHL Win Prediction Model

A production-ready machine learning system that predicts NHL game outcomes using XGBoost with performance tracking.

## 🎯 Overview

This project implements an end-to-end ML pipeline that:
- Ingests daily NHL data (games, standings, team stats) via NHL API
- Engineers comprehensive features from team performance metrics
- Trains XGBoost models to predict game outcomes
- Generates daily predictions for upcoming games
- Tracks and evaluates prediction accuracy over time

**Key Metrics:**
- Features: 120+ engineered features per game
- Historical Data: Multiple seasons of NHL games
- Performance: Tracked via comprehensive evaluation metrics

## 🏗️ Architecture

```mermaid
flowchart TD
    A[NHL API<br/>api-web.nhle.com] --> B[ETL Pipeline<br/>• Schedule Ingestion<br/>• Standings/Stats<br/>• Labels]
    B --> C[PostgreSQL Database<br/>• games<br/>• team_stats<br/>• labels<br/>• feature_snapshots<br/>• predictions]
    C --> D[Feature Engineering<br/>• Rolling form<br/>• Rest/B2B<br/>• Home/Road splits<br/>• Standings position<br/>• Streaks<br/>• Goal differentials]
    D --> E[XGBoost Model<br/>• Hyperparameter tuning<br/>• Early stopping<br/>• Calibration]
    E --> F[Predictions & Evaluation<br/>• Performance tracking<br/>• Model evaluation]
```

## 📁 Project Structure

```
nhl-prediction/
├── ingest/                   # Data ingestion
│   ├── clients/             
│   │   └── nhl.py           # NHL API client with caching
│   └── loaders/             
│       ├── schedule.py      # Game schedule loader
│       ├── standings.py     # Team standings/stats loader
│       ├── labels.py        # Game results loader
│       ├── teams.py         # Team metadata loader
│       └── rosters.py       # Player roster loader
├── features/                 # Feature engineering
│   ├── build_snapshot.py    # Build feature snapshots
│   ├── extract_features.py  # Extract features from snapshots
│   └── build_dataset.py     # Create train/val datasets
├── models/                   # ML models
│   ├── xgboost_model.py     # XGBoost training
│   ├── inference.py         # Production inference
│   ├── evaluate.py          # Model evaluation
│   ├── evaluate_predictions.py  # Prediction tracking
│   ├── hyperparameter_tuning.py # Hyperparameter optimization
│   ├── baseline.py          # Baseline models
│   └── best_params.json     # Tuned hyperparameters
├── database.py               # Database connection
├── schema.sql                # Database schema
├── backfill.py               # Historical data backfill
└── docker-compose.yaml       # PostgreSQL setup
```

## 🚀 Quick Start

### 1. Prerequisites

- Python 3.11+
- PostgreSQL 15+
- Docker (optional, for database)

### 2. Installation

```bash
# Clone repository
git clone <your-repo-url>
cd nhl-prediction

# Install dependencies
pip install -r requirements.txt

# Start PostgreSQL (using Docker)
docker-compose up -d

# Create database schema
psql -h localhost -U nhl_user -d nhl_prediction -f schema.sql
```

### 3. Configuration

Create a `.env` file with your database URL:

```bash
# .env
DB_URL=postgresql://nhl_user:nhl_password@localhost:5432/nhl_prediction
```

### 4. Initial Data Backfill

```bash
# Backfill 2024 season (adjust as needed)
python backfill.py --season 2024
```

This will:
- Fetch all games for the season
- Ingest daily standings/team stats
- Build feature snapshots for each game
- Collect labels for completed games

### 5. Create Modeling Datasets

```bash
python recreate_modeling_tables.py
```

Creates `modeling_train` and `modeling_val` tables for model training.

### 6. Train Model

```bash
# Train with default parameters
python models/xgboost_model.py

# Or tune hyperparameters first (recommended)
python models/hyperparameter_tuning.py
```

### 7. Generate Predictions

```bash
# Generate predictions for upcoming games
python models/inference.py
```

## 📊 Features

The model uses **120+ engineered features** including:

### 1. Rest & Scheduling (5 features)
- Days of rest for each team
- Back-to-back game indicators
- Rest differential

### 2. Rolling Form (20 features)
- Last 10 games: wins, losses, points
- Goals for/against
- Goal differential
- Regulation wins

### 3. Current Standings (12 features)
- League, conference, division rankings
- Home/road specific rankings

### 4. Situational Performance (32 features)
- Home record (wins, losses, goals, etc.)
- Road record
- Goal differentials by location

### 5. Streaks (2 features)
- Current win/loss streak
- Streak type and length

### 6. Season Aggregates (8 features)
- Total points, win percentage
- Goal differential percentage

## 📈 Model Performance

The XGBoost model is evaluated using:

- **Accuracy**: Percentage of correct predictions
- **Log Loss**: Probabilistic prediction quality
- **AUC-ROC**: Discrimination ability
- **Calibration**: Confidence vs actual accuracy
- **Baseline Comparison**: vs. always-home, random, etc.

**Track Performance:**
```bash
# Evaluate last 30 days
python models/evaluate_predictions.py

# Show detailed report
python models/evaluate_predictions.py --days 30 --top-n 20
```

## 🗄️ Database Schema

Key tables:

- `games` - Game schedule and metadata
- `teams` - Team information
- `team_stats` - Daily team statistics/standings
- `labels` - Game outcomes (training labels)
- `feature_snapshots` - Pre-computed features (as JSONB)
- `predictions` - Model predictions (for tracking)

See `schema.sql` for complete schema.

## 🔧 Advanced Usage

### Hyperparameter Tuning

```bash
python models/hyperparameter_tuning.py
```

Saves best parameters to `models/best_params.json`.

## 🧪 Development

### Running Tests

```bash
# Test database connection
python database.py

# Test API client
python ingest/clients/nhl.py

# Test feature extraction
python features/extract_features.py
```

### Local Development

1. Use local PostgreSQL or Docker
2. Set `DB_URL` in `.env`
3. Run `backfill.py` for historical data
4. Develop and test changes locally

## 📝 Data Pipeline Details

### ETL Process

1. **Schedule Loader** (`ingest/loaders/schedule.py`)
   - Fetches games from NHL API
   - Updates game status (scheduled, live, final)
   - Handles preseason, regular, postseason

2. **Standings Loader** (`ingest/loaders/standings.py`)
   - Daily team statistics
   - Home/road/last-10 splits
   - Rankings across league/conference/division

3. **Labels Loader** (`ingest/loaders/labels.py`)
   - Final scores for completed games
   - Binary home_win target variable

4. **Feature Builder** (`features/build_snapshot.py`)
   - Point-in-time features (as of game day)
   - Prevents data leakage
   - Stores as JSONB for flexibility

