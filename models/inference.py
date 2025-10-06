# models/inference.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import xgboost as xgb
import json
from database import get_database_engine
from features.extract_features import extract_features

def load_training_data():
    """Load all training data to retrain the model"""
    engine = get_database_engine()
    
    with engine.connect() as conn:
        train_df = pd.read_sql("SELECT * FROM modeling_train", conn)
        val_df = pd.read_sql("SELECT * FROM modeling_val", conn)
    
    # Combine for full training
    all_df = pd.concat([train_df, val_df])
    return all_df



def load_best_params():
    """Load best hyperparameters from JSON config file if it exists"""
    config_path = os.path.join(os.path.dirname(__file__), 'best_params.json')
    
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        print(f"  Loaded tuned hyperparameters from {config_path}")
        print(f"  CV neg_log_loss: {config['cv_scores']['neg_log_loss']:.6f}")
        return config['best_params']
    else:
        print(f"  No tuned parameters found; using defaults")
        return None


def train_production_model():
    # train model on all available data for production
    print("Training production model on all data...")
    
    # Load all data 
    all_df = load_training_data().sort_values('game_date')
    
    # Extract features from all data
    X_all = extract_features(all_df).fillna(0)
    y_all = all_df['home_win'].astype(int)
    
    print(f"Training on {len(X_all)} games (all available data)")
    
    # Load tuned params if they exist, otherwise use defaults
    best_params = load_best_params()
    params_source = best_params or {
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.05,
        'subsample': 1.0,
        'colsample_bytree': 1.0,
        'min_child_weight': 1,
        'gamma': 0.0,
        'reg_alpha': 0.0,
        'reg_lambda': 1.0,
    }
    
    # Use native XGBoost API (same as xgboost_model.py)
    print(f"Training XGBoost with {'tuned' if best_params else 'default'} parameters...")
    
    num_boost_round = int(params_source.get('n_estimators', 100))
    native_params = {
        'objective': 'binary:logistic',
        'max_depth': int(params_source.get('max_depth', 6)),
        'eta': float(params_source.get('learning_rate', 0.05)),
        'subsample': float(params_source.get('subsample', 1.0)),
        'colsample_bytree': float(params_source.get('colsample_bytree', 1.0)),
        'min_child_weight': float(params_source.get('min_child_weight', 1)),
        'gamma': float(params_source.get('gamma', 0.0)),
        'alpha': float(params_source.get('reg_alpha', 0.0)),
        'lambda': float(params_source.get('reg_lambda', 1.0)),
        'eval_metric': 'logloss',
    }
    
    # Train on all data (no validation/early stopping for production)
    dtrain = xgb.DMatrix(X_all, label=y_all)
    booster = xgb.train(
        params=native_params,
        dtrain=dtrain,
        num_boost_round=num_boost_round,
        verbose_eval=False,
    )
    
    # Store feature columns for consistency
    feature_columns = X_all.columns.tolist()
    
    print("Model training complete!")
    return booster, feature_columns


def predict_upcoming_games():
    """Predict win probabilities for upcoming games"""
    engine = get_database_engine()
    
    # Get upcoming games (no labels yet)
    with engine.connect() as conn:
        upcoming_df = pd.read_sql("""
            SELECT g.game_id, g.season, g.game_date, g.home_team_id, g.away_team_id,
                   g.home_team_name, g.away_team_name, fs.feature_json
            FROM games g
            JOIN feature_snapshots fs ON fs.game_id = g.game_id
            LEFT JOIN labels l ON l.game_id = g.game_id
            WHERE g.game_type = 'REGULAR' 
            AND l.game_id IS NULL
            AND g.game_date > NOW()
            ORDER BY g.game_date
        """, conn)
    
    if len(upcoming_df) == 0:
        print("No upcoming games found")
        return None
    
    print(f"Found {len(upcoming_df)} upcoming games")
    
    # Train model
    model, feature_columns = train_production_model()
    
    # Extract features for upcoming games
    X_upcoming = extract_features(upcoming_df).fillna(0)
    
    # Enforce feature column order (critical for booster consistency)
    missing_cols = set(feature_columns) - set(X_upcoming.columns)
    if missing_cols:
        print(f"Warning: missing columns in upcoming data: {missing_cols}")
        for col in missing_cols:
            X_upcoming[col] = 0
    X_upcoming = X_upcoming[feature_columns]
    
    # Make predictions using booster
    dupcoming = xgb.DMatrix(X_upcoming)
    predictions = model.predict(dupcoming)
    
    # Create results
    results = []
    for i, (_, game) in enumerate(upcoming_df.iterrows()):
        results.append({
            'game_id': game['game_id'],
            'game_date': game['game_date'],
            'home_team': game['home_team_name'],
            'away_team': game['away_team_name'],
            'home_win_probability': predictions[i],
            'away_win_probability': 1 - predictions[i],
            'prediction_confidence': abs(predictions[i] - 0.5) * 2 
        })
    
    results_df = pd.DataFrame(results)
    
    print("\n=== UPCOMING GAME PREDICTIONS ===")
    print(results_df.head(8))
    
    return results_df

def main():
    # main inference function
    print("NHL Win Probability Predictor")
    print("=" * 40)
    
    # Predict upcoming games
    predictions = predict_upcoming_games()
    
    if predictions is not None:
        print(f"\nPredictions generated for {len(predictions)} upcoming games")
        print("Model ready for production use!")

if __name__ == "__main__":
    main()
