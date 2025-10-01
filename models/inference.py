# models/inference.py
import pandas as pd
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from database import get_database_engine

def load_training_data():
    """Load all training data to retrain the model"""
    engine = get_database_engine()
    
    with engine.connect() as conn:
        train_df = pd.read_sql("SELECT * FROM modeling_train", conn)
        val_df = pd.read_sql("SELECT * FROM modeling_val", conn)
    
    # Combine for full training
    all_df = pd.concat([train_df, val_df])
    return all_df

def extract_features(df):
    """Extract features from JSON"""
    features = []
    
    for _, row in df.iterrows():
        feature_dict = row['feature_json']
        
        feature_row = {
            'home_roll_pts': feature_dict.get('home_roll_pts'),
            'away_roll_pts': feature_dict.get('away_roll_pts'),
            'home_roll_w': feature_dict.get('home_roll_w'),
            'away_roll_w': feature_dict.get('away_roll_w'),
            'home_roll_l': feature_dict.get('home_roll_l'),
            'away_roll_l': feature_dict.get('away_roll_l'),
            'home_roll_gf': feature_dict.get('home_roll_gf'),
            'away_roll_gf': feature_dict.get('away_roll_gf'),
            'home_roll_ga': feature_dict.get('home_roll_ga'),
            'away_roll_ga': feature_dict.get('away_roll_ga'),
            'rest_diff': feature_dict.get('rest_diff'),
            'b2b_home': feature_dict.get('b2b_home'),
            'b2b_away': feature_dict.get('b2b_away'),
        }
        features.append(feature_row)
    
    return pd.DataFrame(features)

def train_production_model():
    # train model on all available data for production
    print("Training production model on all data...")
    
    # Load all data
    all_df = load_training_data()
    
    # Extract features
    X_all = extract_features(all_df).fillna(0)
    y_all = all_df['home_win'].astype(int)
    
    print(f"Training on {len(X_all)} games")
    
    # Split for calibration
    X_train, X_cal, y_train, y_cal = train_test_split(
        X_all, y_all, test_size=0.2, random_state=42, stratify=y_all
    )
    
    # Train base model
    base_model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        eval_metric='logloss'
    )
    
    # Use cross-validation for calibration
    calibrated_model = CalibratedClassifierCV(
        base_model, 
        method='sigmoid',
        cv=3  # Use 3-fold CV for calibration
    )
    
    # Train and calibrate on combined data
    X_combined = pd.concat([X_train, X_cal])
    y_combined = pd.concat([y_train, y_cal])
    calibrated_model.fit(X_combined, y_combined)
    
    print("Production model trained successfully!")
    return calibrated_model, X_all.columns

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
    
    # Make predictions
    predictions = model.predict_proba(X_upcoming)[:, 1]
    
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
    print(results_df.round(4))
    
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
