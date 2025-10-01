# models/backtest.py
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from database import get_database_engine
from sqlalchemy import text
import json

def load_data():
    # load train and validation data from database
    engine = get_database_engine()
    
    with engine.connect() as conn:
        train_df = pd.read_sql("SELECT * FROM modeling_train", conn)
        val_df = pd.read_sql("SELECT * FROM modeling_val", conn)
    
    return train_df, val_df

def extract_features(df):
    # extract features from JSON and create feature matrix
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

def train_model(X_train, y_train, X_cal, y_cal):
    # train calibrated XGBoost model
    base_model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        eval_metric='logloss'
    )
    
    # Use cross-validation for calibration (no cv='prefit')
    calibrated_model = CalibratedClassifierCV(
        base_model, 
        method='sigmoid',
        cv=3  # Use 3-fold CV for calibration
    )
    
    # Train and calibrate on combined data
    X_combined = pd.concat([X_train, X_cal])
    y_combined = pd.concat([y_train, y_cal])
    calibrated_model.fit(X_combined, y_combined)
    
    return calibrated_model

def evaluate_model(model, X, y):
    # evaluate model and return metrics
    y_pred_proba = model.predict_proba(X)[:, 1]
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    return {
        'accuracy': accuracy_score(y, y_pred),
        'log_loss': log_loss(y, y_pred_proba),
        'auc': roc_auc_score(y, y_pred_proba),
        'predictions': y_pred_proba
    }

def backtest_by_season():
    # backtest performance by season
    print("=== BACKTEST BY SEASON ===")
    
    train_df, val_df = load_data()
    
    # Extract features
    X_train = extract_features(train_df).fillna(0)
    y_train = train_df['home_win'].astype(int)
    
    X_val = extract_features(val_df).fillna(0)
    y_val = val_df['home_win'].astype(int)
    
    # Split for calibration
    X_train_base, X_cal, y_train_base, y_cal = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    # Train model
    model = train_model(X_train_base, y_train_base, X_cal, y_cal)
    
    # Evaluate by season
    results = []
    for season in val_df['season'].unique():
        season_mask = val_df['season'] == season
        X_season = X_val[season_mask]
        y_season = y_val[season_mask]
        
        if len(y_season) > 0:
            metrics = evaluate_model(model, X_season, y_season)
            results.append({
                'season': season,
                'games': len(y_season),
                'accuracy': metrics['accuracy'],
                'log_loss': metrics['log_loss'],
                'auc': metrics['auc']
            })
    
    results_df = pd.DataFrame(results)
    print(results_df.round(4))
    
    return results_df

def backtest_by_month():
    # backtest performance by month
    print("\n=== BACKTEST BY MONTH ===")
    
    train_df, val_df = load_data()
    
    # Extract features
    X_train = extract_features(train_df).fillna(0)
    y_train = train_df['home_win'].astype(int)
    
    X_val = extract_features(val_df).fillna(0)
    y_val = val_df['home_win'].astype(int)
    
    # Add month column
    val_df['month'] = pd.to_datetime(val_df['game_date']).dt.month
    
    # Split for calibration
    X_train_base, X_cal, y_train_base, y_cal = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    # Train model
    model = train_model(X_train_base, y_train_base, X_cal, y_cal)
    
    # Evaluate by month
    results = []
    for month in sorted(val_df['month'].unique()):
        month_mask = val_df['month'] == month
        X_month = X_val[month_mask]
        y_month = y_val[month_mask]
        
        if len(y_month) > 50:  # Only include months with enough games
            metrics = evaluate_model(model, X_month, y_month)
            results.append({
                'month': month,
                'games': len(y_month),
                'accuracy': metrics['accuracy'],
                'log_loss': metrics['log_loss'],
                'auc': metrics['auc']
            })
    
    results_df = pd.DataFrame(results)
    print(results_df.round(4))
    
    return results_df

def backtest_rolling():
    # rolling backtest - train on past, test on future
    print("\n=== ROLLING BACKTEST ===")
    
    train_df, val_df = load_data()
    
    # Combine and sort by date
    all_df = pd.concat([train_df, val_df]).sort_values('game_date')
    
    # Extract features
    X_all = extract_features(all_df).fillna(0)
    y_all = all_df['home_win'].astype(int)
    
    # Define test periods (last 20% of data)
    test_size = int(len(all_df) * 0.2)
    train_size = len(all_df) - test_size
    
    results = []
    
    # Rolling window backtest
    for i in range(5):  # 5 different train/test splits
        start_idx = i * (test_size // 5)
        end_idx = start_idx + train_size
        
        if end_idx + test_size <= len(all_df):
            X_train_roll = X_all.iloc[start_idx:end_idx]
            y_train_roll = y_all.iloc[start_idx:end_idx]
            X_test_roll = X_all.iloc[end_idx:end_idx + test_size]
            y_test_roll = y_all.iloc[end_idx:end_idx + test_size]
            
            # Split train for calibration
            X_train_base, X_cal, y_train_base, y_cal = train_test_split(
                X_train_roll, y_train_roll, test_size=0.2, random_state=42, stratify=y_train_roll
            )
            
            # Train and evaluate
            model = train_model(X_train_base, y_train_base, X_cal, y_cal)
            metrics = evaluate_model(model, X_test_roll, y_test_roll)
            
            results.append({
                'split': i + 1,
                'train_games': len(y_train_roll),
                'test_games': len(y_test_roll),
                'accuracy': metrics['accuracy'],
                'log_loss': metrics['log_loss'],
                'auc': metrics['auc']
            })
    
    results_df = pd.DataFrame(results)
    print(results_df.round(4))
    
    return results_df

def main():
    # run all backtests
    print("Running comprehensive backtesting...")
    
    # Run all backtests
    season_results = backtest_by_season()
    month_results = backtest_by_month()
    rolling_results = backtest_rolling()
    
    # Summary
    print("\n=== BACKTEST SUMMARY ===")
    print(f"Season Performance - Mean Accuracy: {season_results['accuracy'].mean():.4f}")
    print(f"Month Performance - Mean Accuracy: {month_results['accuracy'].mean():.4f}")
    print(f"Rolling Performance - Mean Accuracy: {rolling_results['accuracy'].mean():.4f}")
    
    print("\nBacktesting complete!")

if __name__ == "__main__":
    main()
