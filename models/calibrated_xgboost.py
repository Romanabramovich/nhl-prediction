# models/calibrated_xgboost.py
import pandas as pd
import numpy as np
import json
import xgboost as xgb
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score, brier_score_loss
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from database import get_database_engine
from sqlalchemy import text

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
        # feature_json is already a dict, not a string
        feature_dict = row['feature_json']
        
        # extract key features, handling nulls
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

def train_calibrated_xgboost():
    # train calibrated XGBoost model
    print("loading data...")
    train_df, val_df = load_data()
    
    print(f"Train set: {len(train_df)} games")
    print(f"Validation set: {len(val_df)} games")
    
    # extract features
    print("extracting features...")
    X_train = extract_features(train_df)
    y_train = train_df['home_win'].astype(int)
    
    X_val = extract_features(val_df)
    y_val = val_df['home_win'].astype(int)
    
    # handle nulls - simple imputation with 0
    print("handling null values...")
    X_train = X_train.fillna(0)
    X_val = X_val.fillna(0)
    
    print(f"Feature matrix shape: {X_train.shape}")
    print(f"Features: {list(X_train.columns)}")
    
    # split training data for calibration
    X_train_base, X_cal, y_train_base, y_cal = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    print(f"Base training: {len(X_train_base)} games")
    print(f"Calibration: {len(X_cal)} games")
    
    # train base XGBoost
    print("training base XGBoost...")
    base_model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        eval_metric='logloss'
    )
    
    # train and calibrate in one step using cross-validation
    print("training and calibrating XGBoost...")
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
        method='sigmoid',  # Platt scaling
        cv=3  # Use 3-fold CV for calibration
    )
    
    # Train and calibrate on combined data
    X_combined = pd.concat([X_train_base, X_cal])
    y_combined = pd.concat([y_train_base, y_cal])
    calibrated_model.fit(X_combined, y_combined)
    
    # make predictions with calibrated model
    y_pred_proba_cal = calibrated_model.predict_proba(X_val)[:, 1]
    
    # For uncalibrated comparison, train a separate base model
    base_model.fit(X_train_base, y_train_base)
    y_pred_proba_uncal = base_model.predict_proba(X_val)[:, 1]
    
    y_pred_uncal = (y_pred_proba_uncal > 0.5).astype(int)
    y_pred_cal = (y_pred_proba_cal > 0.5).astype(int)
    
    # calculate metrics for both
    acc_uncal = accuracy_score(y_val, y_pred_uncal)
    logloss_uncal = log_loss(y_val, y_pred_proba_uncal)
    auc_uncal = roc_auc_score(y_val, y_pred_proba_uncal)
    brier_uncal = brier_score_loss(y_val, y_pred_proba_uncal)
    
    acc_cal = accuracy_score(y_val, y_pred_cal)
    logloss_cal = log_loss(y_val, y_pred_proba_cal)
    auc_cal = roc_auc_score(y_val, y_pred_proba_cal)
    brier_cal = brier_score_loss(y_val, y_pred_proba_cal)
    
    print("\n=== CALIBRATED XGBOOST RESULTS ===")
    print(f"{'Metric':<20} {'Uncalibrated':<12} {'Calibrated':<12} {'Improvement':<12}")
    print("-" * 60)
    print(f"{'Accuracy':<20} {acc_uncal:<12.4f} {acc_cal:<12.4f} {acc_cal-acc_uncal:<+12.4f}")
    print(f"{'Log Loss':<20} {logloss_uncal:<12.4f} {logloss_cal:<12.4f} {logloss_cal-logloss_uncal:<+12.4f}")
    print(f"{'AUC':<20} {auc_uncal:<12.4f} {auc_cal:<12.4f} {auc_cal-auc_uncal:<+12.4f}")
    print(f"{'Brier Score':<20} {brier_uncal:<12.4f} {brier_cal:<12.4f} {brier_cal-brier_uncal:<+12.4f}")
    
    # compare to baselines
    majority_class_acc = (y_val == 1).mean()
    random_acc = 0.5
    
    print(f"\n=== BASELINE COMPARISON ===")
    print(f"Majority Class (Home Win): {majority_class_acc:.4f}")
    print(f"Random (50%): {random_acc:.4f}")
    print(f"Calibrated XGBoost: {acc_cal:.4f}")
    
    # feature importance
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': base_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\n=== TOP FEATURES ===")
    print(feature_importance.head(10))
    
    # probability distribution analysis
    print(f"\n=== PROBABILITY DISTRIBUTION ===")
    print(f"Uncalibrated - Mean: {y_pred_proba_uncal.mean():.3f}, Std: {y_pred_proba_uncal.std():.3f}")
    print(f"Calibrated   - Mean: {y_pred_proba_cal.mean():.3f}, Std: {y_pred_proba_cal.std():.3f}")
    print(f"Actual       - Mean: {y_val.mean():.3f}")
    
    return calibrated_model, base_model, acc_cal, logloss_cal, auc_cal

if __name__ == "__main__":
    train_calibrated_xgboost()
