# models/xgboost_model.py
import pandas as pd
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import xgboost as xgb
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler
from database import get_database_engine
from features.extract_features import extract_features

def load_data():
    # load train and validation data from database
    engine = get_database_engine()
    
    with engine.connect() as conn:
        train_df = pd.read_sql("SELECT * FROM modeling_train", conn)
        val_df = pd.read_sql("SELECT * FROM modeling_val", conn)
    
    return train_df, val_df

# extract_features function now imported from features.extract_features

def train_xgboost():
    # train XGBoost model
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
    
    # train XGBoost
    print("training XGBoost...")
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        eval_metric='logloss'
    )
    
    model.fit(X_train, y_train)
    
    # make predictions
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    # calculate metrics
    accuracy = accuracy_score(y_val, y_pred)
    logloss = log_loss(y_val, y_pred_proba)
    auc = roc_auc_score(y_val, y_pred_proba)
    
    print("\n=== XGBOOST RESULTS ===")
    print(f"Validation Accuracy: {accuracy:.4f}")
    print(f"Validation Log Loss: {logloss:.4f}")
    print(f"Validation AUC: {auc:.4f}")
    
    # compare to baselines
    majority_class_acc = (y_val == 1).mean()  # Home win percentage in val set
    random_acc = 0.5
    
    print(f"\n=== BASELINE COMPARISON ===")
    print(f"Majority Class (Home Win): {majority_class_acc:.4f}")
    print(f"Random (50%): {random_acc:.4f}")
    print(f"XGBoost Model: {accuracy:.4f}")
    
    # feature importance
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\n=== TOP FEATURES (XGBoost) ===")
    print(feature_importance.head(10))
    
    return model, accuracy, logloss, auc

if __name__ == "__main__":
    train_xgboost()
