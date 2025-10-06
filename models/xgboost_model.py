# models/xgboost_model.py
import pandas as pd
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import xgboost as xgb
import json
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from database import get_database_engine
from features.extract_features import extract_features

def load_data():
    # load train and validation data from database
    engine = get_database_engine()
    
    with engine.connect() as conn:
        train_df = pd.read_sql("SELECT * FROM modeling_train", conn)
        val_df = pd.read_sql("SELECT * FROM modeling_val", conn)
    
    return train_df, val_df

def load_best_params():
    # load best hyperparameters from JSON config file if it exists
    config_path = os.path.join(os.path.dirname(__file__), 'best_params.json')
    
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        print(f"  Loaded tuned hyperparameters from {config_path}")
        print(f"  Tuning timestamp: {config['tuning_metadata']['timestamp']}")
        print(f"  CV neg_log_loss: {config['cv_scores']['neg_log_loss']:.6f}")
        return config['best_params']
    else:
        print(f"No tuned parameters found at {config_path}")
        print("Using default parameters. Run hyperparameter_tuning.py to optimize.")
        return None

def train_xgboost(use_tuned_params=True):
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
    
    # handle nulls
    print("handling null values...")
    X_train = X_train.fillna(0)
    X_val = X_val.fillna(0)
    
    print(f"Feature matrix shape: {X_train.shape}")
    print(f"Features: {list(X_train.columns)}")
    
    # load hyperparameters
    print("\n" + "="*60)
    if use_tuned_params:
        best_params = load_best_params()
    else:
        best_params = None
        print("Using default hyperparameters (use_tuned_params=False)")
    
    # train XGBoost (native API to support early stopping across versions)
    print("="*60)
    print("training XGBoost...")

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
        'eval_metric': ['logloss', 'auc'],
    }

    if best_params:
        print("Using tuned hyperparameters")
    else:
        print("Using default hyperparameters")

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    evals = [(dtrain, 'train'), (dval, 'val')]
    booster = xgb.train(
        params=native_params,
        dtrain=dtrain,
        num_boost_round=num_boost_round,
        evals=evals,
        early_stopping_rounds=100,
        verbose_eval=False,
    )
    
    # make predictions using best ntree limit
    # prefer iteration_range if available; fallback to full model
    if hasattr(booster, 'best_iteration') and booster.best_iteration is not None:
        try:
            y_pred_proba = booster.predict(dval, iteration_range=(0, booster.best_iteration + 1))
        except TypeError:
            # older versions may require ntree_limit; if unsupported, use full model
            try:
                y_pred_proba = booster.predict(dval, ntree_limit=booster.best_iteration + 1)
            except TypeError:
                y_pred_proba = booster.predict(dval)
    else:
        y_pred_proba = booster.predict(dval)
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    # calculate metrics
    accuracy = accuracy_score(y_val, y_pred)
    logloss = log_loss(y_val, y_pred_proba)
    auc = roc_auc_score(y_val, y_pred_proba)
    
    print("\n=== XGBOOST RESULTS ===")
    if hasattr(booster, 'best_iteration'):
        print(f"Best iteration (early stopping): {booster.best_iteration}")
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
    
    # feature importance from booster (align by real feature names when available)
    score = booster.get_score(importance_type='gain') or booster.get_score(importance_type='weight')

    # Detect key scheme: either actual column names or f{idx}
    keys = list(score.keys())
    use_f_indices = any(k.startswith('f') and k[1:].isdigit() for k in keys)

    importances = []
    for idx, col in enumerate(X_train.columns):
        key = f'f{idx}' if use_f_indices else col
        importances.append(float(score.get(key, 0.0)))
    feature_importance = pd.DataFrame({'feature': X_train.columns, 'importance': importances})
    # scale to sum to 1 for readability if any non-zero
    total_imp = feature_importance['importance'].sum()
    if total_imp > 0:
        feature_importance['importance'] = feature_importance['importance'] / total_imp
    feature_importance = feature_importance.sort_values('importance', ascending=False)
    
    print(f"\n=== TOP FEATURES (XGBoost) ===")
    print(feature_importance.head(20))
    
    print(f"\n=== BOTTOM FEATURES (XGBoost) ===")
    print(feature_importance.tail(10))
    
    return booster, accuracy, logloss, auc

if __name__ == "__main__":
    train_xgboost()
