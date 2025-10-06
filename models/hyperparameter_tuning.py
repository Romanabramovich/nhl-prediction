# models/hyperparameter_tuning.py
import pandas as pd
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import xgboost as xgb
import json
from datetime import datetime
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from database import get_database_engine
from features.extract_features import extract_features


def load_data():
    # load training data from database, sorted by date for time series split
    engine = get_database_engine()
    
    with engine.connect() as conn:
        # load training data only
        train_df = pd.read_sql("""
            SELECT * FROM modeling_train 
            ORDER BY game_date
        """, conn)
        
        val_df = pd.read_sql("""
            SELECT * FROM modeling_val
            ORDER BY game_date
        """, conn)
    
    return train_df, val_df


def run_hyperparameter_tuning():
    """
    Run comprehensive hyperparameter tuning using GridSearchCV with TimeSeriesSplit.
    Optimizes for neg_log_loss while tracking multiple metrics.
    Saves best parameters to JSON config file.
    """
    print("="*80)
    print("HYPERPARAMETER TUNING FOR XGBOOST")
    print("="*80)
    
    # Load data
    print("\n[1/6] Loading data...")
    train_df, val_df = load_data()
    
    print(f"  Train set: {len(train_df)} games")
    print(f"  Validation set: {len(val_df)} games")
    
    # Extract features
    print("\n[2/6] Extracting features...")
    X_train = extract_features(train_df)
    y_train = train_df['home_win'].astype(int)
    
    X_val = extract_features(val_df)
    y_val = val_df['home_win'].astype(int)
    
    # Handle nulls
    X_train = X_train.fillna(0)
    X_val = X_val.fillna(0)
    
    print(f"  Feature matrix shape: {X_train.shape}")
    print(f"  Number of features: {X_train.shape[1]}")
    
    # Define comprehensive parameter grid
    print("\n[3/6] Setting up parameter grid...")
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [7, 9],
        'learning_rate': [0.01, 0.05],
        'subsample': [0.7, 0.9, 1.0],
        'colsample_bytree': [0.7, 1.0],
        'min_child_weight': [1, 3],
        'gamma': [0.1, 0.2],
        'reg_alpha': [0.1, 0.5],      
        'reg_lambda': [1, 2]       
    }
    
    total_combinations = 1
    for param, values in param_grid.items():
        total_combinations *= len(values)
        print(f"  {param}: {values}")
    
    print(f"\n  Total parameter combinations: {total_combinations:,}")
    print(f"  With 5-fold CV: {total_combinations * 5:,} model fits")
    
    # Create base model
    base_model = xgb.XGBClassifier(
        random_state=42,
        eval_metric='logloss',
        tree_method='hist'  # Faster training
    )
    
    # Setup TimeSeriesSplit for temporal validation
    # This respects the temporal order: train on past, validate on future
    tscv = TimeSeriesSplit(n_splits=5)
    
    print("\n[4/6] Initializing GridSearchCV with TimeSeriesSplit...")
    print("  Cross-validation strategy: TimeSeriesSplit (5 splits)")
    print("  Primary scoring metric: neg_log_loss")
    print("  Additional metrics tracked: roc_auc, accuracy")
    
    # Setup multi-metric scoring
    scoring = {
        'neg_log_loss': 'neg_log_loss',
        'roc_auc': 'roc_auc',
        'accuracy': 'accuracy'
    }
    
    # Initialize GridSearchCV
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=tscv,                          # TimeSeriesSplit for temporal data
        scoring=scoring,                  # Multi-metric scoring
        refit='neg_log_loss',            # Refit best model based on log loss
        n_jobs=-1,                        # Use all CPU cores
        verbose=2,                        # Show progress
        return_train_score=True,          # Track training scores too
        error_score='raise'               # Raise errors if they occur
    )
    
    # Run grid search
    print("\n[5/6] Running grid search (this will take a while)...\n")
    start_time = datetime.now()
    
    grid_search.fit(X_train, y_train)
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    print(f"\n  Grid search completed in {duration/60:.2f} minutes")
    
    # Display results
    print("\n" + "="*80)
    print("GRID SEARCH RESULTS")
    print("="*80)
    
    print("\n[6/6] Best Parameters Found:")
    print("-" * 80)
    for param, value in sorted(grid_search.best_params_.items()):
        print(f"  {param:20s}: {value}")
    
    print("\n" + "-" * 80)
    print("Best Cross-Validation Scores:")
    print("-" * 80)
    print(f"  Neg Log Loss: {grid_search.best_score_:.6f}")
    
    # Get scores for other metrics
    best_index = grid_search.best_index_
    results = grid_search.cv_results_
    
    print(f"  ROC-AUC:      {results['mean_test_roc_auc'][best_index]:.6f} "
          f"(+/- {results['std_test_roc_auc'][best_index]:.6f})")
    print(f"  Accuracy:     {results['mean_test_accuracy'][best_index]:.6f} "
          f"(+/- {results['std_test_accuracy'][best_index]:.6f})")
    
    # Show top 10 parameter combinations
    print("\n" + "="*80)
    print("TOP 10 PARAMETER COMBINATIONS")
    print("="*80)
    
    results_df = pd.DataFrame(grid_search.cv_results_)
    results_df = results_df.sort_values('rank_test_neg_log_loss')
    
    top_results = results_df[[
        'params', 
        'mean_test_neg_log_loss', 
        'std_test_neg_log_loss',
        'mean_test_roc_auc',
        'mean_test_accuracy',
        'rank_test_neg_log_loss'
    ]].head(10)
    
    for idx, row in top_results.iterrows():
        rank = int(row['rank_test_neg_log_loss'])
        print(f"\n  Rank #{rank}:")
        print(f"    Neg Log Loss: {row['mean_test_neg_log_loss']:.6f} (+/- {row['std_test_neg_log_loss']:.6f})")
        print(f"    ROC-AUC:      {row['mean_test_roc_auc']:.6f}")
        print(f"    Accuracy:     {row['mean_test_accuracy']:.6f}")
        print(f"    Parameters:   {row['params']}")
    
    # Evaluate best model on validation set
    print("\n" + "="*80)
    print("VALIDATION SET PERFORMANCE (HOLDOUT)")
    print("="*80)
    
    best_model = grid_search.best_estimator_
    y_pred_proba = best_model.predict_proba(X_val)[:, 1]
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    val_accuracy = accuracy_score(y_val, y_pred)
    val_logloss = log_loss(y_val, y_pred_proba)
    val_auc = roc_auc_score(y_val, y_pred_proba)
    
    print(f"\n  Validation Accuracy:  {val_accuracy:.6f}")
    print(f"  Validation Log Loss:  {val_logloss:.6f}")
    print(f"  Validation ROC-AUC:   {val_auc:.6f}")
    
    # Compare to baseline
    majority_class_acc = (y_val == 1).mean()
    print(f"\n  Baseline (Majority Class): {majority_class_acc:.6f}")
    print(f"  Improvement over baseline: {val_accuracy - majority_class_acc:+.6f}")
    
    # Save best parameters to JSON
    print("\n" + "="*80)
    print("SAVING BEST PARAMETERS")
    print("="*80)
    
    config = {
        'best_params': grid_search.best_params_,
        'cv_scores': {
            'neg_log_loss': float(grid_search.best_score_),
            'roc_auc': float(results['mean_test_roc_auc'][best_index]),
            'accuracy': float(results['mean_test_accuracy'][best_index])
        },
        'validation_scores': {
            'accuracy': float(val_accuracy),
            'log_loss': float(val_logloss),
            'roc_auc': float(val_auc)
        },
        'tuning_metadata': {
            'timestamp': datetime.now().isoformat(),
            'duration_minutes': float(duration / 60),
            'n_splits': 5,
            'cv_strategy': 'TimeSeriesSplit',
            'scoring_metric': 'neg_log_loss',
            'n_parameter_combinations': total_combinations,
            'training_samples': len(X_train),
            'validation_samples': len(X_val)
        }
    }
    
    # Save to JSON file in models directory
    config_path = os.path.join(os.path.dirname(__file__), 'best_params.json')
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    
    print(f"\n  ✓ Best parameters saved to: {config_path}")
    print(f"  ✓ Load in xgboost_model.py with: json.load(open('models/best_params.json'))")
    
    # Save detailed results to CSV
    results_csv_path = os.path.join(os.path.dirname(__file__), 'tuning_results.csv')
    results_df.to_csv(results_csv_path, index=False)
    print(f"  ✓ Full results saved to: {results_csv_path}")
    
    print("\n" + "="*80)
    print("HYPERPARAMETER TUNING COMPLETE!")
    print("="*80)
    
    return grid_search.best_estimator_, grid_search.best_params_, config


if __name__ == "__main__":
    run_hyperparameter_tuning()

