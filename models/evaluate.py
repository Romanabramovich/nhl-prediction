# models/evaluate.py
from models.calibrated_xgboost import train_calibrated_xgboost

def main():
    print("Training and evaluating calibrated XGBoost model...")
    model, base_model, acc, logloss, auc = train_calibrated_xgboost()
    
    print(f"\n=== FINAL MODEL PERFORMANCE ===")
    print(f"Accuracy: {acc:.4f}")
    print(f"Log Loss: {logloss:.4f}")
    print(f"AUC: {auc:.4f}")
    
    print(f"\nModel ready for production use!")

if __name__ == "__main__":
    main()
