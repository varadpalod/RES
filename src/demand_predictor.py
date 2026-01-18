"""
Demand Prediction Model
XGBoost-based model for real estate demand prediction
"""
import pandas as pd
import numpy as np
from typing import Tuple, Dict, List, Optional
from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import MODEL_FILE, MODEL_PARAMS, OUTPUT_DIR

try:
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    from sklearn.ensemble import RandomForestRegressor
    import joblib
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: scikit-learn not installed")

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: XGBoost not installed. Using RandomForest.")


class DemandPredictor:
    """
    Machine learning model for property demand prediction
    Uses XGBoost with sentiment-enhanced features
    """
    
    def __init__(self, model_type: str = 'auto'):
        """
        Initialize the predictor
        
        Args:
            model_type: 'xgboost', 'random_forest', or 'auto'
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required. Install with: pip install scikit-learn")
        
        self.model_type = self._select_model_type(model_type)
        self.model = None
        self.feature_names = []
        self.metrics = {}
        
    def _select_model_type(self, model_type: str) -> str:
        """Select the best available model"""
        if model_type == 'auto':
            return 'xgboost' if XGBOOST_AVAILABLE else 'random_forest'
        return model_type
    
    def _create_model(self):
        """Create the ML model based on type"""
        params = MODEL_PARAMS
        
        if self.model_type == 'xgboost' and XGBOOST_AVAILABLE:
            return XGBRegressor(
                n_estimators=params['n_estimators'],
                max_depth=params['max_depth'],
                learning_rate=params['learning_rate'],
                random_state=params['random_state'],
                verbosity=0
            )
        else:
            return RandomForestRegressor(
                n_estimators=params['n_estimators'],
                max_depth=params['max_depth'],
                random_state=params['random_state'],
                n_jobs=-1
            )
    
    def train(self, X: pd.DataFrame, y: pd.Series, 
              test_size: float = 0.2) -> Dict:
        """
        Train the demand prediction model
        
        Args:
            X: Feature DataFrame
            y: Target series (demand_score)
            test_size: Proportion for test set
            
        Returns:
            Dictionary with training metrics
        """
        self.feature_names = list(X.columns)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=MODEL_PARAMS['random_state']
        )
        
        # Create and train model
        self.model = self._create_model()
        self.model.fit(X_train, y_train)
        
        # Predictions
        train_pred = self.model.predict(X_train)
        test_pred = self.model.predict(X_test)
        
        # Calculate metrics
        self.metrics = {
            'model_type': self.model_type,
            'n_features': len(self.feature_names),
            'n_samples': len(X),
            'train': {
                'rmse': np.sqrt(mean_squared_error(y_train, train_pred)),
                'mae': mean_absolute_error(y_train, train_pred),
                'r2': r2_score(y_train, train_pred)
            },
            'test': {
                'rmse': np.sqrt(mean_squared_error(y_test, test_pred)),
                'mae': mean_absolute_error(y_test, test_pred),
                'r2': r2_score(y_test, test_pred)
            }
        }
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, X, y, cv=5, scoring='r2')
        self.metrics['cv_r2_mean'] = cv_scores.mean()
        self.metrics['cv_r2_std'] = cv_scores.std()
        
        print(f"✓ Model trained ({self.model_type})")
        print(f"  Train R²: {self.metrics['train']['r2']:.3f}")
        print(f"  Test R²:  {self.metrics['test']['r2']:.3f}")
        print(f"  CV R²:    {self.metrics['cv_r2_mean']:.3f} ± {self.metrics['cv_r2_std']:.3f}")
        
        return self.metrics
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make demand predictions
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Array of demand scores
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Ensure columns match training
        missing_cols = set(self.feature_names) - set(X.columns)
        if missing_cols:
            for col in missing_cols:
                X[col] = 0
        
        X = X[self.feature_names]
        
        predictions = self.model.predict(X)
        return np.clip(predictions, 0, 100)
    
    def get_feature_importance(self, top_n: int = 15) -> pd.DataFrame:
        """
        Get feature importance rankings
        
        Args:
            top_n: Number of top features to return
            
        Returns:
            DataFrame with feature importance
        """
        if self.model is None:
            raise ValueError("Model not trained.")
        
        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
        else:
            return pd.DataFrame()
        
        df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        df['importance_pct'] = (df['importance'] / df['importance'].sum() * 100).round(2)
        
        return df.head(top_n).reset_index(drop=True)
    
    def save_model(self, file_path: Optional[Path] = None) -> None:
        """Save trained model to disk"""
        path = file_path or MODEL_FILE
        
        model_data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'metrics': self.metrics,
            'model_type': self.model_type
        }
        
        joblib.dump(model_data, path)
        print(f"✓ Model saved to {path}")
    
    def load_model(self, file_path: Optional[Path] = None) -> None:
        """Load trained model from disk"""
        path = file_path or MODEL_FILE
        
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        
        model_data = joblib.load(path)
        
        self.model = model_data['model']
        self.feature_names = model_data['feature_names']
        self.metrics = model_data['metrics']
        self.model_type = model_data['model_type']
        
        print(f"✓ Model loaded from {path}")
    
    def save_report(self, file_path: Optional[Path] = None) -> None:
        """Save model report to JSON"""
        path = file_path or (OUTPUT_DIR / 'model_report.json')
        
        report = {
            'model_type': self.model_type,
            'performance': self.metrics,
            'top_features': self.get_feature_importance(15).to_dict('records')
        }
        
        with open(path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"✓ Model report saved to {path}")


def train_demand_model(X: pd.DataFrame, y: pd.Series, save: bool = True) -> DemandPredictor:
    """
    Convenience function to train and save demand model
    
    Args:
        X: Features
        y: Target
        save: Whether to save model to disk
        
    Returns:
        Trained DemandPredictor
    """
    predictor = DemandPredictor()
    predictor.train(X, y)
    
    if save:
        predictor.save_model()
        predictor.save_report()
    
    return predictor


if __name__ == "__main__":
    # Test with synthetic data
    np.random.seed(42)
    n_samples = 1000
    
    X = pd.DataFrame({
        'square_feet': np.random.normal(1200, 300, n_samples),
        'num_bedrooms': np.random.choice([1, 2, 3, 4], n_samples),
        'property_age': np.random.randint(0, 25, n_samples),
        'has_garage': np.random.choice([0, 1], n_samples),
        'price_perception': np.random.uniform(-1, 1, n_samples),
        'infrastructure_satisfaction': np.random.uniform(-1, 1, n_samples),
        'investment_confidence': np.random.uniform(-1, 1, n_samples),
        'buying_urgency': np.random.uniform(0, 1, n_samples),
        'overall_sentiment': np.random.uniform(-1, 1, n_samples)
    })
    
    # Synthetic target based on features
    y = (
        50 + 
        X['investment_confidence'] * 15 +
        X['buying_urgency'] * 10 +
        X['overall_sentiment'] * 10 -
        X['property_age'] * 0.5 +
        X['has_garage'] * 5 +
        np.random.normal(0, 5, n_samples)
    ).clip(0, 100)
    
    predictor = DemandPredictor()
    metrics = predictor.train(X, y)
    
    print("\nFeature Importance:")
    print(predictor.get_feature_importance(10))
    
    # Test prediction
    sample = X.iloc[:5]
    predictions = predictor.predict(sample)
    print("\nSample Predictions:")
    print(predictions)
