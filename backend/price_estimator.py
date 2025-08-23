"""
Price estimation using Scikit-learn
Implements various regression models for amulet price prediction
"""
import os
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib

logger = logging.getLogger(__name__)

class PriceEstimator:
    def __init__(self, model_path: str = "models/price_model.pkl"):
        """
        Initialize price estimation system
        
        Args:
            model_path: Path to saved model file
        """
        self.model_path = model_path
        self.model = None
        self.scaler = None
        self.label_encoders = {}
        self.feature_columns = []
        self.model_metrics = {}
        
        # Load existing model if available
        self._load_model()
    
    def _load_model(self):
        """Load trained model and preprocessing objects"""
        if os.path.exists(self.model_path):
            try:
                model_data = joblib.load(self.model_path)
                self.model = model_data.get('model')
                self.scaler = model_data.get('scaler')
                self.label_encoders = model_data.get('label_encoders', {})
                self.feature_columns = model_data.get('feature_columns', [])
                self.model_metrics = model_data.get('metrics', {})
                logger.info(f"✅ Loaded price model from {self.model_path}")
                logger.info(f"📊 Model metrics: {self.model_metrics}")
            except Exception as e:
                logger.warning(f"⚠️ Could not load price model: {e}")
                self.model = None
        else:
            logger.info("🔄 No existing price model found")
    
    def _save_model(self):
        """Save trained model and preprocessing objects"""
        try:
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'label_encoders': self.label_encoders,
                'feature_columns': self.feature_columns,
                'metrics': self.model_metrics
            }
            
            joblib.dump(model_data, self.model_path)
            logger.info(f"💾 Saved price model to {self.model_path}")
        except Exception as e:
            logger.error(f"❌ Could not save price model: {e}")
    
    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features for training or prediction
        
        Args:
            data: Raw data DataFrame
            
        Returns:
            Processed features DataFrame
        """
        df = data.copy()
        
        # Encode categorical features
        categorical_columns = ['class_name', 'temple', 'year_range', 'condition']
        
        for col in categorical_columns:
            if col in df.columns:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    df[f'{col}_encoded'] = self.label_encoders[col].fit_transform(df[col].astype(str))
                else:
                    # Handle unseen categories in prediction
                    try:
                        df[f'{col}_encoded'] = self.label_encoders[col].transform(df[col].astype(str))
                    except ValueError:
                        # Assign unknown category to most frequent class
                        most_frequent = self.label_encoders[col].classes_[0]
                        df[col] = df[col].fillna(most_frequent)
                        df[f'{col}_encoded'] = self.label_encoders[col].transform(df[col].astype(str))
        
        # Create numerical features
        feature_columns = []
        
        # Encoded categorical features
        for col in categorical_columns:
            if f'{col}_encoded' in df.columns:
                feature_columns.append(f'{col}_encoded')
        
        # Numerical features
        numerical_features = ['age_years', 'rarity_score', 'market_demand']
        for col in numerical_features:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                feature_columns.append(col)
        
        self.feature_columns = feature_columns
        return df[feature_columns]
    
    def train_model(self, training_data: pd.DataFrame, target_column: str = 'price'):
        """
        Train price estimation model
        
        Args:
            training_data: DataFrame with features and target
            target_column: Name of price column
        """
        logger.info(f"🏋️ Training price estimation model with {len(training_data)} samples")
        
        # Prepare features and target
        X = self.prepare_features(training_data)
        y = training_data[target_column]
        
        # Handle missing values
        X = X.fillna(X.mean())
        y = y.fillna(y.median())
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train multiple models and select best
        models = {
            'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
            'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'Ridge': Ridge(alpha=1.0),
            'LinearRegression': LinearRegression()
        }
        
        best_model = None
        best_score = -np.inf
        model_results = {}
        
        for name, model in models.items():
            logger.info(f"🔄 Training {name}")
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
            
            # Train on full training set
            model.fit(X_train_scaled, y_train)
            
            # Evaluate
            y_pred = model.predict(X_test_scaled)
            r2 = r2_score(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            
            model_results[name] = {
                'cv_score_mean': cv_scores.mean(),
                'cv_score_std': cv_scores.std(),
                'r2_score': r2,
                'mse': mse,
                'mae': mae,
                'rmse': np.sqrt(mse)
            }
            
            logger.info(f"📊 {name} - R²: {r2:.3f}, RMSE: {np.sqrt(mse):.2f}")
            
            if cv_scores.mean() > best_score:
                best_score = cv_scores.mean()
                best_model = model
        
        # Select best model
        self.model = best_model
        self.model_metrics = model_results
        
        # Hyperparameter tuning for best model (if RandomForest)
        if isinstance(best_model, RandomForestRegressor):
            logger.info("🔧 Fine-tuning RandomForest hyperparameters")
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10]
            }
            
            grid_search = GridSearchCV(
                RandomForestRegressor(random_state=42),
                param_grid,
                cv=3,
                scoring='r2',
                n_jobs=-1
            )
            
            grid_search.fit(X_train_scaled, y_train)
            self.model = grid_search.best_estimator_
            
            logger.info(f"🎯 Best parameters: {grid_search.best_params_}")
        
        # Save model
        self._save_model()
        logger.info("✅ Model training completed")
    
    def predict_price(self, features: Dict) -> Dict:
        """
        Predict price for given features
        
        Args:
            features: Dictionary with feature values
            
        Returns:
            Dictionary with price prediction and confidence intervals
        """
        if self.model is None:
            logger.warning("⚠️ No trained model available, using mock predictions")
            return self._mock_price_prediction(features)
        
        try:
            # Create DataFrame from features
            df = pd.DataFrame([features])
            
            # Prepare features
            X = self.prepare_features(df)
            X = X.fillna(X.mean()) if len(X) > 0 else X.fillna(0)
            
            # Scale features
            X_scaled = self.scaler.transform(X)
            
            # Predict
            prediction = self.model.predict(X_scaled)[0]
            
            # Calculate confidence intervals (approximate)
            if hasattr(self.model, 'estimators_'):
                # For ensemble methods, use prediction variance
                predictions = [estimator.predict(X_scaled)[0] for estimator in self.model.estimators_]
                std = np.std(predictions)
                
                p05 = max(0, prediction - 1.96 * std)
                p95 = prediction + 1.96 * std
                p50 = prediction
            else:
                # Simple approach for other models
                error_margin = prediction * 0.2  # 20% error margin
                p05 = max(0, prediction - error_margin)
                p95 = prediction + error_margin
                p50 = prediction
            
            return {
                "p05": float(p05),
                "p50": float(p50),
                "p95": float(p95),
                "confidence": "high" if std < prediction * 0.1 else "medium"
            }
            
        except Exception as e:
            logger.error(f"❌ Prediction error: {e}")
            return self._mock_price_prediction(features)
    
    def _mock_price_prediction(self, features: Dict) -> Dict:
        """Generate mock price prediction based on class"""
        class_name = features.get('class_name', 'unknown')
        
        # Mock price ranges based on class
        price_ranges = {
            'หลวงพ่อกวยแหวกม่าน': (2000, 8000, 20000),
            'โพธิ์ฐานบัว': (1500, 6000, 15000),
            'ฐานสิงห์': (1000, 4000, 12000),
            'สีวลี': (800, 3000, 10000),
            'default': (1000, 5000, 15000)
        }
        
        p05, p50, p95 = price_ranges.get(class_name, price_ranges['default'])
        
        return {
            "p05": float(p05),
            "p50": float(p50),
            "p95": float(p95),
            "confidence": "mock"
        }
    
    def get_feature_importance(self) -> Optional[Dict]:
        """Get feature importance from trained model"""
        if self.model is None or not hasattr(self.model, 'feature_importances_'):
            return None
        
        importance_dict = {}
        for i, importance in enumerate(self.model.feature_importances_):
            if i < len(self.feature_columns):
                importance_dict[self.feature_columns[i]] = float(importance)
        
        return importance_dict
    
    def evaluate_model(self, test_data: pd.DataFrame, target_column: str = 'price') -> Dict:
        """
        Evaluate model performance on test data
        
        Args:
            test_data: Test DataFrame
            target_column: Target column name
            
        Returns:
            Dictionary with evaluation metrics
        """
        if self.model is None:
            return {"error": "No trained model available"}
        
        try:
            X_test = self.prepare_features(test_data)
            y_test = test_data[target_column]
            
            X_test_scaled = self.scaler.transform(X_test)
            y_pred = self.model.predict(X_test_scaled)
            
            metrics = {
                'r2_score': r2_score(y_test, y_pred),
                'mse': mean_squared_error(y_test, y_pred),
                'mae': mean_absolute_error(y_test, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred))
            }
            
            return metrics
            
        except Exception as e:
            return {"error": f"Evaluation failed: {str(e)}"}

def create_mock_training_data() -> pd.DataFrame:
    """Create mock training data for development"""
    np.random.seed(42)
    
    classes = ['หลวงพ่อกวยแหวกม่าน', 'โพธิ์ฐานบัว', 'ฐานสิงห์', 'สีวลี']
    temples = ['วัดแสงอรุณ', 'วัดใหญ่', 'วัดเก่า', 'วัดใหม่']
    conditions = ['ใหม่', 'ใช้แล้ว', 'เก่า']
    year_ranges = ['2500-2520', '2521-2540', '2541-2560', '2561-2580']
    
    n_samples = 500
    data = []
    
    for i in range(n_samples):
        class_name = np.random.choice(classes)
        temple = np.random.choice(temples)
        condition = np.random.choice(conditions)
        year_range = np.random.choice(year_ranges)
        
        # Generate correlated features
        age_years = np.random.randint(5, 50)
        rarity_score = np.random.uniform(0.1, 1.0)
        market_demand = np.random.uniform(0.2, 1.0)
        
        # Generate price based on features
        base_price = {
            'หลวงพ่อกวยแหวกม่าน': 8000,
            'โพธิ์ฐานบัว': 6000,
            'ฐานสิงห์': 4000,
            'สีวลี': 3000
        }[class_name]
        
        price = base_price * (1 + rarity_score) * (0.5 + market_demand) * (1 + age_years/100)
        price *= np.random.uniform(0.8, 1.2)  # Add noise
        
        data.append({
            'class_name': class_name,
            'temple': temple,
            'condition': condition,
            'year_range': year_range,
            'age_years': age_years,
            'rarity_score': rarity_score,
            'market_demand': market_demand,
            'price': price
        })
    
    return pd.DataFrame(data)

# Integration function for API
def get_enhanced_price_estimation(class_id: int, class_name: str) -> Dict:
    """
    Get enhanced price estimation using ML model
    
    Args:
        class_id: Predicted class ID
        class_name: Predicted class name
        
    Returns:
        Price estimation dictionary
    """
    estimator = PriceEstimator()
    
    # Create features for prediction
    features = {
        'class_name': class_name,
        'temple': 'วัดแสงอรุณ',  # Default values
        'condition': 'ใช้แล้ว',
        'year_range': '2541-2560',
        'age_years': 20,
        'rarity_score': 0.7,
        'market_demand': 0.8
    }
    
    return estimator.predict_price(features)
