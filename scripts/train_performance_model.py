import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import logging
import os
from typing import Tuple, Dict, List

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PerformancePredictor:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.encoder = OneHotEncoder(handle_unknown='ignore')
        
        # Define feature columns by category
        self.categorical_features = ['role', 'team']
        self.numerical_features = [
            # Basic metrics
            'batting_avg', 'strike_rate', 'boundary_rate',
            'economy_rate', 'bowling_avg', 'bowling_sr',
            
            # Advanced batting metrics
            'dot_ball_rate', 'death_over_strike_rate',
            
            # Advanced bowling metrics
            'dot_ball_percentage', 'death_over_economy',
            
            # Wicket-keeping metrics
            'dismissals_per_match', 'catches_per_match', 'stumpings_per_match',
            
            # Form metrics
            'recent_runs', 'recent_wickets',
            
            # Composite scores
            'batting_impact', 'bowling_impact', 'all_rounder_score',
            
            # Match context
            'total_innings'
        ]
        
        # Create preprocessing pipeline
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), self.numerical_features),
                ('cat', OneHotEncoder(), self.categorical_features)
            ]
        )
    
    def prepare_data(self, historical_metrics: pd.DataFrame, squad_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare training data from historical metrics and squad data."""
        # Using the historical metrics directly since it has all the data we need
        df = historical_metrics.copy()
        
        logger.info(f"Historical metrics columns: {df.columns.tolist()}")
        
        # Fill missing values with 0 for numerical features
        for col in self.numerical_features:
            if col in df.columns:
                df[col] = df[col].fillna(0)
        
        # Fill missing values with default values for categorical features
        for col in self.categorical_features:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'UNKNOWN')
        
        # Ensure all required columns exist
        for col in self.numerical_features + self.categorical_features:
            if col not in df.columns:
                df[col] = 0 if col in self.numerical_features else 'UNKNOWN'
        
        # Prepare features and target
        X = df[self.numerical_features + self.categorical_features]
        y = df['credits']
        
        return X, y
    
    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Train the performance prediction model."""
        logger.info("Training performance prediction model...")
        
        # Split data into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Create and train the pipeline
        pipeline = Pipeline([
            ('preprocessor', self.preprocessor),
            ('model', self.model)
        ])
        
        pipeline.fit(X_train, y_train)
        self.model = pipeline
        
        # Calculate and log model performance
        train_score = pipeline.score(X_train, y_train)
        val_score = pipeline.score(X_val, y_val)
        
        logger.info(f"Training R² score: {train_score:.4f}")
        logger.info(f"Validation R² score: {val_score:.4f}")
        
        # Log feature importance
        try:
            feature_names = []
            
            # Get numeric feature names (these remain unchanged)
            for col in self.numerical_features:
                if col in X_train.columns:
                    feature_names.append(col)
            
            # Get one-hot encoded feature names for categorical features
            if hasattr(pipeline.named_steps['preprocessor'], 'transformers_'):
                for name, transformer, cols in pipeline.named_steps['preprocessor'].transformers_:
                    if name == 'cat' and hasattr(transformer, 'get_feature_names_out'):
                        cat_features = transformer.get_feature_names_out(input_features=self.categorical_features)
                        feature_names.extend(cat_features)
            
            # Get feature importances
            if hasattr(pipeline.named_steps['model'], 'feature_importances_'):
                importances = pipeline.named_steps['model'].feature_importances_
                
                # If lengths match, create a DataFrame of feature importances
                if len(importances) == len(feature_names):
                    feature_importance = pd.DataFrame({
                        'feature': feature_names,
                        'importance': importances
                    }).sort_values('importance', ascending=False)
                    
                    logger.info("\nTop 10 most important features:")
                    for _, row in feature_importance.head(10).iterrows():
                        logger.info(f"{row['feature']}: {row['importance']:.4f}")
                else:
                    logger.warning(f"Feature names count ({len(feature_names)}) doesn't match importances count ({len(importances)})")
            else:
                logger.warning("Model doesn't have feature_importances_ attribute")
        except Exception as e:
            logger.warning(f"Could not calculate feature importances: {str(e)}")
        
        logger.info("Model training completed")
    
    def predict_performance(self, player_data: pd.DataFrame) -> pd.Series:
        """Predict performance scores for players."""
        # Prepare features
        X = player_data[self.numerical_features + self.categorical_features]
        
        # Make predictions
        predictions = self.model.predict(X)
        
        # Normalize predictions to be between 0 and 1
        min_pred = predictions.min()
        max_pred = predictions.max()
        normalized_predictions = (predictions - min_pred) / (max_pred - min_pred)
        
        return pd.Series(normalized_predictions, index=player_data.index)
    
    def save_model(self, model_path: str = 'models/performance_model.joblib') -> None:
        """Save the trained model and preprocessing steps."""
        os.makedirs('models', exist_ok=True)
        joblib.dump({
            'model': self.model,
            'preprocessor': self.preprocessor,
            'numerical_features': self.numerical_features,
            'categorical_features': self.categorical_features
        }, model_path)
        logger.info(f"Model saved to {model_path}")
    
    @classmethod
    def load_model(cls, model_path: str = 'models/performance_model.joblib') -> 'PerformancePredictor':
        """Load a trained model and preprocessing steps."""
        predictor = cls()
        saved_data = joblib.load(model_path)
        predictor.model = saved_data['model']
        predictor.preprocessor = saved_data['preprocessor']
        predictor.numerical_features = saved_data['numerical_features']
        predictor.categorical_features = saved_data['categorical_features']
        return predictor

def main():
    # Load data
    historical_metrics = pd.read_csv('data/player_historical_metrics.csv')
    squad_data = pd.read_csv('backup/data/squaddata_allteams.csv')
    
    logger.info(f"Loaded historical metrics for {len(historical_metrics)} players")
    logger.info(f"Loaded squad data for {len(squad_data)} players")
    
    # Initialize and train the model
    predictor = PerformancePredictor()
    X, y = predictor.prepare_data(historical_metrics, squad_data)
    
    logger.info(f"Prepared training data with {X.shape[0]} samples and {X.shape[1]} features")
    
    if X.shape[0] > 0:
        predictor.train(X, y)
        
        # Save the trained model
        predictor.save_model()
        
        # Test predictions - use the same X data
        predictions = predictor.predict_performance(X)
        logger.info("\nSample predictions (first 10):")
        for i, (idx, score) in enumerate(zip(X.index, predictions)):
            if i < 10:  # Only show first 10
                player_name = historical_metrics.loc[idx, 'player_name'] if idx in historical_metrics.index else "Unknown"
                logger.info(f"{player_name}: {score:.2f}")
    else:
        logger.error("No data available for training. Check your input files.")

if __name__ == "__main__":
    main() 