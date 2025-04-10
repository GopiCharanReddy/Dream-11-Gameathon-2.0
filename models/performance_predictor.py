import pandas as pd
import numpy as np
import joblib
import os
from typing import List, Dict, Tuple
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PerformancePredictor:
    """Predicts player performance for Dream11 team selection."""
    
    def __init__(self):
        """Initialize the performance predictor."""
        self.model = None
        self.preprocessor = None
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
        self.categorical_features = ['role', 'team']
        self.feature_columns = self.numerical_features + self.categorical_features
    
    @classmethod
    def load_model(cls, model_path: str = 'models/performance_model.joblib') -> 'PerformancePredictor':
        """Load a trained model."""
        predictor = cls()
        
        try:
            if not os.path.exists(model_path):
                logger.warning(f"Model file {model_path} not found. Using deterministic model.")
                return predictor
            
            saved_data = joblib.load(model_path)
            
            predictor.model = saved_data.get('model')
            predictor.preprocessor = saved_data.get('preprocessor')
            
            # Update feature columns if available in the model
            if 'numerical_features' in saved_data:
                predictor.numerical_features = saved_data['numerical_features']
            if 'categorical_features' in saved_data:
                predictor.categorical_features = saved_data['categorical_features']
            
            predictor.feature_columns = predictor.numerical_features + predictor.categorical_features
            logger.info(f"Model loaded from {model_path}")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}. Using deterministic model.")
        
        return predictor
    
    def predict_performance(self, player_data: pd.DataFrame) -> pd.Series:
        """Predict performance scores for players.
        
        If a trained model is available, it will be used.
        Otherwise, falls back to a deterministic scoring method.
        """
        if self.model is not None:
            try:
                # Prepare features
                X = player_data[self.feature_columns].copy()
                
                # Ensure role column exists and is properly formatted
                if 'role' in X.columns:
                    # Standardize role values
                    role_mapping = {
                        'BAT': 'BAT',
                        'BOWL': 'BOWL',
                        'AR': 'AR',
                        'ALL': 'AR',
                        'WK': 'WK',
                        'WICKETKEEPER': 'WK',
                        'BATSMAN': 'BAT',
                        'BOWLER': 'BOWL',
                        'ALLROUNDER': 'AR'
                    }
                    X['role'] = X['role'].map(role_mapping).fillna('UNKNOWN')
                    
                    # Add role-specific features if they don't exist
                    if 'is_batsman' not in X.columns:
                        X['is_batsman'] = (X['role'] == 'BAT').astype(int)
                    if 'is_bowler' not in X.columns:
                        X['is_bowler'] = (X['role'] == 'BOWL').astype(int)
                    if 'is_allrounder' not in X.columns:
                        X['is_allrounder'] = (X['role'] == 'AR').astype(int)
                    if 'is_wicketkeeper' not in X.columns:
                        X['is_wicketkeeper'] = (X['role'] == 'WK').astype(int)
                    
                    # Add versatility score - players who can perform multiple roles
                    X['versatility_score'] = 0
                    if 'batting_avg' in X.columns and 'economy_rate' in X.columns:
                        # Players with good batting and bowling stats are more versatile
                        X['versatility_score'] = (
                            (X['batting_avg'].fillna(0) / 50) * 0.5 +  # Normalize batting avg (50 is good)
                            (1 - X['economy_rate'].fillna(12) / 12) * 0.5  # Lower economy is better
                        ).clip(0, 1)
                
                # Fill missing values for numerical features
                for col in self.numerical_features:
                    if col in X.columns:
                        X[col] = X[col].fillna(0)
                
                # Fill missing values for categorical features
                for col in self.categorical_features:
                    if col in X.columns:
                        X[col] = X[col].fillna('UNKNOWN')
                
                # Make predictions
                predictions = self.model.predict(X)
                
                # Normalize predictions to be between 0 and 1
                min_pred = predictions.min()
                max_pred = predictions.max()
                
                # Avoid division by zero
                if max_pred > min_pred:
                    normalized_predictions = (predictions - min_pred) / (max_pred - min_pred)
                else:
                    normalized_predictions = predictions
                
                logger.info("Used trained model for predictions")
                return pd.Series(normalized_predictions, index=player_data.index)
            
            except Exception as e:
                logger.error(f"Error using trained model: {str(e)}. Falling back to deterministic method.")
        
        # Fall back to deterministic method
        logger.info("Using deterministic method for predictions")
        return self._deterministic_score(player_data)
    
    def _deterministic_score(self, player_data: pd.DataFrame) -> pd.Series:
        """Calculate deterministic performance scores based on player attributes."""
        scores = []
        
        for _, player in player_data.iterrows():
            # Base score from credits (assuming higher credit players are better)
            base_score = 0.5  # Default value
            if 'credits' in player and not pd.isna(player['credits']):
                base_score = float(player['credits']) / 10.0
            
            # Role-specific adjustments
            role = None
            for role_field in ['role', 'Role', 'Player Type']:
                if role_field in player and not pd.isna(player[role_field]):
                    role = player[role_field]
                    break
                    
            role_factor = 1.0
            if role:
                role_mapping = {
                    'BAT': 'BAT',
                    'BOWL': 'BOWL',
                    'AR': 'AR',
                    'ALL': 'AR',  # Map ALL to AR
                    'WK': 'WK'
                }
                std_role = role_mapping.get(role, role)
                role_factor = {
                    'BAT': 1.0,
                    'BOWL': 1.0,
                    'AR': 1.1,  # All-rounders get a slight boost
                    'WK': 0.95  # Wicketkeepers slightly lower
                }.get(std_role, 1.0)
            
            # Team factor - could be based on team's performance
            team_factor = 1.0
            
            # Lineup order adjustment
            lineup_factor = 1.0
            if 'lineup_order' in player and not pd.isna(player['lineup_order']):
                try:
                    lineup_order = float(player['lineup_order'])
                    # Batsmen benefit from batting early
                    if role in ['BAT', 'WK'] and isinstance(lineup_order, (int, float)):
                        if lineup_order <= 3:
                            lineup_factor = 1.1
                        elif lineup_order <= 5:
                            lineup_factor = 1.05
                    
                    # Bowlers less affected by lineup order
                    if role in ['BOWL'] and isinstance(lineup_order, (int, float)):
                        lineup_factor = 1.0
                except (ValueError, TypeError):
                    pass
            
            # Historical stats adjustment if available
            stats_factor = 1.0
            batting_avg = player.get('batting_avg', 0)
            strike_rate = player.get('strike_rate', 0)
            economy_rate = player.get('economy_rate', 0)
            
            if not pd.isna(batting_avg) and not pd.isna(strike_rate) and not pd.isna(economy_rate):
                # Batters value strike rate and average
                if role in ['BAT', 'WK']:
                    if batting_avg > 30:
                        stats_factor *= 1.1
                    if strike_rate > 140:
                        stats_factor *= 1.1
                
                # Bowlers value economy rate
                if role in ['BOWL']:
                    if economy_rate < 8:
                        stats_factor *= 1.1
                
                # All-rounders benefit from both batting and bowling stats
                if role in ['AR']:
                    if batting_avg > 25 and strike_rate > 130:
                        stats_factor *= 1.1
                    if economy_rate < 9:
                        stats_factor *= 1.1
            
            # Recent form adjustment
            form_factor = 1.0
            recent_runs = player.get('recent_runs', 0)
            recent_wickets = player.get('recent_wickets', 0)
            
            if not pd.isna(recent_runs) and not pd.isna(recent_wickets):
                if role in ['BAT', 'WK'] and recent_runs > 30:
                    form_factor *= 1.1
                if role in ['BOWL'] and recent_wickets > 2:
                    form_factor *= 1.1
                if role in ['AR'] and (recent_runs > 20 or recent_wickets > 1):
                    form_factor *= 1.1
            
            # Calculate final score
            final_score = base_score * role_factor * team_factor * lineup_factor * stats_factor * form_factor
            
            # Add some randomness (optional)
            random_factor = np.random.normal(1.0, 0.05)  # 5% variation
            final_score *= random_factor
            
            # Ensure score is between 0 and 1
            final_score = max(0, min(1, final_score))
            
            scores.append(final_score)
        
        return pd.Series(scores, index=player_data.index) 