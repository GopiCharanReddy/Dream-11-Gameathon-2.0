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
from datetime import datetime
from sklearn.impute import SimpleImputer

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
        
        # Ensure role column exists and is properly formatted
        if 'role' in df.columns:
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
            df['role'] = df['role'].map(role_mapping).fillna('UNKNOWN')
            
            # Add role-specific features to help the model understand roles better
            df['is_batsman'] = (df['role'] == 'BAT').astype(int)
            df['is_bowler'] = (df['role'] == 'BOWL').astype(int)
            df['is_allrounder'] = (df['role'] == 'AR').astype(int)
            df['is_wicketkeeper'] = (df['role'] == 'WK').astype(int)
            
            # Add these new features to numerical features
            self.numerical_features.extend(['is_batsman', 'is_bowler', 'is_allrounder', 'is_wicketkeeper'])
        else:
            logger.warning("Role column not found in historical metrics. Adding default role column.")
            df['role'] = 'UNKNOWN'
        
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

    def _process_match_data(self, match_data):
        """Process match data to extract player performance metrics."""
        try:
            # Extract match metadata
            match_id = match_data.get('match_id', 'unknown')
            match_date = match_data.get('match_date', '')
            teams = match_data.get('teams', [])
            
            # Try to parse the date
            try:
                if match_date and isinstance(match_date, str) and match_date.strip():
                    # Try different date formats
                    for date_format in ['%Y-%m-%d', '%d-%m-%Y', '%Y/%m/%d', '%d/%m/%Y']:
                        try:
                            parsed_date = datetime.strptime(match_date.strip(), date_format)
                            match_date = parsed_date
                            break
                        except ValueError:
                            continue
                else:
                    # If date is empty or invalid, use current date
                    match_date = datetime.now()
                    logger.warning(f"Empty or invalid date for match {match_id}, using current date")
            except Exception as e:
                logger.warning(f"Error parsing date for match {match_id}: {str(e)}")
                match_date = datetime.now()
            
            # Process each delivery
            for delivery in match_data.get('deliveries', []):
                try:
                    # Extract basic delivery info
                    batter = delivery.get('batter', '')
                    bowler = delivery.get('bowler', '')
                    runs = delivery.get('runs', 0)
                    extras = delivery.get('extras', 0)
                    is_wicket = delivery.get('is_wicket', False)
                    dismissal_type = delivery.get('dismissal_type', '')
                    
                    # Skip if no batter or bowler
                    if not batter or not bowler:
                        continue
                    
                    # Update batter stats
                    if batter in self.player_stats:
                        self.player_stats[batter]['runs'] += runs
                        self.player_stats[batter]['balls_faced'] += 1
                        self.player_stats[batter]['fours'] += 1 if runs == 4 else 0
                        self.player_stats[batter]['sixes'] += 1 if runs == 6 else 0
                        self.player_stats[batter]['dot_balls'] += 1 if runs == 0 else 0
                        
                        # Update strike rate
                        if self.player_stats[batter]['balls_faced'] > 0:
                            self.player_stats[batter]['strike_rate'] = (
                                self.player_stats[batter]['runs'] / 
                                self.player_stats[batter]['balls_faced'] * 100
                            )
                        
                        # Update boundary rate
                        if self.player_stats[batter]['balls_faced'] > 0:
                            self.player_stats[batter]['boundary_rate'] = (
                                (self.player_stats[batter]['fours'] + self.player_stats[batter]['sixes']) / 
                                self.player_stats[batter]['balls_faced'] * 100
                            )
                        
                        # Update dot ball rate
                        if self.player_stats[batter]['balls_faced'] > 0:
                            self.player_stats[batter]['dot_ball_rate'] = (
                                self.player_stats[batter]['dot_balls'] / 
                                self.player_stats[batter]['balls_faced'] * 100
                            )
                    
                    # Update bowler stats
                    if bowler in self.player_stats:
                        self.player_stats[bowler]['balls_bowled'] += 1
                        self.player_stats[bowler]['runs_conceded'] += runs + extras
                        self.player_stats[bowler]['dot_balls_bowled'] += 1 if runs == 0 else 0
                        
                        # Update economy rate
                        if self.player_stats[bowler]['balls_bowled'] > 0:
                            self.player_stats[bowler]['economy_rate'] = (
                                self.player_stats[bowler]['runs_conceded'] / 
                                (self.player_stats[bowler]['balls_bowled'] / 6)
                            )
                        
                        # Update dot ball percentage
                        if self.player_stats[bowler]['balls_bowled'] > 0:
                            self.player_stats[bowler]['dot_ball_percentage'] = (
                                self.player_stats[bowler]['dot_balls_bowled'] / 
                                self.player_stats[bowler]['balls_bowled'] * 100
                            )
                    
                    # Update wicket stats
                    if is_wicket and bowler in self.player_stats:
                        self.player_stats[bowler]['wickets'] += 1
                        
                        # Update bowling average
                        if self.player_stats[bowler]['wickets'] > 0:
                            self.player_stats[bowler]['bowling_avg'] = (
                                self.player_stats[bowler]['runs_conceded'] / 
                                self.player_stats[bowler]['wickets']
                            )
                        
                        # Update bowling strike rate
                        if self.player_stats[bowler]['wickets'] > 0:
                            self.player_stats[bowler]['bowling_sr'] = (
                                self.player_stats[bowler]['balls_bowled'] / 
                                self.player_stats[bowler]['wickets']
                            )
                    
                    # Update death over stats (last 5 overs)
                    over_number = delivery.get('over', 0)
                    if over_number >= 15:  # Last 5 overs
                        if batter in self.player_stats:
                            self.player_stats[batter]['death_over_runs'] += runs
                            self.player_stats[batter]['death_over_balls'] += 1
                            
                            # Update death over strike rate
                            if self.player_stats[batter]['death_over_balls'] > 0:
                                self.player_stats[batter]['death_over_strike_rate'] = (
                                    self.player_stats[batter]['death_over_runs'] / 
                                    self.player_stats[batter]['death_over_balls'] * 100
                                )
                        
                        if bowler in self.player_stats:
                            self.player_stats[bowler]['death_over_runs'] += runs + extras
                            self.player_stats[bowler]['death_over_balls'] += 1
                            
                            # Update death over economy
                            if self.player_stats[bowler]['death_over_balls'] > 0:
                                self.player_stats[bowler]['death_over_economy'] = (
                                    self.player_stats[bowler]['death_over_runs'] / 
                                    (self.player_stats[bowler]['death_over_balls'] / 6)
                                )
                    
                    # Update wicketkeeper stats
                    if dismissal_type in ['caught', 'stumped', 'run out']:
                        for player, stats in self.player_stats.items():
                            if stats.get('is_wicketkeeper', False):
                                if dismissal_type == 'stumped':
                                    stats['stumpings'] += 1
                                elif dismissal_type == 'caught':
                                    stats['catches'] += 1
                                elif dismissal_type == 'run out':
                                    stats['run_outs'] += 1
                                
                                # Update dismissals per match
                                stats['dismissals'] += 1
                                if stats['matches_played'] > 0:
                                    stats['dismissals_per_match'] = stats['dismissals'] / stats['matches_played']
                                    stats['catches_per_match'] = stats['catches'] / stats['matches_played']
                                    stats['stumpings_per_match'] = stats['stumpings'] / stats['matches_played']
                
                except Exception as e:
                    logger.warning(f"Error processing delivery in match {match_id}: {str(e)}")
                    continue
            
            # Update match counts and batting/bowling averages
            for player, stats in self.player_stats.items():
                if stats.get('runs', 0) > 0 or stats.get('wickets', 0) > 0:
                    stats['matches_played'] += 1
                    
                    # Update batting average
                    if stats.get('innings_batted', 0) > 0:
                        stats['batting_avg'] = stats['runs'] / stats['innings_batted']
                    
                    # Update innings batted
                    if stats.get('runs', 0) > 0:
                        stats['innings_batted'] += 1
                    
                    # Calculate impact scores
                    if stats.get('runs', 0) > 0:
                        # Batting impact: runs * strike rate / 100
                        stats['batting_impact'] = stats['runs'] * stats.get('strike_rate', 0) / 100
                    
                    if stats.get('wickets', 0) > 0:
                        # Bowling impact: wickets * (1 - economy_rate/12)
                        stats['bowling_impact'] = stats['wickets'] * (1 - stats.get('economy_rate', 12) / 12)
                    
                    # All-rounder score: combination of batting and bowling impact
                    if stats.get('batting_impact', 0) > 0 and stats.get('bowling_impact', 0) > 0:
                        stats['all_rounder_score'] = (stats['batting_impact'] + stats['bowling_impact']) / 2
                    
                    # Add form tracking - store recent performances
                    if 'recent_performances' not in stats:
                        stats['recent_performances'] = []
                    
                    # Store last 5 performances
                    recent_perf = {
                        'match_id': match_id,
                        'date': match_date,
                        'runs': stats.get('runs', 0),
                        'wickets': stats.get('wickets', 0),
                        'strike_rate': stats.get('strike_rate', 0),
                        'economy_rate': stats.get('economy_rate', 0)
                    }
                    
                    stats['recent_performances'].append(recent_perf)
                    if len(stats['recent_performances']) > 5:
                        stats['recent_performances'].pop(0)
                    
                    # Calculate recent form metrics
                    if len(stats['recent_performances']) > 0:
                        stats['recent_runs'] = sum(p['runs'] for p in stats['recent_performances'])
                        stats['recent_wickets'] = sum(p['wickets'] for p in stats['recent_performances'])
                        stats['recent_strike_rate'] = sum(p['strike_rate'] for p in stats['recent_performances']) / len(stats['recent_performances'])
                        stats['recent_economy_rate'] = sum(p['economy_rate'] for p in stats['recent_performances']) / len(stats['recent_performances'])
                    
                    # Add team performance tracking
                    if 'team_performances' not in stats:
                        stats['team_performances'] = {}
                    
                    for team in teams:
                        if team in stats.get('teams', []):
                            if team not in stats['team_performances']:
                                stats['team_performances'][team] = {
                                    'matches': 0,
                                    'runs': 0,
                                    'wickets': 0,
                                    'strike_rate': 0,
                                    'economy_rate': 0
                                }
                            
                            team_stats = stats['team_performances'][team]
                            team_stats['matches'] += 1
                            team_stats['runs'] += stats.get('runs', 0)
                            team_stats['wickets'] += stats.get('wickets', 0)
                            
                            # Update team-specific strike rate and economy
                            if stats.get('innings_batted', 0) > 0:
                                team_stats['strike_rate'] = team_stats['runs'] / stats.get('innings_batted', 1) * 100
                            
                            if stats.get('balls_bowled', 0) > 0:
                                team_stats['economy_rate'] = team_stats['runs'] / (stats.get('balls_bowled', 1) / 6)
        
        except Exception as e:
            logger.error(f"Error processing match data: {str(e)}")

    def _create_training_data(self):
        """Create training data for the model."""
        # Create a DataFrame from player stats
        player_data = []
        
        for player_name, stats in self.player_stats.items():
            player_info = {
                'player_name': player_name,
                'team': stats.get('teams', ['UNKNOWN'])[0],  # Use first team as primary
                'role': stats.get('role', 'UNKNOWN'),
                'credits': stats.get('credits', 0),
                'is_all_rounder': 1 if stats.get('is_all_rounder', False) else 0,
                'is_wicketkeeper': 1 if stats.get('is_wicketkeeper', False) else 0,
                
                # Batting stats
                'batting_avg': stats.get('batting_avg', 0),
                'strike_rate': stats.get('strike_rate', 0),
                'boundary_rate': stats.get('boundary_rate', 0),
                'dot_ball_rate': stats.get('dot_ball_rate', 0),
                'death_over_strike_rate': stats.get('death_over_strike_rate', 0),
                
                # Bowling stats
                'economy_rate': stats.get('economy_rate', 0),
                'bowling_avg': stats.get('bowling_avg', 0),
                'bowling_sr': stats.get('bowling_sr', 0),
                'dot_ball_percentage': stats.get('dot_ball_percentage', 0),
                'death_over_economy': stats.get('death_over_economy', 0),
                
                # Wicketkeeping stats
                'dismissals_per_match': stats.get('dismissals_per_match', 0),
                'catches_per_match': stats.get('catches_per_match', 0),
                'stumpings_per_match': stats.get('stumpings_per_match', 0),
                
                # Form stats
                'recent_runs': stats.get('recent_runs', 0),
                'recent_wickets': stats.get('recent_wickets', 0),
                
                # Impact scores
                'batting_impact': stats.get('batting_impact', 0),
                'bowling_impact': stats.get('bowling_impact', 0),
                'all_rounder_score': stats.get('all_rounder_score', 0),
                
                # Match context
                'total_innings': stats.get('innings_batted', 0)
            }
            
            # Add team-specific performance
            for team, team_stats in stats.get('team_performances', {}).items():
                player_info[f'team_{team}_matches'] = team_stats.get('matches', 0)
                player_info[f'team_{team}_runs'] = team_stats.get('runs', 0)
                player_info[f'team_{team}_wickets'] = team_stats.get('wickets', 0)
                player_info[f'team_{team}_strike_rate'] = team_stats.get('strike_rate', 0)
                player_info[f'team_{team}_economy_rate'] = team_stats.get('economy_rate', 0)
            
            # Add role-specific features
            player_info['is_batsman'] = 1 if stats.get('role') in ['BAT', 'WK'] else 0
            player_info['is_bowler'] = 1 if stats.get('role') in ['BOWL'] else 0
            player_info['is_allrounder'] = 1 if stats.get('role') in ['AR'] else 0
            player_info['is_wicketkeeper'] = 1 if stats.get('role') in ['WK'] else 0
            
            # Add versatility score
            if stats.get('batting_avg', 0) > 0 and stats.get('economy_rate', 0) > 0:
                player_info['versatility_score'] = (
                    (stats.get('batting_avg', 0) / 50) * 0.5 +  # Normalize batting avg (50 is good)
                    (1 - stats.get('economy_rate', 12) / 12) * 0.5  # Lower economy is better
                )
            else:
                player_info['versatility_score'] = 0
            
            # Add form trend (improving or declining)
            if len(stats.get('recent_performances', [])) >= 2:
                recent_perfs = stats['recent_performances']
                if stats.get('role') in ['BAT', 'WK']:
                    # For batsmen, check if runs are increasing
                    player_info['form_trend'] = 1 if recent_perfs[-1]['runs'] > recent_perfs[0]['runs'] else 0
                elif stats.get('role') in ['BOWL']:
                    # For bowlers, check if wickets are increasing
                    player_info['form_trend'] = 1 if recent_perfs[-1]['wickets'] > recent_perfs[0]['wickets'] else 0
                elif stats.get('role') in ['AR']:
                    # For all-rounders, check if combined performance is improving
                    recent_impact = recent_perfs[-1]['runs'] * 0.5 + recent_perfs[-1]['wickets'] * 2
                    old_impact = recent_perfs[0]['runs'] * 0.5 + recent_perfs[0]['wickets'] * 2
                    player_info['form_trend'] = 1 if recent_impact > old_impact else 0
                else:
                    player_info['form_trend'] = 0
            else:
                player_info['form_trend'] = 0
            
            # Add consistency score (standard deviation of recent performances)
            if len(stats.get('recent_performances', [])) >= 2:
                recent_runs = [p['runs'] for p in stats['recent_performances']]
                recent_wickets = [p['wickets'] for p in stats['recent_performances']]
                
                if stats.get('role') in ['BAT', 'WK']:
                    # For batsmen, lower standard deviation means more consistent
                    std_dev = np.std(recent_runs) if len(recent_runs) > 1 else 0
                    player_info['consistency_score'] = 1 - (std_dev / (np.mean(recent_runs) + 1))
                elif stats.get('role') in ['BOWL']:
                    # For bowlers, lower standard deviation means more consistent
                    std_dev = np.std(recent_wickets) if len(recent_wickets) > 1 else 0
                    player_info['consistency_score'] = 1 - (std_dev / (np.mean(recent_wickets) + 1))
                elif stats.get('role') in ['AR']:
                    # For all-rounders, consider both batting and bowling consistency
                    runs_std = np.std(recent_runs) if len(recent_runs) > 1 else 0
                    wickets_std = np.std(recent_wickets) if len(recent_wickets) > 1 else 0
                    runs_consistency = 1 - (runs_std / (np.mean(recent_runs) + 1))
                    wickets_consistency = 1 - (wickets_std / (np.mean(recent_wickets) + 1))
                    player_info['consistency_score'] = (runs_consistency + wickets_consistency) / 2
                else:
                    player_info['consistency_score'] = 0
            else:
                player_info['consistency_score'] = 0
            
            player_data.append(player_info)
        
        # Convert to DataFrame
        df = pd.DataFrame(player_data)
        
        # Fill missing values
        for col in df.columns:
            if col not in ['player_name', 'team', 'role']:
                df[col] = df[col].fillna(0)
        
        return df

    def train_model(self):
        """Train the performance prediction model."""
        try:
            # Create training data
            df = self._create_training_data()
            
            if df.empty:
                logger.error("No training data available")
                return
            
            # Log available columns
            logger.info(f"Historical metrics columns: {df.columns.tolist()}")
            
            # Define features and target
            numerical_features = [
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
                'total_innings',
                
                # New features
                'versatility_score', 'form_trend', 'consistency_score'
            ]
            
            # Add team-specific features
            team_features = [col for col in df.columns if col.startswith('team_')]
            numerical_features.extend(team_features)
            
            categorical_features = ['role', 'team']
            
            # Create feature matrix
            X = df[numerical_features + categorical_features].copy()
            
            # Create target variable (performance score)
            # For this example, we'll use a combination of batting and bowling stats
            y = np.zeros(len(df))
            
            for i, row in df.iterrows():
                # Base score from credits
                base_score = row['credits'] / 10.0
                
                # Role-specific scoring
                if row['role'] in ['BAT', 'WK']:
                    # Batting score: batting_avg * strike_rate / 100
                    batting_score = row['batting_avg'] * row['strike_rate'] / 100
                    # Add form boost
                    form_boost = row['recent_runs'] / 50.0  # Normalize recent runs
                    y[i] = base_score * (1 + batting_score / 10 + form_boost)
                
                elif row['role'] == 'BOWL':
                    # Bowling score: wickets * (1 - economy_rate/12)
                    bowling_score = row['bowling_avg'] * (1 - row['economy_rate'] / 12)
                    # Add form boost
                    form_boost = row['recent_wickets'] / 5.0  # Normalize recent wickets
                    y[i] = base_score * (1 + bowling_score / 10 + form_boost)
                
                elif row['role'] == 'AR':
                    # All-rounder score: combination of batting and bowling
                    batting_score = row['batting_avg'] * row['strike_rate'] / 100
                    bowling_score = row['bowling_avg'] * (1 - row['economy_rate'] / 12)
                    # Add form boost
                    form_boost = (row['recent_runs'] / 50.0 + row['recent_wickets'] / 5.0) / 2
                    # Add versatility boost
                    versatility_boost = row['versatility_score']
                    y[i] = base_score * (1 + (batting_score + bowling_score) / 20 + form_boost + versatility_boost * 0.2)
                
                else:
                    # Default score for unknown roles
                    y[i] = base_score
            
            # Normalize target to [0, 1]
            y = (y - y.min()) / (y.max() - y.min())
            
            # Split data into training and validation sets
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Create preprocessing pipeline
            numerical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
                ('scaler', StandardScaler())
            ])
            
            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='constant', fill_value='UNKNOWN')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ])
            
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numerical_transformer, numerical_features),
                    ('cat', categorical_transformer, categorical_features)
                ]
            )
            
            # Create model pipeline
            model = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
            ])
            
            # Train model
            logger.info("Training performance prediction model...")
            model.fit(X_train, y_train)
            
            # Evaluate model
            train_score = model.score(X_train, y_train)
            val_score = model.score(X_val, y_val)
            
            logger.info(f"Training R² score: {train_score:.4f}")
            logger.info(f"Validation R² score: {val_score:.4f}")
            
            # Get feature importance
            feature_importance = model.named_steps['regressor'].feature_importances_
            feature_names = (
                numerical_features + 
                model.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_features).tolist()
            )
            
            # Sort feature importance
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': feature_importance
            }).sort_values('importance', ascending=False)
            
            logger.info("\nTop 10 most important features:")
            for _, row in importance_df.head(10).iterrows():
                logger.info(f"{row['feature']}: {row['importance']:.4f}")
            
            # Save model
            model_path = 'models/performance_model.joblib'
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            
            # Save model with additional metadata
            model_data = {
                'model': model,
                'numerical_features': numerical_features,
                'categorical_features': categorical_features,
                'feature_columns': numerical_features + categorical_features
            }
            
            joblib.dump(model_data, model_path)
            logger.info(f"Model saved to {model_path}")
            
            # Generate sample predictions
            sample_players = df.sample(min(10, len(df)))
            sample_X = sample_players[numerical_features + categorical_features]
            sample_predictions = model.predict(sample_X)
            
            logger.info("\nSample predictions (first 10):")
            for i, (_, player) in enumerate(sample_players.iterrows()):
                logger.info(f"{player['player_name']}: {sample_predictions[i]:.2f}")
            
            return model
            
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

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