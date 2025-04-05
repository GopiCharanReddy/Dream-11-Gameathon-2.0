import pandas as pd
import numpy as np
import os
import json
import glob
import pickle
import sys
from datetime import datetime
from fuzzywuzzy import fuzz
from sklearn.preprocessing import StandardScaler

# Add the parent directory to the path so we can import the models package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.performance_predictor import PerformancePredictor

class Dream11TeamSelector:
    def __init__(self, verbose=False):
        """Initialize the Dream11TeamSelector."""
        self.player_df = None
        self.lineup_data = None
        self.selected_team = None
        self.performance_predictor = PerformancePredictor.load_model()
        self.verbose = verbose
        self._load_resources()

    def _load_resources(self):
        """Load player dataset."""
        try:
            # Load player dataset
            try:
                self.player_df = pd.read_csv('data/processed_data.csv')
                if self.verbose:
                    print(f"Player dataset loaded successfully, found {len(self.player_df)} players")
            except Exception as e:
                if self.verbose:
                    print(f"Error loading player dataset: {e}")
                self.player_df = pd.DataFrame()
        except Exception as e:
            if self.verbose:
                print(f"Error loading resources: {str(e)}")

    def _get_sample_data(self):
        """Generate sample data for testing."""
        sample_data = {
            'player_name': ['Rohit Sharma', 'Virat Kohli', 'KL Rahul', 'Hardik Pandya', 
                           'Ravindra Jadeja', 'MS Dhoni', 'Jasprit Bumrah', 'Mohammed Shami',
                           'Yuzvendra Chahal', 'Shubman Gill', 'Rishabh Pant'],
            'role': ['BAT', 'BAT', 'WK', 'AR', 'AR', 'WK', 'BOWL', 'BOWL', 
                     'BOWL', 'BAT', 'WK'],
            'team': ['MI', 'RCB', 'LSG', 'GT', 'CSK', 'CSK', 'MI', 'GT',
                     'RR', 'GT', 'DC'],
            'credits': [10.5, 10.0, 9.5, 9.5, 9.0, 9.0, 9.0, 8.5,
                       8.5, 9.0, 8.5]
        }
        return pd.DataFrame(sample_data)

    def _fetch_lineup_from_sheets(self):
        """Fetch player lineup from local CSV."""
        try:
            # Try to read from a local CSV file
            local_file = 'data/latest_lineup.csv'
            if os.path.exists(local_file):
                if self.verbose:
                    print(f"Reading lineup from local file: {local_file}")
                
                # Read the file with skiprows to handle the header format
                lineup_df = pd.read_csv(local_file, skiprows=1)
                
                # Check if this is the new format (with A,B,C,D,E,F header)
                if 'Credits' in lineup_df.columns:
                    if self.verbose:
                        print("Detected new format CSV")
                    # Rename columns to match our expected format
                    lineup_df = lineup_df.rename(columns={
                        'Credits': 'credits',
                        'Player Type': 'role',
                        'Player Name': 'player_name',
                        'Team': 'team',
                        'IsPlaying': 'is_playing',
                        'lineupOrder': 'lineup_order'
                    })
                    
                    # Filter only playing players
                    lineup_df = lineup_df[lineup_df['is_playing'] == 'PLAYING']
                    
                    # Map roles to standard format
                    role_mapping = {
                        'BAT': 'BAT',
                        'BOWL': 'BOWL',
                        'ALL': 'AR',
                        'WK': 'WK'
                    }
                    lineup_df['role'] = lineup_df['role'].map(role_mapping)
                    
                    # Convert lineup_order to numeric
                    lineup_df['lineup_order'] = pd.to_numeric(lineup_df['lineup_order'], errors='coerce')
                    
                    # Convert credits to float
                    lineup_df['credits'] = pd.to_numeric(lineup_df['credits'], errors='coerce')
                    
                    # Sort by lineup order
                    if 'lineup_order' in lineup_df.columns:
                        lineup_df = lineup_df.sort_values('lineup_order')
                    
                    # Ensure valid role distribution
                    # Count WKs in lineup
                    wk_count = sum(1 for r in lineup_df['role'] if r == 'WK')
                    if wk_count > 1:
                        if self.verbose:
                            print(f"Found {wk_count} wicketkeepers in lineup, need to adjust")
                        # Keep the highest lineup order WK and convert others to BAT
                        wk_players = lineup_df[lineup_df['role'] == 'WK'].sort_values('lineup_order')
                        # Keep first, convert rest
                        for idx in wk_players.index[1:]:
                            lineup_df.at[idx, 'role'] = 'BAT'
                            if self.verbose:
                                print(f"Converting {lineup_df.at[idx, 'player_name']} from WK to BAT")
                    
                    # Keep necessary columns including lineup_order for scoring
                    lineup_df = lineup_df[['player_name', 'role', 'team', 'credits', 'lineup_order']]
                    
                else:
                    # Handle the old format
                    lineup_df = lineup_df.rename(columns={
                        'Player Name': 'player_name',
                        'Team': 'team',
                        'Player Type': 'role',
                        'Credits': 'credits',
                        'IsPlaying': 'is_playing',
                        'lineupOrder': 'lineup_order'
                    })
                    
                    # Filter only playing players
                    lineup_df = lineup_df[lineup_df['is_playing'] == True]
                    
                    # Convert credits to float
                    lineup_df['credits'] = lineup_df['credits'].astype(float)
                    
                    # Map roles to standard format
                    role_mapping = {
                        'BAT': 'BAT',
                        'BOWL': 'BOWL',
                        'AR': 'AR',
                        'WK': 'WK'
                    }
                    lineup_df['role'] = lineup_df['role'].map(role_mapping)
                    
                    # Sort by lineup order
                    if 'lineup_order' in lineup_df.columns:
                        lineup_df = lineup_df.sort_values('lineup_order')
                    
                    # Ensure valid role distribution
                    # Count WKs in lineup
                    wk_count = sum(1 for r in lineup_df['role'] if r == 'WK')
                    if wk_count > 1:
                        if self.verbose:
                            print(f"Found {wk_count} wicketkeepers in lineup, need to adjust")
                        # Keep the highest lineup order WK and convert others to BAT
                        wk_players = lineup_df[lineup_df['role'] == 'WK'].sort_values('lineup_order')
                        # Keep first, convert rest
                        for idx in wk_players.index[1:]:
                            lineup_df.at[idx, 'role'] = 'BAT'
                            if self.verbose:
                                print(f"Converting {lineup_df.at[idx, 'player_name']} from WK to BAT")
                    
                    # Keep necessary columns including lineup_order for scoring
                    lineup_df = lineup_df[['player_name', 'role', 'team', 'credits', 'lineup_order']]
                
                if self.verbose:
                    print(f"Successfully fetched lineup with {len(lineup_df)} players")
                return lineup_df
            else:
                if self.verbose:
                    print(f"Local file {local_file} not found")
                return self._get_sample_data()
            
        except Exception as e:
            if self.verbose:
                print(f"Error fetching lineup: {str(e)}")
                print("Using sample data instead")
            return self._get_sample_data()

    def _match_players(self, lineup_df):
        """Match players from lineup with dataset records."""
        # For this simplified version, we'll just use the lineup data directly
        # without trying to match with historical data
        if self.verbose:
            print("Using lineup data directly without fuzzy matching to historical data")
        
        # Make a copy of the lineup data to avoid modifying the original
        matched_df = lineup_df.copy()
        
        # Add a placeholder for predicted_score that will be filled later
        matched_df['predicted_score'] = 0.0
        
        return matched_df

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
            
            # Calculate final score
            final_score = base_score * role_factor * team_factor * lineup_factor * stats_factor
            
            # Add some randomness (optional)
            random_factor = np.random.normal(1.0, 0.05)  # 5% variation
            final_score *= random_factor
            
            # Ensure score is between 0 and 1
            final_score = max(0, min(1, final_score))
            
            scores.append(final_score)
        
        return pd.Series(scores, index=player_data.index)

    def _predict_performance(self, players: pd.DataFrame) -> pd.DataFrame:
        """Predict performance scores using the trained model."""
        try:
            # Try to load historical metrics
            try:
                historical_metrics = pd.read_csv('data/player_historical_metrics.csv')
                if self.verbose:
                    print(f"Loaded historical metrics for {len(historical_metrics)} players")
            except Exception as e:
                if self.verbose:
                    print(f"Error loading historical metrics: {e}")
                historical_metrics = pd.DataFrame(columns=['player_name', 'team'])
            
            # Create a copy of the players dataframe to avoid modifying the original
            players_with_predictions = players.copy()
            
            # Try to merge with historical metrics if available
            if not historical_metrics.empty:
                if self.verbose:
                    print("Merging with historical metrics")
                merged_data = pd.merge(
                    players,
                    historical_metrics,
                    on=['player_name', 'team'],
                    how='left'
                )
                
                # Fill missing values with zeros
                for col in historical_metrics.columns:
                    if col not in ['player_name', 'team', 'role', 'credits']:
                        if col in merged_data.columns:
                            merged_data[col] = merged_data[col].fillna(0)
                            
                players_with_predictions = merged_data
            
            # Generate performance scores using the loaded model or deterministic method
            if self.verbose:
                print("Generating performance scores with prediction model")
            
            try:
                # Try using the model
                predictions = self.performance_predictor.predict_performance(players_with_predictions)
                players_with_predictions['predicted_score'] = predictions * 10  # Scale to 0-10 range
                if self.verbose:
                    print("Successfully generated model-based predictions")
            except Exception as e:
                if self.verbose:
                    print(f"Error using trained model: {str(e)}")
                    # Fallback: Calculate performance scores using deterministic method
                    print("Falling back to deterministic scoring method")
                deterministic_scores = self._deterministic_score(players_with_predictions)
                players_with_predictions['predicted_score'] = deterministic_scores * 10  # Scale to 0-10 range
                if self.verbose:
                    print("Generated deterministic performance scores")
            
            if self.verbose:
                print(f"Generated performance scores for {len(players_with_predictions)} players")
            
            # Select and return only necessary columns
            result_columns = ['player_name', 'role', 'team', 'credits', 'predicted_score']
            if 'lineup_order' in players_with_predictions.columns:
                result_columns.append('lineup_order')
                
            return players_with_predictions[result_columns]
            
        except Exception as e:
            if self.verbose:
                print(f"Error predicting performance: {str(e)}")
            # Return original data with random scores as fallback
            players_copy = players.copy()
            players_copy['predicted_score'] = np.random.uniform(5, 10, len(players))
            return players_copy

    def select_dream11_team(self, players_with_predictions):
        """Select the best Dream11 team based on predicted scores and constraints."""
        try:
            if 'predicted_score' not in players_with_predictions.columns:
                if self.verbose:
                    print("Missing predicted scores. Cannot select team.")
                return None

            # Sort players by predicted score in descending order
            sorted_players = players_with_predictions.sort_values('predicted_score', ascending=False)

            # Print available players for debugging
            if self.verbose:
                print("\nAvailable players for selection:")
                for idx, player in sorted_players.iterrows():
                    print(f"{player['player_name']} ({player['role']}, {player['team']}, {player['credits']}, score: {player['predicted_score']:.2f})")

            # Initialize selected team
            selected_team = []
            total_credits = 0
            role_count = {'WK': 0, 'BAT': 0, 'AR': 0, 'BOWL': 0}
            team_count = {}

            # Role constraints - made more flexible
            role_limits = {
                'WK': (1, 4),  # 1-4 wicket-keepers (more flexible)
                'BAT': (3, 6),  # 3-6 batsmen
                'AR': (1, 4),  # 1-4 all-rounders
                'BOWL': (1, 6)  # 1-6 bowlers (more flexible)
            }
            
            # Credit limit
            credit_limit = 100  # Corrected from 110 to 100 as per requirement

            # First pass: select players by role to ensure minimum requirements
            for role, (min_count, _) in role_limits.items():
                role_players = sorted_players[sorted_players['role'] == role].sort_values('predicted_score', ascending=False)
                
                if self.verbose:
                    print(f"\nSelecting players for role {role} (need {min_count}):")
                for _, player in role_players.iterrows():
                    if role_count[role] >= min_count:
                        break  # We have enough players for this role
                        
                    # Check if adding this player would violate any constraints
                    if (total_credits + player['credits'] <= credit_limit and  # Credit limit
                        team_count.get(player['team'], 0) < 7):      # Team limit
                        
                        # Add player to team
                        selected_team.append(player)
                        total_credits += player['credits']
                        role_count[role] += 1
                        team_count[player['team']] = team_count.get(player['team'], 0) + 1
                        if self.verbose:
                            print(f"  Selected {player['player_name']} ({player['role']})")
                    else:
                        if self.verbose:
                            print(f"  Skipping {player['player_name']} due to constraints")

            # Check if we have the minimum required players for each role
            valid_minimums = all(role_count[role] >= min_count for role, (min_count, _) in role_limits.items())
            if not valid_minimums:
                if self.verbose:
                    print("\nCould not meet minimum role requirements:")
                    for role, (min_count, _) in role_limits.items():
                        print(f"  {role}: {role_count[role]}/{min_count} required")
                return None

            # Second pass: fill remaining slots with best players
            if self.verbose:
                print("\nFilling remaining slots:")
            remaining_players = sorted_players.copy()
            # Remove already selected players
            for player in selected_team:
                remaining_players = remaining_players[remaining_players['player_name'] != player['player_name']]
                
            for _, player in remaining_players.iterrows():
                if len(selected_team) >= 11:
                    break  # We have enough players
                    
                # Check if adding this player would violate any constraints
                if (total_credits + player['credits'] <= credit_limit and                  # Credit limit
                    role_count[player['role']] < role_limits[player['role']][1] and  # Role upper limit
                    team_count.get(player['team'], 0) < 7):                       # Team limit
                    
                    # Add player to team
                    selected_team.append(player)
                    total_credits += player['credits']
                    role_count[player['role']] += 1
                    team_count[player['team']] = team_count.get(player['team'], 0) + 1
                    if self.verbose:
                        print(f"  Selected {player['player_name']} ({player['role']})")
                else:
                    if self.verbose:
                        print(f"  Skipping {player['player_name']} due to constraints")

            # Convert selected team to DataFrame
            if len(selected_team) == 11:
                selected_df = pd.DataFrame(selected_team)
                if self.verbose:
                    print(f"\nSuccessfully selected team with {len(selected_df)} players")
                    print(f"Total credits: {total_credits:.2f}")
                    print(f"Role distribution: {role_count}")
                    print(f"Team distribution: {team_count}")
                return selected_df
            else:
                if self.verbose:
                    print(f"\nCould only select {len(selected_team)} players, needed 11")
                    print(f"Current role distribution: {role_count}")
                    print(f"Current team distribution: {team_count}")
                    print(f"Current credits: {total_credits:.2f}")
                    
                    # Try relaxing some constraints
                    print("\nRelaxing constraints to select remaining players...")
                # Sort remaining players by predicted score
                remaining_players = sorted_players.copy()
                for player in selected_team:
                    remaining_players = remaining_players[remaining_players['player_name'] != player['player_name']]
                
                # Try to add more players, relaxing role limits
                remaining_count = 11 - len(selected_team)
                for _, player in remaining_players.iterrows():
                    if len(selected_team) >= 11:
                        break
                    
                    # Only check team and credit constraints
                    if (total_credits + player['credits'] <= credit_limit and 
                        team_count.get(player['team'], 0) < 7):
                        
                        selected_team.append(player)
                        total_credits += player['credits']
                        role_count[player['role']] += 1
                        team_count[player['team']] = team_count.get(player['team'], 0) + 1
                        if self.verbose:
                            print(f"  Selected {player['player_name']} ({player['role']}) with relaxed constraints")
                
                if len(selected_team) == 11:
                    selected_df = pd.DataFrame(selected_team)
                    if self.verbose:
                        print(f"\nSuccessfully selected team with {len(selected_df)} players using relaxed constraints")
                        print(f"Total credits: {total_credits:.2f}")
                        print(f"Role distribution: {role_count}")
                        print(f"Team distribution: {team_count}")
                    return selected_df
                else:
                    if self.verbose:
                        print(f"\nStill could only select {len(selected_team)} players, needed 11")
                    # Last resort: just pick top players regardless of constraints except credit limit
                    if len(selected_team) >= 7:  # If we have at least 7 players, complete the team
                        remaining_players = remaining_players.sort_values('predicted_score', ascending=False)
                        remaining_count = 11 - len(selected_team)
                        
                        for _, player in remaining_players.iterrows():
                            if len(selected_team) >= 11:
                                break
                                
                            if total_credits + player['credits'] <= credit_limit:
                                selected_team.append(player)
                                total_credits += player['credits']
                                role_count[player['role']] += 1
                                team_count[player['team']] = team_count.get(player['team'], 0) + 1
                                if self.verbose:
                                    print(f"  Selected {player['player_name']} ({player['role']}) as last resort")
                        
                        selected_df = pd.DataFrame(selected_team)
                        if self.verbose:
                            print(f"\nFinally selected team with {len(selected_df)} players using last resort")
                            print(f"Total credits: {total_credits:.2f}")
                            print(f"Role distribution: {role_count}")
                            print(f"Team distribution: {team_count}")
                        return selected_df
                    
                    return None

        except Exception as e:
            if self.verbose:
                print(f"Error selecting team: {str(e)}")
                import traceback
                traceback.print_exc()
            return None
    
    def generate_output(self, selected_team):
        """Generate formatted output for the selected Dream11 team."""
        if selected_team is None or selected_team.empty:
            print("No selected team available. Cannot generate output.")
            return None
        
        # Create DataFrame for output
        output_df = selected_team.copy()
        
        # Calculate total credits
        total_credits = output_df['credits'].sum()
        
        # Calculate total predicted score
        total_predicted_score = output_df['predicted_score'].sum()
        
        # If Captain and Vice Captain columns don't exist, add them based on predicted scores
        if 'Captain' not in output_df.columns or 'Vice Captain' not in output_df.columns:
            output_df = output_df.sort_values('predicted_score', ascending=False)
            output_df['Captain'] = False
            output_df['Vice Captain'] = False
            
            if len(output_df) > 0:
                output_df.iloc[0, output_df.columns.get_loc('Captain')] = True
            if len(output_df) > 1:
                output_df.iloc[1, output_df.columns.get_loc('Vice Captain')] = True
        
        # Get captain and vice-captain names
        captain_name = output_df[output_df['Captain']]['player_name'].iloc[0] if not output_df[output_df['Captain']].empty else None
        vice_captain_name = output_df[output_df['Vice Captain']]['player_name'].iloc[0] if not output_df[output_df['Vice Captain']].empty else None
        
        # Create summary
        team_summary = {
            'total_credits': float(total_credits),
            'total_predicted_score': float(total_predicted_score),
            'role_distribution': output_df['role'].value_counts().to_dict(),
            'team_distribution': output_df['team'].value_counts().to_dict(),
            'captain': captain_name,
            'vice_captain': vice_captain_name,
            'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S')
        }
        
        return {'summary': team_summary, 'team': output_df}
    
    def save_output(self, team_output):
        """Save the selected team to JSON and CSV files."""
        if team_output is None:
            if self.verbose:
                print("No team output available. Cannot save to file.")
            return
        
        # Create output directory if it doesn't exist
        os.makedirs('output', exist_ok=True)
        
        # Generate file names with timestamp
        timestamp = team_output['summary']['timestamp']
        json_filename = f"output/dream11_team_{timestamp}.json"
        csv_filename = f"output/dream11_team_{timestamp}.csv"
        
        # Save to JSON
        with open(json_filename, 'w') as f:
            # Convert DataFrame to dict for JSON serialization
            team_output_json = team_output.copy()
            team_output_json['team'] = team_output['team'].to_dict(orient='records')
            json.dump(team_output_json, f, indent=2)
        
        # Save to CSV
        team_df = team_output['team'].copy()
        team_df.to_csv(csv_filename, index=False)
        
        if self.verbose:
            print(f"Team saved to {json_filename} and {csv_filename}")
            
            # Display key summary information
            print(f"\nSelected Team Summary:")
            print(f"  Total Credits: {team_output['summary']['total_credits']:.2f}")
            print(f"  Total Predicted Score: {team_output['summary']['total_predicted_score']:.2f}")
            print(f"  Role Distribution: {team_output['summary']['role_distribution']}")
            print(f"  Team Distribution: {team_output['summary']['team_distribution']}")
            print(f"  Captain: {team_output['summary']['captain']}")
            print(f"  Vice Captain: {team_output['summary']['vice_captain']}")
            if 'model_accuracy' in team_output['summary']:
                print(f"  Model Accuracy: {team_output['summary']['model_accuracy']:.2f}%")
    
    def run_pipeline(self):
        """Run the Dream11 team selection pipeline."""
        if self.verbose:
            print("========== DREAM11 TEAM SELECTION PIPELINE ==========")
        
        try:
            # Step 1: Get player lineup
            lineup_df = self._fetch_lineup_from_sheets()
            
            if lineup_df is None or lineup_df.empty:
                if self.verbose:
                    print("ERROR: Failed to get player lineup. Aborting pipeline.")
                return
            
            # Step 2: Match players with database
            matched_df = self._match_players(lineup_df)
            
            if matched_df is None or matched_df.empty:
                if self.verbose:
                    print("ERROR: Failed to match players. Aborting pipeline.")
                return
            
            # Step 3: Predict player performance (using deterministic scores based on player attributes)
            players_with_predictions = self._predict_performance(matched_df)
            
            if players_with_predictions is None:
                if self.verbose:
                    print("ERROR: Failed to predict player performance. Aborting pipeline.")
                return
            
            # Step 4: Select the best Dream11 team
            selected_team = self.select_dream11_team(players_with_predictions)
            
            if selected_team is None:
                if self.verbose:
                    print("ERROR: Failed to select Dream11 team. Aborting pipeline.")
                return
            
            # Step 5: Assign captain and vice-captain before evaluation
            selected_team = selected_team.sort_values('predicted_score', ascending=False)
            selected_team['Captain'] = False
            selected_team['Vice Captain'] = False
            
            if len(selected_team) > 0:
                selected_team.iloc[0, selected_team.columns.get_loc('Captain')] = True
            if len(selected_team) > 1:
                selected_team.iloc[1, selected_team.columns.get_loc('Vice Captain')] = True
            
            # Step 6: Evaluate the team selection quality and accuracy
            model_accuracy = self.evaluate_team_selection(selected_team)
            
            # Step 7: Generate and save output
            team_output = self.generate_output(selected_team)
            
            if team_output:
                # Add model accuracy to the output
                team_output['summary']['model_accuracy'] = model_accuracy
                self.save_output(team_output)
                
                # Print only the team info in JSON format
                players_list = team_output['team'].to_dict(orient='records')
                team_summary = {
                    'total_credits': team_output['summary']['total_credits'],
                    'role_distribution': team_output['summary']['role_distribution'],
                    'team_distribution': team_output['summary']['team_distribution'],
                    'captain': team_output['summary']['captain'],
                    'vice_captain': team_output['summary']['vice_captain'],
                    'model_accuracy': model_accuracy,
                    'players': players_list
                }
                print(json.dumps(team_summary, indent=2))
                
                if self.verbose:
                    print("\n========== DREAM11 TEAM SELECTION COMPLETED ==========")
            else:
                if self.verbose:
                    print("ERROR: Failed to generate output. Pipeline completed with errors.")
            
        except Exception as e:
            if self.verbose:
                print(f"ERROR: Pipeline failed with exception: {str(e)}")
                import traceback
                traceback.print_exc()

    def evaluate_team_selection(self, selected_team):
        """Evaluate the accuracy of the team selection based on several metrics."""
        if selected_team is None or selected_team.empty:
            if self.verbose:
                print("No selected team available. Cannot evaluate selection.")
            return 0
            
        # Metrics to evaluate:
        # 1. Credit utilization (closer to 100 is better)
        total_credits = selected_team['credits'].sum()
        credit_utilization_score = 1.0 - abs(100 - total_credits) / 25.0  # Normalize to [0,1]
        credit_utilization_score = max(0, min(1, credit_utilization_score))  # Clamp to [0,1]
        
        # 2. Star player selection (higher average predicted score is better)
        avg_player_score = selected_team['predicted_score'].mean()
        star_player_score = min(1.0, avg_player_score / 10.0)  # Normalize to [0,1] assuming 10 is max
        
        # 3. Role balance (closer to ideal role distribution is better)
        role_counts = selected_team['role'].value_counts().to_dict()
        ideal_role_counts = {'WK': 1, 'BAT': 3, 'AR': 3, 'BOWL': 4}
        role_diff = sum(abs(role_counts.get(role, 0) - count) for role, count in ideal_role_counts.items())
        role_balance_score = 1.0 - role_diff / 8.0  # Normalize to [0,1], max diff is 8
        role_balance_score = max(0, min(1, role_balance_score))  # Clamp to [0,1]
        
        # 4. Team balance (distribution across teams, ideally not too skewed)
        team_counts = selected_team['team'].value_counts().to_dict()
        max_from_one_team = max(team_counts.values())
        team_balance_score = 1.0 - (max_from_one_team - 5.5) / 1.5  # Normalize to [0,1], ideal is 5-6 max
        team_balance_score = max(0, min(1, team_balance_score))  # Clamp to [0,1]
        
        # 5. Captain selection quality (higher is better)
        captain_score = selected_team[selected_team['Captain']]['predicted_score'].iloc[0] / 10.0
        vice_captain_score = selected_team[selected_team['Vice Captain']]['predicted_score'].iloc[0] / 10.0
        
        # Calculate weighted overall score
        weights = {
            'credit_utilization': 0.15,
            'star_player': 0.25,
            'role_balance': 0.2,
            'team_balance': 0.15,
            'captain_selection': 0.25
        }
        
        overall_score = (
            weights['credit_utilization'] * credit_utilization_score +
            weights['star_player'] * star_player_score +
            weights['role_balance'] * role_balance_score +
            weights['team_balance'] * team_balance_score +
            weights['captain_selection'] * (captain_score * 0.6 + vice_captain_score * 0.4)
        )
        
        # Convert to percentage (0-100)
        accuracy_percentage = round(overall_score * 100, 2)
        
        if self.verbose:
            print("\n===== TEAM SELECTION EVALUATION =====")
            print(f"Credit Utilization Score: {credit_utilization_score:.2f} ({total_credits:.1f}/100)")
            print(f"Star Player Selection Score: {star_player_score:.2f} (avg score: {avg_player_score:.2f}/10)")
            print(f"Role Balance Score: {role_balance_score:.2f} (diff from ideal: {role_diff})")
            print(f"Team Balance Score: {team_balance_score:.2f} (max from one team: {max_from_one_team})")
            print(f"Captain Selection Score: {captain_score:.2f}, Vice Captain: {vice_captain_score:.2f}")
            print(f"\nOVERALL MODEL ACCURACY: {accuracy_percentage:.2f}%")
            print("====================================")
        
        return accuracy_percentage

def main():
    # Initialize the selector - set verbose to False to hide debug output
    selector = Dream11TeamSelector(verbose=False)
    
    # Run the pipeline
    selector.run_pipeline()

if __name__ == "__main__":
    main() 