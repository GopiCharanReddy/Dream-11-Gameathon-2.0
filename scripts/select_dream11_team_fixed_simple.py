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
import argparse

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

    def fetch_lineup(self, verbose=False):
        """Fetch lineup data from local file."""
        try:
            # Read lineup from local file
            if verbose:
                print("Reading lineup from local file: data/latest_lineup.csv")
            
            # Skip the first row (A,B,C,D,E,F) and use the second row as headers
            lineup_df = pd.read_csv('data/latest_lineup.csv', skiprows=[0])
            
            if verbose:
                print("Detected new format CSV")
            
            # Filter only playing players
            lineup_df = lineup_df[lineup_df['IsPlaying'] == 'PLAYING']
            
            # Standardize column names
            lineup_df = lineup_df.rename(columns={
                'Player Name': 'player_name',
                'Team': 'team',
                'Player Type': 'role',
                'Credits': 'credits',
                'lineupOrder': 'lineup_order'
            })
            
            # Convert credits to float
            lineup_df['credits'] = pd.to_numeric(lineup_df['credits'], errors='coerce')
            
            # Convert lineup_order to numeric
            lineup_df['lineup_order'] = pd.to_numeric(lineup_df['lineup_order'], errors='coerce')
            
            # Standardize role values
            role_mapping = {
                'BAT': 'BAT',
                'BOWL': 'BOWL',
                'AR': 'AR',
                'ALL': 'AR',
                'WK': 'WK'
            }
            lineup_df['role'] = lineup_df['role'].map(role_mapping).fillna('UNKNOWN')
            
            # Count wicketkeepers
            wk_count = len(lineup_df[lineup_df['role'] == 'WK'])
            if wk_count > 1:
                if verbose:
                    print(f"Found {wk_count} wicketkeepers in lineup, need to adjust")
                
                # Convert excess wicketkeepers to batsmen, keeping the highest credit WK
                wks = lineup_df[lineup_df['role'] == 'WK'].sort_values('credits', ascending=False)
                keep_wk = wks.iloc[0]['player_name']
                
                for _, wk in wks.iterrows():
                    if wk['player_name'] != keep_wk:
                        if verbose:
                            print(f"Converting {wk['player_name']} from WK to BAT (credits: {wk['credits']}, lineup_order: {wk['lineup_order']})")
                        lineup_df.loc[lineup_df['player_name'] == wk['player_name'], 'role'] = 'BAT'
            
            # Add role-specific features
            lineup_df['is_batsman'] = (lineup_df['role'].isin(['BAT', 'WK'])).astype(int)
            lineup_df['is_bowler'] = (lineup_df['role'] == 'BOWL').astype(int)
            lineup_df['is_allrounder'] = (lineup_df['role'] == 'AR').astype(int)
            lineup_df['is_wicketkeeper'] = (lineup_df['role'] == 'WK').astype(int)
            
            # Initialize form-related columns
            form_columns = ['recent_runs', 'recent_wickets', 'strike_rate', 'economy_rate', 
                          'boundary_rate', 'death_over_economy', 'batting_avg', 'bowling_avg']
            for col in form_columns:
                lineup_df[col] = 0.0
            
            if verbose:
                print(f"Successfully fetched lineup with {len(lineup_df)} players")
                print("\nRole distribution:")
                print(lineup_df['role'].value_counts())
                print("\nTeam distribution:")
                print(lineup_df['team'].value_counts())
            
            self.lineup_data = lineup_df
            return lineup_df
            
        except Exception as e:
            if verbose:
                print(f"Error fetching lineup: {str(e)}")
                import traceback
                traceback.print_exc()
            return None

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

    def _predict_performance(self, players_df, verbose=False):
        """
        Predict performance scores using a simple credit and role-based approach.
        """
        try:
            # Ensure required columns exist
            required_columns = ['player_name', 'team', 'role', 'credits']
            missing_columns = [col for col in required_columns if col not in players_df.columns]
            if missing_columns:
                if verbose:
                    print(f"Missing required columns: {missing_columns}")
                    print("Available columns:", players_df.columns.tolist())
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            # Create a copy of the players dataframe
            players_with_predictions = players_df.copy()
            
            # Initialize prediction columns
            players_with_predictions['predicted_score'] = 0.0
            players_with_predictions['is_captain'] = False
            players_with_predictions['is_vice_captain'] = False
            
            # Calculate base score from credits (assuming higher credit players are better)
            players_with_predictions['predicted_score'] = players_with_predictions['credits'] * 1.2
            
            # Add role-based bonus
            role_bonus = {
                'WK': 1.1,  # Wicketkeepers get a slight bonus
                'BAT': 1.0,
                'BOWL': 1.1,  # Bowlers get a slight bonus
                'AR': 1.2   # All-rounders get a bigger bonus
            }
            
            for role, bonus in role_bonus.items():
                mask = players_with_predictions['role'] == role
                players_with_predictions.loc[mask, 'predicted_score'] *= bonus
            
            # Add random variation to break ties (small random number between 0.95 and 1.05)
            np.random.seed(42)  # For reproducibility
            random_factors = np.random.uniform(0.95, 1.05, len(players_with_predictions))
            players_with_predictions['predicted_score'] *= random_factors
            
            if verbose:
                print("\nGenerated predictions for all players")
                print("\nTop 5 players by predicted score:")
                top_5 = players_with_predictions.nlargest(5, 'predicted_score')
                print(top_5[['player_name', 'team', 'role', 'credits', 'predicted_score']])
            
            return players_with_predictions
            
        except Exception as e:
            if verbose:
                print(f"Error predicting performance: {str(e)}")
                import traceback
                traceback.print_exc()
            return None

    def select_dream11_team(self) -> pd.DataFrame:
        """
        Select the best Dream11 team based on player performance predictions and constraints.
        """
        try:
            print("========== DREAM11 TEAM SELECTION PIPELINE ==========")
            
            # Fetch lineup data
            lineup_df = self.fetch_lineup(verbose=True)
            if lineup_df is None or len(lineup_df) == 0:
                print("ERROR: Failed to fetch lineup data. Aborting pipeline.")
                return None
            
            # Predict player performance
            players_with_predictions = self._predict_performance(lineup_df, verbose=True)
            if players_with_predictions is None or len(players_with_predictions) == 0:
                print("ERROR: Failed to predict player performance. Aborting pipeline.")
                return None
            
            # Define role limits for team selection
            role_limits = {
                'WK': {'min': 1, 'max': 1},
                'BAT': {'min': 3, 'max': 5},
                'AR': {'min': 1, 'max': 4},
                'BOWL': {'min': 3, 'max': 5}
            }
            
            # Select team using strategy
            selected_team = self._select_team_with_strategy(
                players_with_predictions,
                role_limits=role_limits,
                max_players_per_team=7,
                total_credits=100,
                verbose=True
            )
            
            if selected_team is None or len(selected_team) == 0:
                print("ERROR: Failed to select team. Aborting pipeline.")
                return None
            
            # Generate output
            output_success = self.generate_output(selected_team)
            if not output_success:
                print("ERROR: Failed to generate output. Pipeline completed with errors.")
            else:
                print("SUCCESS: Dream11 team selection completed successfully.")
            
            return selected_team
            
        except Exception as e:
            print(f"ERROR: An unexpected error occurred: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def _select_team_with_strategy(self, players_df, role_limits, max_players_per_team=7, total_credits=100, verbose=False):
        """
        Select team using a balanced strategy considering form, credits, and role distribution.
        """
        try:
            # Sort players by predicted score (descending)
            sorted_players = players_df.sort_values('predicted_score', ascending=False)
            
            # Initialize selected team
            selected_team = pd.DataFrame(columns=sorted_players.columns)
            total_credits_used = 0
            team_counts = {}
            role_counts = {'WK': 0, 'BAT': 0, 'AR': 0, 'BOWL': 0}
            selected_players = set()  # Keep track of selected players
            
            if verbose:
                print("\nSelecting team with strategy...")
                print(f"Available players: {len(sorted_players)}")
            
            # First, select one wicketkeeper
            wks = sorted_players[sorted_players['role'] == 'WK']
            if len(wks) == 0:
                if verbose:
                    print("ERROR: No wicketkeepers available")
                return None
            
            best_wk = wks.iloc[0]
            selected_team = pd.concat([selected_team, pd.DataFrame([best_wk])], ignore_index=True)
            total_credits_used += best_wk['credits']
            team_counts[best_wk['team']] = 1
            role_counts['WK'] = 1
            selected_players.add(best_wk['player_name'])
            
            if verbose:
                print(f"\nSelected WK: {best_wk['player_name']} (Credits: {best_wk['credits']}, Score: {best_wk['predicted_score']:.2f})")
            
            # Then select remaining players by role
            remaining_roles = ['BAT', 'AR', 'BOWL']
            for role in remaining_roles:
                role_players = sorted_players[
                    (sorted_players['role'] == role) & 
                    (~sorted_players['player_name'].isin(selected_players))
                ]
                
                min_required = role_limits[role]['min']
                max_allowed = role_limits[role]['max']
                
                # Select players for this role
                for _ in range(min_required):
                    for _, player in role_players.iterrows():
                        # Skip if player already selected
                        if player['player_name'] in selected_players:
                            continue
                            
                        # Check if adding this player would violate any constraints
                        if (total_credits_used + player['credits'] <= total_credits and
                            team_counts.get(player['team'], 0) < max_players_per_team):
                            
                            selected_team = pd.concat([selected_team, pd.DataFrame([player])], ignore_index=True)
                            total_credits_used += player['credits']
                            team_counts[player['team']] = team_counts.get(player['team'], 0) + 1
                            role_counts[role] += 1
                            selected_players.add(player['player_name'])
                            
                            if verbose:
                                print(f"Selected {role}: {player['player_name']} (Credits: {player['credits']}, Score: {player['predicted_score']:.2f})")
                            break
            
            # Fill remaining slots optimally
            while len(selected_team) < 11 and total_credits_used < total_credits:
                best_remaining = None
                best_score = 0
                
                for _, player in sorted_players.iterrows():
                    # Skip if player already selected
                    if player['player_name'] in selected_players:
                        continue
                        
                    # Check all constraints
                    if (total_credits_used + player['credits'] <= total_credits and
                        team_counts.get(player['team'], 0) < max_players_per_team and
                        role_counts[player['role']] < role_limits[player['role']]['max']):
                        
                        if player['predicted_score'] > best_score:
                            best_remaining = player
                            best_score = player['predicted_score']
                
                if best_remaining is None:
                    break
                    
                selected_team = pd.concat([selected_team, pd.DataFrame([best_remaining])], ignore_index=True)
                total_credits_used += best_remaining['credits']
                team_counts[best_remaining['team']] = team_counts.get(best_remaining['team'], 0) + 1
                role_counts[best_remaining['role']] += 1
                selected_players.add(best_remaining['player_name'])
                
                if verbose:
                    print(f"Selected {best_remaining['role']}: {best_remaining['player_name']} (Credits: {best_remaining['credits']}, Score: {best_remaining['predicted_score']:.2f})")
            
            if len(selected_team) < 11:
                if verbose:
                    print("ERROR: Could not select 11 players with given constraints")
                return None
            
            # Select captain and vice captain
            selected_team = selected_team.sort_values('predicted_score', ascending=False)
            selected_team.loc[selected_team.index[0], 'is_captain'] = True
            selected_team.loc[selected_team.index[1], 'is_vice_captain'] = True
            
            # Select substitute players
            substitute_team = self._select_substitute_players(
                sorted_players, 
                selected_players, 
                team_counts, 
                role_counts,
                total_credits_used,
                max_players_per_team,
                verbose
            )
            
            if verbose:
                print("\nFinal team selection:")
                print(f"Total credits used: {total_credits_used}")
                print("\nRole distribution:")
                for role, count in role_counts.items():
                    print(f"{role}: {count}")
                print("\nTeam distribution:")
                for team, count in team_counts.items():
                    print(f"{team}: {count}")
                print(f"\nCaptain: {selected_team.iloc[0]['player_name']} ({selected_team.iloc[0]['role']})")
                print(f"Vice Captain: {selected_team.iloc[1]['player_name']} ({selected_team.iloc[1]['role']})")
                
                if substitute_team is not None and len(substitute_team) > 0:
                    print("\nSubstitute Players:")
                    for _, sub in substitute_team.iterrows():
                        print(f"{sub['role']}: {sub['player_name']} (Credits: {sub['credits']}, Score: {sub['predicted_score']:.2f})")
            
            # Add substitute players to the selected team with a flag
            if substitute_team is not None and len(substitute_team) > 0:
                substitute_team['is_substitute'] = True
                selected_team['is_substitute'] = False
                selected_team = pd.concat([selected_team, substitute_team], ignore_index=True)
            
            return selected_team
            
        except Exception as e:
            if verbose:
                print(f"Error selecting team: {str(e)}")
                import traceback
                traceback.print_exc()
            return None

    def _select_substitute_players(self, sorted_players, selected_players, team_counts, role_counts, 
                                  total_credits_used, max_players_per_team, verbose=False):
        """
        Select substitute players for the Dream11 team.
        
        Substitute selection criteria:
        1. Players not in the main team
        2. Maintain role balance (2 AR, 1 BAT, 1 BOWL)
        3. Maintain team balance (but can be relaxed for last slot)
        4. Select players with high predicted scores
        5. No credit limit for substitutes
        """
        try:
            # Initialize substitute team
            substitute_team = pd.DataFrame(columns=sorted_players.columns)
            
            # Define required role distribution for substitutes (2 AR, 1 BAT, 1 BOWL)
            required_sub_roles = {'AR': 2, 'BAT': 1, 'BOWL': 1}
            
            # First, try to select a bowler since that's the most critical missing piece
            bowlers = sorted_players[
                (sorted_players['role'] == 'BOWL') & 
                (~sorted_players['player_name'].isin(selected_players))
            ]
            
            if len(bowlers) > 0:
                best_bowler = None
                best_score = 0
                
                for _, bowler in bowlers.iterrows():
                    # Check if adding this player would violate team constraints
                    if team_counts.get(bowler['team'], 0) < max_players_per_team:
                        if bowler['predicted_score'] > best_score:
                            best_bowler = bowler
                            best_score = bowler['predicted_score']
                
                if best_bowler is not None:
                    substitute_team = pd.concat([substitute_team, pd.DataFrame([best_bowler])], ignore_index=True)
                    selected_players.add(best_bowler['player_name'])
                    team_counts[best_bowler['team']] = team_counts.get(best_bowler['team'], 0) + 1
                    
                    if verbose:
                        print(f"Selected substitute BOWL: {best_bowler['player_name']} (Credits: {best_bowler['credits']}, Score: {best_bowler['predicted_score']:.2f})")
            
            # Then select remaining substitutes by role
            for role, count in required_sub_roles.items():
                if role == 'BOWL':  # Skip BOWL as we've already handled it
                    continue
                    
                # Find available players of this role
                role_players = sorted_players[
                    (sorted_players['role'] == role) & 
                    (~sorted_players['player_name'].isin(selected_players))
                ]
                
                if len(role_players) == 0:
                    if verbose:
                        print(f"WARNING: No {role} players available for substitutes")
                    continue
                
                # Select the required number of players for this role
                for _ in range(count):
                    best_player = None
                    best_score = 0
                    
                    for _, player in role_players.iterrows():
                        # Skip if player already selected
                        if player['player_name'] in selected_players:
                            continue
                            
                        # Check if adding this player would violate team constraints
                        # For BAT role, we'll relax the team constraints
                        if role == 'BAT' or team_counts.get(player['team'], 0) < max_players_per_team:
                            if player['predicted_score'] > best_score:
                                best_player = player
                                best_score = player['predicted_score']
                    
                    if best_player is not None:
                        substitute_team = pd.concat([substitute_team, pd.DataFrame([best_player])], ignore_index=True)
                        selected_players.add(best_player['player_name'])
                        team_counts[best_player['team']] = team_counts.get(best_player['team'], 0) + 1
                        
                        if verbose:
                            print(f"Selected substitute {role}: {best_player['player_name']} (Credits: {best_player['credits']}, Score: {best_player['predicted_score']:.2f})")
                    else:
                        if verbose:
                            print(f"WARNING: Could not find {role} player for substitutes")
            
            # If we still don't have 4 substitutes, try to find any available players
            if len(substitute_team) < 4:
                if verbose:
                    print(f"WARNING: Only found {len(substitute_team)} substitutes, need 4")
                
                # Try to fill remaining slots with any available players
                remaining_slots = 4 - len(substitute_team)
                for _ in range(remaining_slots):
                    best_remaining = None
                    best_score = 0
                    
                    for _, player in sorted_players.iterrows():
                        # Skip if player already selected
                        if player['player_name'] in selected_players:
                            continue
                            
                        # For remaining slots, we don't check team constraints
                        if player['predicted_score'] > best_score:
                            best_remaining = player
                            best_score = player['predicted_score']
                    
                    if best_remaining is None:
                        break
                        
                    substitute_team = pd.concat([substitute_team, pd.DataFrame([best_remaining])], ignore_index=True)
                    selected_players.add(best_remaining['player_name'])
                    team_counts[best_remaining['team']] = team_counts.get(best_remaining['team'], 0) + 1
                    
                    if verbose:
                        print(f"Selected additional substitute {best_remaining['role']}: {best_remaining['player_name']} (Credits: {best_remaining['credits']}, Score: {best_remaining['predicted_score']:.2f})")
            
            # If we still don't have 4 substitutes, log a warning
            if len(substitute_team) < 4:
                if verbose:
                    print(f"WARNING: Still only found {len(substitute_team)} substitutes after trying to fill slots")
            
            return substitute_team
            
        except Exception as e:
            if verbose:
                print(f"Error selecting substitute players: {str(e)}")
                import traceback
                traceback.print_exc()
            return None

    def _calculate_form_score(self, player):
        """Calculate form score based on recent performance."""
        form_score = 0.0
        
        # Batting form
        if player['role'] in ['BAT', 'WK', 'AR']:
            if player.get('recent_runs', 0) > 30:  # Good batting form
                form_score += 0.2
            if player.get('strike_rate', 0) > 150:  # Aggressive batting
                form_score += 0.1
            if player.get('boundary_rate', 0) > 0.2:  # Good boundary hitting
                form_score += 0.1
        
        # Bowling form
        if player['role'] in ['BOWL', 'AR']:
            if player.get('recent_wickets', 0) > 2:  # Good bowling form
                form_score += 0.2
            if player.get('economy_rate', 0) < 8:  # Economical bowling
                form_score += 0.1
            if player.get('death_over_economy', 0) < 9:  # Good death overs
                form_score += 0.1
        
        # All-rounder form
        if player['role'] == 'AR':
            if player.get('recent_runs', 0) > 20 and player.get('recent_wickets', 0) > 1:
                form_score += 0.2  # Bonus for all-round performance
        
        return min(form_score, 0.5)  # Cap form boost at 50%

    def generate_output(self, selected_team):
        """Generate formatted output for the selected Dream11 team."""
        if selected_team is None or selected_team.empty:
            print("No selected team available. Cannot generate output.")
            return None
        
        # Create DataFrame for output
        output_df = selected_team.copy()
        
        # Separate main team and substitutes
        main_team = output_df[~output_df['is_substitute']]
        substitutes = output_df[output_df['is_substitute']]
        
        # Calculate total credits for main team only
        total_credits = main_team['credits'].sum()
        
        # Calculate total predicted score for main team only
        total_predicted_score = main_team['predicted_score'].sum()
        
        # Get captain and vice-captain names
        captain_name = main_team[main_team['is_captain']]['player_name'].iloc[0] if not main_team[main_team['is_captain']].empty else None
        vice_captain_name = main_team[main_team['is_vice_captain']]['player_name'].iloc[0] if not main_team[main_team['is_vice_captain']].empty else None
        
        # Create summary
        team_summary = {
            'total_credits': float(total_credits),
            'total_predicted_score': float(total_predicted_score),
            'role_distribution': main_team['role'].value_counts().to_dict(),
            'team_distribution': main_team['team'].value_counts().to_dict(),
            'captain': captain_name,
            'vice_captain': vice_captain_name,
            'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S')
        }
        
        # Add substitute information if available
        if not substitutes.empty:
            team_summary['substitutes'] = {
                'count': len(substitutes),
                'role_distribution': substitutes['role'].value_counts().to_dict(),
                'team_distribution': substitutes['team'].value_counts().to_dict(),
                'total_credits': float(substitutes['credits'].sum()),
                'total_predicted_score': float(substitutes['predicted_score'].sum()),
                'players': substitutes.to_dict(orient='records')
            }
        
        return {'summary': team_summary, 'team': main_team}
    
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
            
            # Display substitute information if available
            if 'substitutes' in team_output['summary']:
                print(f"  Substitutes: {team_output['summary']['substitutes']['count']}")
                print(f"  Substitute Role Distribution: {team_output['summary']['substitutes']['role_distribution']}")
                print(f"  Substitute Team Distribution: {team_output['summary']['substitutes']['team_distribution']}")
            
            if 'model_accuracy' in team_output['summary']:
                print(f"  Model Accuracy: {team_output['summary']['model_accuracy']:.2f}%")
    
    def run_pipeline(self):
        """Run the Dream11 team selection pipeline."""
        if self.verbose:
            print("========== DREAM11 TEAM SELECTION PIPELINE ==========")
        
        try:
            # Step 1: Get player lineup
            lineup_df = self.fetch_lineup()
            
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
            selected_team = self.select_dream11_team()
            
            if selected_team is None:
                if self.verbose:
                    print("ERROR: Failed to select Dream11 team. Aborting pipeline.")
                return
            
            # Step 5: Generate and save output
            team_output = self.generate_output(selected_team)
            
            if team_output:
                self.save_output(team_output)
                
                # Print only the team info in JSON format
                players_list = team_output['team'].to_dict(orient='records')
                team_summary = {
                    'total_credits': team_output['summary']['total_credits'],
                    'role_distribution': team_output['summary']['role_distribution'],
                    'team_distribution': team_output['summary']['team_distribution'],
                    'captain': team_output['summary']['captain'],
                    'vice_captain': team_output['summary']['vice_captain'],
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
        
        # 6. Form-based evaluation (if form data is available)
        form_score = 0.5  # Default middle score
        if 'form_score' in selected_team.columns:
            avg_form_score = selected_team['form_score'].mean()
            form_score = min(1.0, avg_form_score * 2)  # Normalize to [0,1]
        
        # 7. Versatility score (players who can perform multiple roles)
        versatility_score = 0.5  # Default middle score
        if 'versatility_score' in selected_team.columns:
            avg_versatility = selected_team['versatility_score'].mean()
            versatility_score = min(1.0, avg_versatility)
        
        # 8. Team composition score (how well the team complements each other)
        composition_score = 0.5  # Default middle score
        
        # Check for good mix of players
        has_aggressive_batsman = False
        has_anchoring_batsman = False
        has_power_hitter = False
        has_spinner = False
        has_pacer = False
        
        for _, player in selected_team.iterrows():
            # Check for aggressive batsman (high strike rate)
            if player['role'] in ['BAT', 'WK'] and player.get('strike_rate', 0) > 150:
                has_aggressive_batsman = True
            
            # Check for anchoring batsman (good average)
            if player['role'] in ['BAT', 'WK'] and player.get('batting_avg', 0) > 30:
                has_anchoring_batsman = True
            
            # Check for power hitter (high boundary rate)
            if player['role'] in ['BAT', 'WK', 'AR'] and player.get('boundary_rate', 0) > 20:
                has_power_hitter = True
            
            # Check for spinner (good economy rate)
            if player['role'] in ['BOWL', 'AR'] and player.get('economy_rate', 0) < 8:
                has_spinner = True
            
            # Check for pacer (good bowling strike rate)
            if player['role'] in ['BOWL', 'AR'] and player.get('bowling_sr', 0) < 20:
                has_pacer = True
        
        # Calculate composition score based on player mix
        composition_score = (
            (1 if has_aggressive_batsman else 0) * 0.2 +
            (1 if has_anchoring_batsman else 0) * 0.2 +
            (1 if has_power_hitter else 0) * 0.2 +
            (1 if has_spinner else 0) * 0.2 +
            (1 if has_pacer else 0) * 0.2
        )
        
        # Calculate weighted overall score
        weights = {
            'credit_utilization': 0.10,
            'star_player': 0.15,
            'role_balance': 0.15,
            'team_balance': 0.10,
            'captain_selection': 0.15,
            'form': 0.10,
            'versatility': 0.10,
            'composition': 0.15
        }
        
        overall_score = (
            weights['credit_utilization'] * credit_utilization_score +
            weights['star_player'] * star_player_score +
            weights['role_balance'] * role_balance_score +
            weights['team_balance'] * team_balance_score +
            weights['captain_selection'] * (captain_score * 0.6 + vice_captain_score * 0.4) +
            weights['form'] * form_score +
            weights['versatility'] * versatility_score +
            weights['composition'] * composition_score
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
            print(f"Form Score: {form_score:.2f}")
            print(f"Versatility Score: {versatility_score:.2f}")
            print(f"Team Composition Score: {composition_score:.2f}")
            print(f"\nOVERALL MODEL ACCURACY: {accuracy_percentage:.2f}%")
            print("====================================")
            
            # Provide recommendations for improvement
            print("\nRECOMMENDATIONS FOR IMPROVEMENT:")
            if credit_utilization_score < 0.7:
                print(f"- Consider using more credits (currently using {total_credits:.1f}/100)")
            
            if role_balance_score < 0.7:
                print("- Adjust role distribution to be closer to ideal (1 WK, 3 BAT, 3 AR, 4 BOWL)")
                for role, count in role_counts.items():
                    ideal = ideal_role_counts.get(role, 0)
                    if count < ideal:
                        print(f"  - Need {ideal - count} more {role} player(s)")
                    elif count > ideal:
                        print(f"  - Have {count - ideal} too many {role} player(s)")
            
            if team_balance_score < 0.7:
                print(f"- Team distribution is too skewed (max {max_from_one_team} from one team)")
                for team, count in team_counts.items():
                    if count > 6:
                        print(f"  - Consider replacing {count - 6} player(s) from {team}")
            
            if form_score < 0.7:
                print("- Consider selecting more players in good form")
            
            if versatility_score < 0.7:
                print("- Consider selecting more versatile players who can perform multiple roles")
            
            if composition_score < 0.7:
                print("- Team composition could be improved:")
                if not has_aggressive_batsman:
                    print("  - Add an aggressive batsman (high strike rate)")
                if not has_anchoring_batsman:
                    print("  - Add an anchoring batsman (good average)")
                if not has_power_hitter:
                    print("  - Add a power hitter (high boundary rate)")
                if not has_spinner:
                    print("  - Add a spinner (good economy rate)")
                if not has_pacer:
                    print("  - Add a pacer (good bowling strike rate)")
        
        return accuracy_percentage

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Dream11 Team Selection')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    args = parser.parse_args()
    
    # Initialize the selector with the verbose flag from command line
    selector = Dream11TeamSelector(verbose=args.verbose)
    
    # Run the pipeline
    selector.run_pipeline()

if __name__ == "__main__":
    main() 