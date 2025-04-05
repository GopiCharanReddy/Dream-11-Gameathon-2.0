import pandas as pd
import numpy as np
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple
import logging
from datetime import datetime
import re

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IPLDataProcessor:
    def __init__(self, historical_data_dir: str, squad_data_file: str):
        self.historical_data_dir = historical_data_dir
        self.squad_data_file = squad_data_file
        self.current_players = self._load_squad_data()
        self.player_stats = {}
        self.recent_matches = {}  # Store recent match data for form calculation
        self.name_mapping = self._create_name_mapping()
        
    def _normalize_name(self, name: str) -> str:
        """Normalize player name by removing special characters and converting to lowercase."""
        # Remove special characters and convert to lowercase
        normalized = re.sub(r'[^a-zA-Z\s]', '', name.lower())
        # Remove extra spaces
        normalized = ' '.join(normalized.split())
        return normalized
    
    def _get_name_variations(self, name: str) -> List[str]:
        """Generate variations of a player name."""
        variations = []
        
        # Original name
        variations.append(name)
        
        # Normalized name
        normalized = self._normalize_name(name)
        variations.append(normalized)
        
        # Handle hyphenated names
        if '-' in name:
            variations.append(self._normalize_name(name.replace('-', ' ')))
        
        # Split into parts
        parts = name.split()
        if len(parts) > 1:
            # Last name only
            variations.append(self._normalize_name(parts[-1]))
            
            # First name only
            variations.append(self._normalize_name(parts[0]))
            
            # Generate initials format
            # For full names, create initial format (e.g., "Virat Kohli" -> "V Kohli")
            if len(parts[0]) > 2:  # Not already initials
                initial = parts[0][0]
                last_name = ' '.join(parts[1:])
                variations.append(f"{initial} {last_name}")
                variations.append(f"{initial}{last_name}")
                variations.append(self._normalize_name(f"{initial} {last_name}"))
                
                # For names with middle names, try different initial combinations
                if len(parts) > 2:
                    initials = ''.join(p[0] for p in parts[:-1])
                    variations.append(f"{initials} {parts[-1]}")
                    variations.append(f"{initials}{parts[-1]}")
        
        # Remove duplicates while preserving order
        seen = set()
        return [x for x in variations if not (x in seen or seen.add(x))]
    
    def _create_name_mapping(self) -> Dict[str, str]:
        """Create a mapping of normalized names to original names."""
        name_mapping = {}
        
        # First, create a mapping of full names to their variations
        full_name_variations = {}
        for player in self.current_players:
            variations = self._get_name_variations(player)
            
            # Create reverse mapping for players with known initials
            parts = player.split()
            if len(parts) > 1:
                # Map common known players' initials
                known_initials = {
                    "Mahendra Singh Dhoni": "MS Dhoni",
                    "Sachin Ramesh Tendulkar": "SR Tendulkar",
                    "Virat Kohli": "V Kohli",
                    "Rohit Sharma": "RG Sharma",
                    "Ravichandran Ashwin": "R Ashwin",
                    "Ravindra Jadeja": "RA Jadeja",
                    "Suryakumar Yadav": "SKY Yadav",
                    "Jasprit Bumrah": "JJ Bumrah",
                    "Bhuvneshwar Kumar": "B Kumar",
                    "Mohammed Siraj": "M Siraj",
                    "Hardik Pandya": "HH Pandya",
                    "Krunal Pandya": "KH Pandya",
                    "Lokesh Rahul": "KL Rahul",
                    "Rishabh Pant": "RR Pant"
                }
                
                if player in known_initials:
                    variations.append(known_initials[player])
                    # Also add normalized version
                    variations.append(self._normalize_name(known_initials[player]))
            
            for variation in variations:
                if variation in full_name_variations:
                    # If we have a conflict, prefer the longer name
                    if len(player) > len(full_name_variations[variation]):
                        full_name_variations[variation] = player
                else:
                    full_name_variations[variation] = player
        
        # Then create the final mapping
        for variation, full_name in full_name_variations.items():
            name_mapping[variation] = full_name
        
        return name_mapping
    
    def _match_player_name(self, name: str) -> str:
        """Match a player name from historical data to current squad."""
        if not name:
            return None
        
        # Try direct match first
        if name in self.name_mapping:
            return self.name_mapping[name]
        
        # Try variations
        normalized = self._normalize_name(name)
        if normalized in self.name_mapping:
            return self.name_mapping[normalized]
        
        # Hard-coded mappings for well-known players
        # This maps historical data initials to current squad names
        known_players = {
            'MS Dhoni': 'MS Dhoni',
            'V Kohli': 'Virat Kohli',
            'R Sharma': 'Rohit Sharma',
            'JJ Bumrah': 'Jasprit Bumrah',
            'KL Rahul': 'Lokesh Rahul',
            'R Jadeja': 'Ravindra Jadeja',
            'RA Jadeja': 'Ravindra Jadeja',
            'R Ashwin': 'Ravichandran Ashwin',
            'B Kumar': 'Bhuvneshwar Kumar',
            'SK Yadav': 'Suryakumar Yadav',
            'H Pandya': 'Hardik Pandya',
            'K Pandya': 'Krunal Pandya',
            'S Gill': 'Shubman Gill',
            'R Pant': 'Rishabh Pant',
            'DA Warner': 'David Warner',
            'M Siraj': 'Mohammed Siraj',
            'A Singh': 'Arshdeep Singh',
            'Y Chahal': 'Yuzvendra Chahal',
            'K Yadav': 'Kuldeep Yadav'
        }
        
        if name in known_players and known_players[name] in self.current_players:
            return known_players[name]
        
        # Try matching initials + last name
        parts = name.split()
        if len(parts) > 1:
            # If first part looks like initials (1-2 characters)
            if len(parts[0]) <= 2:
                last_name = ' '.join(parts[1:])
                
                # Look for exact matches with last name
                for player in self.current_players:
                    player_parts = player.split()
                    if player_parts and player_parts[-1] == last_name:
                        return player
                
                # Try normalized last name
                normalized_last = self._normalize_name(last_name)
                for player in self.current_players:
                    player_parts = player.split()
                    if player_parts and self._normalize_name(player_parts[-1]) == normalized_last:
                        return player
        
        return None
    
    def _load_squad_data(self) -> Dict[str, Dict]:
        """Load current IPL 2025 players from the squad data file."""
        df = pd.read_csv(self.squad_data_file)
        # Create a mapping of player name to their details
        players = {}
        for _, row in df.iterrows():
            players[row['Player Name']] = {
                'team': row['Team'],
                'role': row['Player Type'],
                'credits': row['Credits'],
                'is_all_rounder': row['Player Type'] == 'ALL',
                'is_wicketkeeper': row['Player Type'] == 'WK'
            }
        return players
    
    def _process_match_data(self, match_file: str) -> None:
        """Process a single match JSON file and update player statistics."""
        try:
            with open(match_file, 'r') as f:
                match_data = json.load(f)
            
            # Extract match metadata with better error handling
            match_date = None
            try:
                date_str = match_data.get('date', '')
                if date_str:
                    match_date = datetime.strptime(date_str, '%Y-%m-%d')
            except (ValueError, TypeError) as e:
                # Try alternate date format from info.dates
                try:
                    dates = match_data.get('info', {}).get('dates', [])
                    if dates:
                        match_date = datetime.strptime(dates[0], '%Y-%m-%d')
                except (ValueError, TypeError, IndexError) as e:
                    logger.warning(f"Invalid date format in {match_file}")
                    match_date = datetime.now()
            
            # Create a direct player mapping from this match's registry if available
            player_registry = {}
            registry_data = match_data.get('info', {}).get('registry', {}).get('people', {})
            if registry_data:
                # Print a few registry entries for debugging
                for i, (player_id, player_info) in enumerate(registry_data.items()):
                    player_registry[player_id] = player_info
                    if i < 5:  # Print first 5 entries for debugging
                        logger.debug(f"Registry: {player_id} -> {player_info}")
            
            # Process player lists for each team
            players_data = match_data.get('info', {}).get('players', {})
            # Create lookup for player IDs to full names for this match
            for team, players in players_data.items():
                for player in players:
                    # Try to match the player to current squad
                    matched_player = self._match_player_name(player)
                    if matched_player:
                        # If we found a match, initialize player stats if needed
                        if matched_player not in self.player_stats:
                            self._initialize_player_stats(matched_player)
                            self.player_stats[matched_player]['matches'] += 1
            
            # Extract innings data
            innings = match_data.get('innings', [])
            
            for inning in innings:
                team = inning.get('team')
                deliveries = inning.get('deliveries', [])
                
                # Track death overs (16-20)
                death_overs = [16, 17, 18, 19, 20]
                
                for delivery in deliveries:
                    for ball_num, ball_data in delivery.items():
                        try:
                            over = float(ball_num)
                            is_death_over = int(over) in death_overs
                            
                            # Process batting stats
                            batter_name = ball_data.get('batter')
                            batter = self._match_player_name(batter_name)
                            if batter in self.current_players and batter in self.player_stats:
                                # Update batting stats
                                self.player_stats[batter]['runs'] += ball_data.get('runs', {}).get('batter', 0)
                                self.player_stats[batter]['balls_faced'] += 1
                                
                                # Track boundaries
                                runs = ball_data.get('runs', {}).get('batter', 0)
                                if runs == 4:
                                    self.player_stats[batter]['fours'] += 1
                                elif runs == 6:
                                    self.player_stats[batter]['sixes'] += 1
                                
                                # Track dot balls
                                if runs == 0:
                                    self.player_stats[batter]['dot_balls_faced'] += 1
                                
                                # Track death over performance
                                if is_death_over:
                                    self.player_stats[batter]['death_over_runs'] += runs
                                    self.player_stats[batter]['death_over_balls'] += 1
                                
                                # Check for dismissal
                                if 'wicket' in ball_data:
                                    self.player_stats[batter]['dismissals'] += 1
                                    self.player_stats[batter]['innings'] += 1
                            
                            # Process bowling stats
                            bowler_name = ball_data.get('bowler')
                            bowler = self._match_player_name(bowler_name)
                            if bowler in self.current_players and bowler in self.player_stats:
                                # Update bowling stats
                                runs_conceded = ball_data.get('runs', {}).get('total', 0)
                                self.player_stats[bowler]['runs_conceded'] += runs_conceded
                                self.player_stats[bowler]['balls_bowled'] += 1
                                
                                # Track dot balls
                                if runs_conceded == 0:
                                    self.player_stats[bowler]['dot_balls_bowled'] += 1
                                
                                # Track death over performance
                                if is_death_over:
                                    self.player_stats[bowler]['death_over_runs_conceded'] += runs_conceded
                                    self.player_stats[bowler]['death_over_balls'] += 1
                                
                                if 'wicket' in ball_data:
                                    self.player_stats[bowler]['wickets'] += 1
                                    self.player_stats[bowler]['innings'] += 1
                                
                                # Track maidens
                                if int(ball_num.split('.')[1]) == 0:  # Start of over
                                    self.player_stats[bowler]['maidens'] += 1
                            
                            # Process wicket-keeping stats
                            if 'wicket' in ball_data:
                                keeper_names = ball_data.get('wicket', {}).get('fielders', [])
                                if keeper_names:
                                    keeper = self._match_player_name(keeper_names[0])
                                    if keeper in self.current_players and keeper in self.player_stats:
                                        self.player_stats[keeper]['dismissals'] += 1
                                        if ball_data['wicket'].get('kind') == 'stumped':
                                            self.player_stats[keeper]['stumpings'] += 1
                                        else:
                                            self.player_stats[keeper]['catches'] += 1
                        except Exception as e:
                            logger.warning(f"Error processing ball {ball_num} in {match_file}: {str(e)}")
                            continue
                
                # Store recent match data for form calculation
                for player in self.current_players:
                    if player in self.player_stats:
                        if player not in self.recent_matches:
                            self.recent_matches[player] = []
                        self.recent_matches[player].append({
                            'date': match_date,
                            'runs': self.player_stats[player].get('runs', 0),
                            'wickets': self.player_stats[player].get('wickets', 0),
                            'dismissals': self.player_stats[player].get('dismissals', 0)
                        })
                                
        except Exception as e:
            logger.error(f"Error processing match file {match_file}: {str(e)}")
    
    def _initialize_player_stats(self, player: str) -> None:
        """Initialize statistics for a new player."""
        if player not in self.player_stats:
            self.player_stats[player] = {
                # Basic info
                'team': self.current_players[player]['team'],
                'role': self.current_players[player]['role'],
                'credits': self.current_players[player]['credits'],
                'is_all_rounder': self.current_players[player]['is_all_rounder'],
                'is_wicketkeeper': self.current_players[player]['is_wicketkeeper'],
                
                # Batting stats
                'runs': 0,
                'balls_faced': 0,
                'fours': 0,
                'sixes': 0,
                'dismissals': 0,
                'innings': 0,
                'dot_balls_faced': 0,
                'death_over_runs': 0,
                'death_over_balls': 0,
                
                # Bowling stats
                'runs_conceded': 0,
                'balls_bowled': 0,
                'wickets': 0,
                'maidens': 0,
                'dot_balls_bowled': 0,
                'death_over_runs_conceded': 0,
                'death_over_balls': 0,
                
                # Wicket-keeping stats
                'catches': 0,
                'stumpings': 0,
                
                # Match stats
                'matches': 0
            }
    
    def process_all_matches(self) -> None:
        """Process all historical match files."""
        match_files = [f for f in os.listdir(self.historical_data_dir) if f.endswith('.json')]
        total_matches = len(match_files)
        
        logger.info(f"Processing {total_matches} historical matches...")
        logger.info(f"Looking for {len(self.current_players)} current players...")
        
        for i, match_file in enumerate(match_files, 1):
            if i % 10 == 0:
                logger.info(f"Processed {i}/{total_matches} matches")
            self._process_match_data(os.path.join(self.historical_data_dir, match_file))
        
        # Log stats about processed data
        players_with_data = len([p for p in self.player_stats.values() if p['matches'] > 0])
        logger.info(f"Found historical data for {players_with_data} players")
        
        if players_with_data == 0:
            logger.warning("No players matched between historical data and current squad!")
        logger.info(f"Found historical data for {len(self.player_stats)} players")
    
    def calculate_player_metrics(self) -> pd.DataFrame:
        """Calculate final performance metrics for each player."""
        metrics = []
        
        for player, stats in self.player_stats.items():
            # Calculate batting metrics
            batting_avg = stats['runs'] / max(1, stats['dismissals'])
            strike_rate = (stats['runs'] / max(1, stats['balls_faced'])) * 100
            boundary_rate = (stats['fours'] + stats['sixes']) / max(1, stats['balls_faced'])
            dot_ball_rate = stats['dot_balls_faced'] / max(1, stats['balls_faced'])
            death_over_strike_rate = (stats['death_over_runs'] / max(1, stats['death_over_balls'])) * 100
            
            # Calculate bowling metrics
            economy_rate = (stats['runs_conceded'] / max(1, stats['balls_bowled'])) * 6
            bowling_avg = stats['runs_conceded'] / max(1, stats['wickets'])
            bowling_sr = stats['balls_bowled'] / max(1, stats['wickets'])
            dot_ball_percentage = (stats['dot_balls_bowled'] / max(1, stats['balls_bowled'])) * 100
            death_over_economy = (stats['death_over_runs_conceded'] / max(1, stats['death_over_balls'])) * 6
            
            # Calculate wicket-keeping metrics
            dismissals_per_match = stats['dismissals'] / max(1, stats['innings'])
            catches_per_match = stats['catches'] / max(1, stats['innings'])
            stumpings_per_match = stats['stumpings'] / max(1, stats['innings'])
            
            # Calculate form metrics (last 5 matches)
            recent_runs = 0
            recent_wickets = 0
            
            # Safely handle recent matches with proper date sorting
            recent_matches = self.recent_matches.get(player, [])
            if recent_matches:
                # Filter out matches with None dates
                valid_matches = [m for m in recent_matches if m['date'] is not None]
                if valid_matches:
                    # Sort by date and take last 5
                    sorted_matches = sorted(valid_matches, key=lambda x: x['date'])[-5:]
                    recent_runs = sum(m['runs'] for m in sorted_matches) / max(1, len(sorted_matches))
                    recent_wickets = sum(m['wickets'] for m in sorted_matches) / max(1, len(sorted_matches))
            
            # Calculate composite scores
            batting_impact = (batting_avg * 0.4 + strike_rate * 0.3 + boundary_rate * 0.3)
            bowling_impact = (1/economy_rate * 0.4 + 1/bowling_avg * 0.3 + 1/bowling_sr * 0.3) if stats['balls_bowled'] > 0 else 0
            all_rounder_score = (batting_impact + bowling_impact) if stats['is_all_rounder'] else 0
            
            metrics.append({
                'player_name': player,
                'team': stats['team'],
                'role': stats['role'],
                'credits': stats['credits'],
                'is_all_rounder': stats['is_all_rounder'],
                'is_wicketkeeper': stats['is_wicketkeeper'],
                
                # Batting metrics
                'batting_avg': batting_avg,
                'strike_rate': strike_rate,
                'boundary_rate': boundary_rate,
                'dot_ball_rate': dot_ball_rate,
                'death_over_strike_rate': death_over_strike_rate,
                
                # Bowling metrics
                'economy_rate': economy_rate,
                'bowling_avg': bowling_avg,
                'bowling_sr': bowling_sr,
                'dot_ball_percentage': dot_ball_percentage,
                'death_over_economy': death_over_economy,
                
                # Wicket-keeping metrics
                'dismissals_per_match': dismissals_per_match,
                'catches_per_match': catches_per_match,
                'stumpings_per_match': stumpings_per_match,
                
                # Form metrics
                'recent_runs': recent_runs,
                'recent_wickets': recent_wickets,
                
                # Composite scores
                'batting_impact': batting_impact,
                'bowling_impact': bowling_impact,
                'all_rounder_score': all_rounder_score,
                
                # Match context
                'total_innings': stats['innings'],
                'matches_played': stats.get('matches', 0)
            })
        
        return pd.DataFrame(metrics)

def main():
    # Initialize the data processor
    processor = IPLDataProcessor(
        historical_data_dir='data/historical_data',
        squad_data_file='backup/data/squaddata_allteams.csv'
    )
    
    # Process all historical matches
    processor.process_all_matches()
    
    # Calculate and save player metrics
    player_metrics = processor.calculate_player_metrics()
    
    # Create output directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Save the metrics
    player_metrics.to_csv('data/player_historical_metrics.csv', index=False)
    logger.info(f"Player metrics saved to data/player_historical_metrics.csv")
    logger.info(f"Processed data for {len(player_metrics)} players")

if __name__ == "__main__":
    main() 