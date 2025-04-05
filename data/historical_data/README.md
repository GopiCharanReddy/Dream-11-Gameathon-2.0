# AI Dream11 - Fantasy Cricket Team Predictor

An advanced machine learning system that predicts optimal Dream11 fantasy cricket teams based on historical player performance, match conditions, and other relevant factors.

## Project Overview

This project uses historical cricket data and machine learning to predict the best possible fantasy team for upcoming cricket matches. The system analyzes player statistics, lineup order, and other factors to optimize team selection within Dream11's constraints.

## Features

- **Latest Lineup Data**: Uses latest match lineup data for team selection
- **Flexible Team Selection**: Selects optimal players within Dream11 credit constraints
- **Captain Selection**: Identifies the best Captain and Vice-Captain choices based on predicted scores
- **Role and Team Distribution**: Ensures balanced team composition

## Project Structure

```
AI_Dream11/
│── data/                  
│   ├── latest_lineup.csv          # Latest match lineup data  
│   ├── processed_data.csv         # Processed historical player data  
│── scripts/                        
│   ├── select_dream11_team_fixed_simple.py  # Main script for team selection  
│── models/                         
│   ├── player_performance_model_XXXXXXXX_XXXXXX.h5  # Latest performance prediction model  
│   ├── player_scaler_XXXXXXXX_XXXXXX.pkl           # Latest scaler model 
│   ├── feature_columns_XXXXXXXX_XXXXXX.json        # Latest feature columns 
│── output/                         
│   ├── dream11_team_XXXXXXXX_XXXXXX.json           # Selected team details in JSON format
│   ├── dream11_team_XXXXXXXX_XXXXXX.csv            # Selected team details in CSV format
│── requirements.txt                # Python dependencies  
│── README.md                       # Project documentation  
```

## Getting Started

### Prerequisites

- Python 3.7+
- Required packages (see requirements.txt)

### Installation

1. Clone the repository:
```
git clone https://github.com/yourusername/AI_Dream11.git
cd AI_Dream11
```

2. Install dependencies:
```
pip install -r requirements.txt
```

### Usage

1. **Update Lineup Data**:
   - Update the `data/latest_lineup.csv` file with the latest match lineup

2. **Team Prediction**:
```
python scripts/select_dream11_team_fixed_simple.py
```

3. **View Results**:
   - Check the `output/` directory for the generated team in JSON and CSV formats

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Dream11 for the fantasy cricket platform 