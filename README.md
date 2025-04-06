# AI Dream11 Team Selector

An AI-powered Dream11 team selection system that uses historical player data and machine learning to predict player performance and optimize team selection.

## Features

- Historical data processing and analysis
- Player performance prediction using machine learning
- Optimized team selection based on multiple factors
- Role and team balance consideration
- Captain and vice-captain selection
- Performance evaluation metrics

## Prerequisites

- Python 3.8 or higher
- Git
- pip (Python package installer)

## Installation Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/AI_Dream11.git
   cd AI_Dream11
   ```

2. **Create and activate a virtual environment**
   ```bash
   # On Windows
   python -m venv venv
   .\venv\Scripts\activate

   # On macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install required packages**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up Git configuration**
   ```bash
   git config user.email "your.email@example.com"
   git config user.name "Your Name"
   ```

## Project Structure

```
AI_Dream11/
├── data/
│   ├── historical_data/     # Historical match data
│   ├── player_data/         # Player information
│   └── squaddata_allteams.csv
├── models/
│   ├── __init__.py
│   └── performance_predictor.py
├── scripts/
│   ├── train_model.py
│   ├── train_performance_model.py
│   └── select_dream11_team_fixed_simple.py
├── output/                  # Generated team selections
├── requirements.txt
└── README.md
```

## Usage

1. **Process Historical Data**
   ```bash
   python scripts/train_model.py
   ```
   This will process all historical match data and create player metrics.

2. **Train the Performance Model**
   ```bash
   python scripts/train_performance_model.py
   ```
   This will train the machine learning model for performance prediction.

3. **Generate Dream11 Team**
   ```bash
   python scripts/select_dream11_team_fixed_simple.py
   ```
   This will generate an optimized Dream11 team based on the trained model.

## Output

The generated team will be saved in two formats:
- JSON file: `output/dream11_team_[timestamp].json`
- CSV file: `output/dream11_team_[timestamp].csv`

## Model Evaluation

The system provides several evaluation metrics:
- Credit Utilization Score
- Star Player Selection Score
- Role Balance Score
- Team Balance Score
- Captain Selection Score
- Overall Model Accuracy

## Contributing

1. Fork the repository
2. Create a new branch for your feature
3. Make your changes
4. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For any issues or questions, please open an issue in the GitHub repository. 