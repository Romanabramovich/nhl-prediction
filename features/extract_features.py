# features/extract_features.py
import pandas as pd
from typing import Dict, List

def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract comprehensive features from feature_json column.
       
    Args:
        df: DataFrame with 'feature_json' column containing feature dictionaries
        
    Returns:
        DataFrame with extracted features as columns
    """
    features = []
    
    for _, row in df.iterrows():
        feature_dict = row['feature_json']
        
        # extract comprehensive feature set
        feature_row = {
            # basic game info
            'home_team_id': feature_dict.get('home_team_id'),
            'away_team_id': feature_dict.get('away_team_id'),
            
            # rest and back-to-back
            'home_rest_days': feature_dict.get('home_rest_days'),
            'away_rest_days': feature_dict.get('away_rest_days'),
            'rest_diff': feature_dict.get('rest_diff'),
            'b2b_home': feature_dict.get('b2b_home'),
            'b2b_away': feature_dict.get('b2b_away'),
            
            # rolling form stats
            'home_roll_pts': feature_dict.get('home_roll_pts'),
            'away_roll_pts': feature_dict.get('away_roll_pts'),
            'home_roll_w': feature_dict.get('home_roll_w'),
            'away_roll_w': feature_dict.get('away_roll_w'),
            'home_roll_l': feature_dict.get('home_roll_l'),
            'away_roll_l': feature_dict.get('away_roll_l'),
            'home_roll_ot_l': feature_dict.get('home_roll_ot_l'),
            'away_roll_ot_l': feature_dict.get('away_roll_ot_l'),
            'home_roll_gf': feature_dict.get('home_roll_gf'),
            'away_roll_gf': feature_dict.get('away_roll_gf'),
            'home_roll_ga': feature_dict.get('home_roll_ga'),
            'away_roll_ga': feature_dict.get('away_roll_ga'),
            'home_roll_goal_diff': feature_dict.get('home_roll_goal_diff'),
            'away_roll_goal_diff': feature_dict.get('away_roll_goal_diff'),
            'home_roll_goal_diff_pct': feature_dict.get('home_roll_goal_diff_pct'),
            'away_roll_goal_diff_pct': feature_dict.get('away_roll_goal_diff_pct'),
            'home_roll_reg_wins': feature_dict.get('home_roll_reg_wins'),
            'away_roll_reg_wins': feature_dict.get('away_roll_reg_wins'),
            'home_roll_reg_ot_wins': feature_dict.get('home_roll_reg_ot_wins'),
            'away_roll_reg_ot_wins': feature_dict.get('away_roll_reg_ot_wins'),
            'home_roll_league_seq': feature_dict.get('home_roll_league_seq'),
            'away_roll_league_seq': feature_dict.get('away_roll_league_seq'),
            'home_roll_conf_seq': feature_dict.get('home_roll_conf_seq'),
            'away_roll_conf_seq': feature_dict.get('away_roll_conf_seq'),
            'home_roll_div_seq': feature_dict.get('home_roll_div_seq'),
            'away_roll_div_seq': feature_dict.get('away_roll_div_seq'),
            
            # contextual stats
            'home_league_seq': feature_dict.get('home_league_seq'),
            'away_league_seq': feature_dict.get('away_league_seq'),
            'home_conf_seq': feature_dict.get('home_conf_seq'),
            'away_conf_seq': feature_dict.get('away_conf_seq'),
            'home_div_seq': feature_dict.get('home_div_seq'),
            'away_div_seq': feature_dict.get('away_div_seq'),
            'home_league_home_seq': feature_dict.get('home_league_home_seq'),
            'away_league_home_seq': feature_dict.get('away_league_home_seq'),
            'home_league_road_seq': feature_dict.get('home_league_road_seq'),
            'away_league_road_seq': feature_dict.get('away_league_road_seq'),
            'home_conf_home_seq': feature_dict.get('home_conf_home_seq'),
            'away_conf_home_seq': feature_dict.get('away_conf_home_seq'),
            'home_conf_road_seq': feature_dict.get('home_conf_road_seq'),
            'away_conf_road_seq': feature_dict.get('away_conf_road_seq'),
            'home_div_home_seq': feature_dict.get('home_div_home_seq'),
            'away_div_home_seq': feature_dict.get('away_div_home_seq'),
            'home_div_road_seq': feature_dict.get('home_div_road_seq'),
            'away_div_road_seq': feature_dict.get('away_div_road_seq'),
            
            # streak information
            'home_streak_code': feature_dict.get('home_streak_code'),
            'away_streak_code': feature_dict.get('away_streak_code'),
            'home_streak_count': feature_dict.get('home_streak_count'),
            'away_streak_count': feature_dict.get('away_streak_count'),
            
            # home/road situational stats
            'home_home_wins': feature_dict.get('home_home_wins'),
            'away_home_wins': feature_dict.get('away_home_wins'),
            'home_home_losses': feature_dict.get('home_home_losses'),
            'away_home_losses': feature_dict.get('away_home_losses'),
            'home_home_ot_losses': feature_dict.get('home_home_ot_losses'),
            'away_home_ot_losses': feature_dict.get('away_home_ot_losses'),
            'home_home_points': feature_dict.get('home_home_points'),
            'away_home_points': feature_dict.get('away_home_points'),
            'home_home_gf': feature_dict.get('home_home_gf'),
            'away_home_gf': feature_dict.get('away_home_gf'),
            'home_home_ga': feature_dict.get('home_home_ga'),
            'away_home_ga': feature_dict.get('away_home_ga'),
            'home_home_goal_diff': feature_dict.get('home_home_goal_diff'),
            'away_home_goal_diff': feature_dict.get('away_home_goal_diff'),
            'home_home_reg_wins': feature_dict.get('home_home_reg_wins'),
            'away_home_reg_wins': feature_dict.get('away_home_reg_wins'),
            'home_home_reg_ot_wins': feature_dict.get('home_home_reg_ot_wins'),
            'away_home_reg_ot_wins': feature_dict.get('away_home_reg_ot_wins'),
            
            'home_road_wins': feature_dict.get('home_road_wins'),
            'away_road_wins': feature_dict.get('away_road_wins'),
            'home_road_losses': feature_dict.get('home_road_losses'),
            'away_road_losses': feature_dict.get('away_road_losses'),
            'home_road_ot_losses': feature_dict.get('home_road_ot_losses'),
            'away_road_ot_losses': feature_dict.get('away_road_ot_losses'),
            'home_road_points': feature_dict.get('home_road_points'),
            'away_road_points': feature_dict.get('away_road_points'),
            'home_road_gf': feature_dict.get('home_road_gf'),
            'away_road_gf': feature_dict.get('away_road_gf'),
            'home_road_ga': feature_dict.get('home_road_ga'),
            'away_road_ga': feature_dict.get('away_road_ga'),
            'home_road_goal_diff': feature_dict.get('home_road_goal_diff'),
            'away_road_goal_diff': feature_dict.get('away_road_goal_diff'),
            'home_road_reg_wins': feature_dict.get('home_road_reg_wins'),
            'away_road_reg_wins': feature_dict.get('away_road_reg_wins'),
            'home_road_reg_ot_wins': feature_dict.get('home_road_reg_ot_wins'),
            'away_road_reg_ot_wins': feature_dict.get('away_road_reg_ot_wins'),
            
            # overall season stats
            'home_games_played': feature_dict.get('home_games_played'),
            'away_games_played': feature_dict.get('away_games_played'),
            'home_points': feature_dict.get('home_points'),
            'away_points': feature_dict.get('away_points'),
            'home_win_pct': feature_dict.get('home_win_pct'),
            'away_win_pct': feature_dict.get('away_win_pct'),
            'home_goal_diff_pct': feature_dict.get('home_goal_diff_pct'),
            'away_goal_diff_pct': feature_dict.get('away_goal_diff_pct'),
        }
        
        features.append(feature_row)
    
    feature_df = pd.DataFrame(features)
    
    # convert categorical variables to numerical
    feature_df = _encode_categorical_features(feature_df)
    
    # ensure proper data types for all features
    feature_df = _convert_data_types(feature_df)
    
    return feature_df

def _encode_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert categorical features to numerical representations.
    
    Args:
        df: DataFrame with categorical features
        
    Returns:
        DataFrame with encoded categorical features
    """
    df = df.copy()
    
    # combine streak code and count into single numerical feature
    # W5 = +5, L4 = -4, OTL3 = -1.5, etc.
    streak_code_mapping = {'W': 1, 'L': -1, 'OTL': -0.5, 'SOL': -0.5, 'SOW': 0.5}
    
    if 'home_streak_code' in df.columns:
        # create combined streak feature: code_value * count
        home_code_values = df['home_streak_code'].map(streak_code_mapping).fillna(0).astype(float)
        home_counts = df['home_streak_count'].fillna(0).astype(float)
        df['home_streak'] = home_code_values * home_counts
        
        away_code_values = df['away_streak_code'].map(streak_code_mapping).fillna(0).astype(float)
        away_counts = df['away_streak_count'].fillna(0).astype(float)
        df['away_streak'] = away_code_values * away_counts
        
        # drop original streak columns
        df = df.drop(['home_streak_code', 'away_streak_code', 'home_streak_count', 'away_streak_count'], 
                     axis=1, errors='ignore')
    
    return df

def _convert_data_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert all features to appropriate numeric data types.
    
    Args:
        df: DataFrame with mixed data types
        
    Returns:
        DataFrame with proper numeric data types
    """
    df = df.copy()
    
    # Define which columns should be integers vs floats
    integer_columns = [
        'home_team_id', 'away_team_id', 'home_rest_days', 'away_rest_days', 'rest_diff',
        'b2b_home', 'b2b_away', 'home_roll_w', 'away_roll_w', 'home_roll_l', 'away_roll_l',
        'home_roll_ot_l', 'away_roll_ot_l', 'home_roll_gf', 'away_roll_gf', 'home_roll_ga', 'away_roll_ga',
        'home_roll_reg_wins', 'away_roll_reg_wins', 'home_roll_reg_ot_wins', 'away_roll_reg_ot_wins',
        'home_roll_league_seq', 'away_roll_league_seq', 'home_roll_conf_seq', 'away_roll_conf_seq',
        'home_roll_div_seq', 'away_roll_div_seq', 'home_league_seq', 'away_league_seq',
        'home_conf_seq', 'away_conf_seq', 'home_div_seq', 'away_div_seq',
        'home_league_home_seq', 'away_league_home_seq', 'home_league_road_seq', 'away_league_road_seq',
        'home_conf_home_seq', 'away_conf_home_seq', 'home_conf_road_seq', 'away_conf_road_seq',
        'home_div_home_seq', 'away_div_home_seq', 'home_div_road_seq', 'away_div_road_seq',
        'home_home_wins', 'away_home_wins', 'home_home_losses', 'away_home_losses',
        'home_home_ot_losses', 'away_home_ot_losses', 'home_home_gf', 'away_home_gf',
        'home_home_ga', 'away_home_ga', 'home_home_reg_wins', 'away_home_reg_wins',
        'home_home_reg_ot_wins', 'away_home_reg_ot_wins', 'home_road_wins', 'away_road_wins',
        'home_road_losses', 'away_road_losses', 'home_road_ot_losses', 'away_road_ot_losses',
        'home_road_gf', 'away_road_gf', 'home_road_ga', 'away_road_ga',
        'home_road_reg_wins', 'away_road_reg_wins', 'home_road_reg_ot_wins', 'away_road_reg_ot_wins',
        'home_games_played', 'away_games_played'
    ]
    
    # Convert all columns to numeric, handling missing values
    for col in df.columns:
        if col in integer_columns:
            # Convert to int, filling NaN with 0
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype('int64')
        else:
            # Convert to float, filling NaN with 0
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype('float64')
    
    return df

def get_feature_groups() -> Dict[str, List[str]]:
    """
    Get feature groups for analysis and feature selection.
    
    Returns:
        Dictionary mapping group names to lists of feature names
    """
    return {
        'basic_info': ['home_team_id', 'away_team_id'],
        'rest_back_to_back': ['home_rest_days', 'away_rest_days', 'rest_diff', 'b2b_home', 'b2b_away'],
        'rolling_form': [
            'home_roll_pts', 'away_roll_pts', 'home_roll_w', 'away_roll_w', 'home_roll_l', 'away_roll_l',
            'home_roll_ot_l', 'away_roll_ot_l', 'home_roll_gf', 'away_roll_gf', 'home_roll_ga', 'away_roll_ga',
            'home_roll_goal_diff', 'away_roll_goal_diff', 'home_roll_goal_diff_pct', 'away_roll_goal_diff_pct',
            'home_roll_reg_wins', 'away_roll_reg_wins', 'home_roll_reg_ot_wins', 'away_roll_reg_ot_wins'
        ],
        'rolling_sequences': [
            'home_roll_league_seq', 'away_roll_league_seq', 'home_roll_conf_seq', 'away_roll_conf_seq',
            'home_roll_div_seq', 'away_roll_div_seq'
        ],
        'current_standings': [
            'home_league_seq', 'away_league_seq', 'home_conf_seq', 'away_conf_seq', 'home_div_seq', 'away_div_seq'
        ],
        'home_road_sequences': [
            'home_league_home_seq', 'away_league_home_seq', 'home_league_road_seq', 'away_league_road_seq',
            'home_conf_home_seq', 'away_conf_home_seq', 'home_conf_road_seq', 'away_conf_road_seq',
            'home_div_home_seq', 'away_div_home_seq', 'home_div_road_seq', 'away_div_road_seq'
        ],
        'streaks': ['home_streak', 'away_streak'],
        'home_stats': [
            'home_home_wins', 'away_home_wins', 'home_home_losses', 'away_home_losses',
            'home_home_ot_losses', 'away_home_ot_losses', 'home_home_points', 'away_home_points',
            'home_home_gf', 'away_home_gf', 'home_home_ga', 'away_home_ga',
            'home_home_goal_diff', 'away_home_goal_diff', 'home_home_reg_wins', 'away_home_reg_wins',
            'home_home_reg_ot_wins', 'away_home_reg_ot_wins'
        ],
        'road_stats': [
            'home_road_wins', 'away_road_wins', 'home_road_losses', 'away_road_losses',
            'home_road_ot_losses', 'away_road_ot_losses', 'home_road_points', 'away_road_points',
            'home_road_gf', 'away_road_gf', 'home_road_ga', 'away_road_ga',
            'home_road_goal_diff', 'away_road_goal_diff', 'home_road_reg_wins', 'away_road_reg_wins',
            'home_road_reg_ot_wins', 'away_road_reg_ot_wins'
        ],
        'season_overview': [
            'home_games_played', 'away_games_played', 'home_points', 'away_points',
            'home_win_pct', 'away_win_pct', 'home_goal_diff_pct', 'away_goal_diff_pct'
        ]
    }

def get_feature_info() -> Dict[str, str]:
    """
    Get feature descriptions for documentation and analysis.
    
    Returns:
        Dictionary mapping feature names to descriptions
    """
    return {
        # Basic info
        'home_team_id': 'Home team identifier',
        'away_team_id': 'Away team identifier',
        
        # Rest and back-to-back
        'home_rest_days': 'Days of rest for home team',
        'away_rest_days': 'Days of rest for away team',
        'rest_diff': 'Difference in rest days (home - away)',
        'b2b_home': 'Home team playing back-to-back (1/0)',
        'b2b_away': 'Away team playing back-to-back (1/0)',
        
        # Rolling form (recent performance)
        'home_roll_pts': 'Home team points in last 10 games',
        'away_roll_pts': 'Away team points in last 10 games',
        'home_roll_w': 'Home team wins in last 10 games',
        'away_roll_w': 'Away team wins in last 10 games',
        'home_roll_l': 'Home team losses in last 10 games',
        'away_roll_l': 'Away team losses in last 10 games',
        'home_roll_ot_l': 'Home team OT losses in last 10 games',
        'away_roll_ot_l': 'Away team OT losses in last 10 games',
        'home_roll_gf': 'Home team goals for in last 10 games',
        'away_roll_gf': 'Away team goals for in last 10 games',
        'home_roll_ga': 'Home team goals against in last 10 games',
        'away_roll_ga': 'Away team goals against in last 10 games',
        'home_roll_goal_diff': 'Home team goal differential in last 10 games',
        'away_roll_goal_diff': 'Away team goal differential in last 10 games',
        'home_roll_goal_diff_pct': 'Home team goal differential per game in last 10',
        'away_roll_goal_diff_pct': 'Away team goal differential per game in last 10',
        'home_roll_reg_wins': 'Home team regulation wins in last 10 games',
        'away_roll_reg_wins': 'Away team regulation wins in last 10 games',
        'home_roll_reg_ot_wins': 'Home team regulation+OT wins in last 10 games',
        'away_roll_reg_ot_wins': 'Away team regulation+OT wins in last 10 games',
        
        # Current standings (season-long context)
        'home_league_seq': 'Home team current league standing position',
        'away_league_seq': 'Away team current league standing position',
        'home_conf_seq': 'Home team current conference standing position',
        'away_conf_seq': 'Away team current conference standing position',
        'home_div_seq': 'Home team current division standing position',
        'away_div_seq': 'Away team current division standing position',
        
        # Streaks (combined code and count)
        'home_streak': 'Home team current streak (W5=+5, L4=-4, OTL3=-1.5, etc.)',
        'away_streak': 'Away team current streak (W5=+5, L4=-4, OTL3=-1.5, etc.)',
        
        # Home/road splits (situational context)
        'home_home_wins': 'Home team wins at home this season',
        'away_home_wins': 'Away team wins at home this season',
        'home_road_wins': 'Home team wins on road this season',
        'away_road_wins': 'Away team wins on road this season',
        
        # Overall season performance
        'home_points': 'Home team total points this season',
        'away_points': 'Away team total points this season',
        'home_win_pct': 'Home team win percentage this season',
        'away_win_pct': 'Away team win percentage this season',
    }