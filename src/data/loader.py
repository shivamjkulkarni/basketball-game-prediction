"""
Data Ingestion, Validation & Chronological Ordering Module.
"""

from typing import Dict, Optional
import os
import pandas as pd


def load_game_logs(file_path: str) -> pd.DataFrame:
    """
    Ingest raw collegiate game logs, eliminate empty padding rows,
    validate schema, and enforce strict chronological sequence.

    Args:
        file_path (str): Path to CSV game log file.

    Returns:
        pd.DataFrame: Cleaned and chronologically sorted DataFrame.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Game log file not found at: {file_path}")

    # Read data safely
    df = pd.read_csv(file_path, low_memory=False)

    # Standardize column headers
    df.columns = df.columns.str.lower().str.strip()

    # Drop Excel padding rows (empty/zero game_id)
    if 'game_id' in df.columns:
        valid_mask = df['game_id'].notna() & (df['game_id'] != 0) & (df['game_id'] != '0')
        clean_df = df[valid_mask].copy()
    else:
        clean_df = df.dropna(how='all').copy()

    # Convert game_date to datetime
    if 'game_date' in clean_df.columns:
        clean_df['game_date'] = pd.to_datetime(clean_df['game_date'], errors='coerce')
        # Sort chronologically by game_date, preserving game_id order for multi-game dates
        clean_df = clean_df.sort_values(by=['game_date', 'game_id']).reset_index(drop=True)

    return clean_df


def extract_team_regions(df: pd.DataFrame) -> Dict[str, str]:
    """
    Extract a mapping from team identifier to geographic region.

    Args:
        df (pd.DataFrame): DataFrame containing 'team' and 'region' columns.

    Returns:
        Dict[str, str]: team_name -> region_name mapping.
    """
    team_region = {}
    if 'team' in df.columns and 'region' in df.columns:
        for _, row in df.iterrows():
            team = str(row['team']).strip()
            region = str(row['region']).strip().capitalize()
            if team and region and region not in ['Nan', 'None', '', '0']:
                team_region[team] = region
    return team_region

