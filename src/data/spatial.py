"""
Spatial Logistics, Geodesic Distances, Airport Lookups, and Rest Imputation.
"""

import json
import math
import os
from typing import Dict, Optional, Tuple, Union
import numpy as np
import pandas as pd

from src.config import DEFAULT_REST_DAYS, EARTH_RADIUS_KM

# In-memory airport cache
_AIRPORT_CACHE: Optional[Dict[str, Dict]] = None


def _load_airport_cache() -> Dict[str, Dict]:
    """Load embedded or local team airport database."""
    global _AIRPORT_CACHE
    if _AIRPORT_CACHE is not None:
        return _AIRPORT_CACHE

    json_path = os.path.join(
        os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')),
        'data',
        'team_airports.json',
    )
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            _AIRPORT_CACHE = json.load(f)
    else:
        _AIRPORT_CACHE = {}
    return _AIRPORT_CACHE


def get_team_airport_info(team: str) -> Optional[Dict[str, Union[str, float]]]:
    """
    Retrieve nearest airport code and geographic coordinates for a collegiate program.

    Args:
        team (str): Team name identifier.

    Returns:
        Optional[Dict]: Dictionary containing 'iata', 'latitude', 'longitude', 'timezone', 'gmt_offset'.
    """
    cache = _load_airport_cache()
    if not cache:
        return None
    raw = str(team).strip().lower()
    norm = raw.replace(' ', '_').replace('-', '_')
    if norm in cache:
        return cache[norm]
    if raw in cache:
        return cache[raw]
    # Check if team is prefix e.g. "duke" -> "duke_blue_devils"
    prefix_matches = [k for k in cache if k.startswith(norm + '_')]
    if prefix_matches:
        return cache[prefix_matches[0]]
    # Check substring match
    sub_matches = [k for k in cache if norm in k]
    if sub_matches:
        return cache[sub_matches[0]]
    return None


def haversine_distance(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
    radius: float = EARTH_RADIUS_KM,
) -> float:
    """
    Compute geodesic distance between two coordinate pairs using Haversine formula.

    Args:
        lat1, lon1: Coordinates of location 1 in decimal degrees.
        lat2, lon2: Coordinates of location 2 in decimal degrees.
        radius: Sphere radius in kilometers (default 6371 km).

    Returns:
        float: Great-circle distance in kilometers.
    """
    if any(map(math.isnan, [lat1, lon1, lat2, lon2])):
        return 0.0

    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = (
        math.sin(delta_phi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
    )
    a_clamped = min(max(a, 0.0), 1.0)
    c = 2.0 * math.atan2(math.sqrt(a_clamped), math.sqrt(1.0 - a_clamped))
    return radius * c


def haversine_distance_vectorized(
    lat1: Union[np.ndarray, pd.Series],
    lon1: Union[np.ndarray, pd.Series],
    lat2: Union[np.ndarray, pd.Series],
    lon2: Union[np.ndarray, pd.Series],
    radius: float = EARTH_RADIUS_KM,
) -> np.ndarray:
    """
    Vectorized great-circle Haversine distance computation across NumPy arrays.

    Args:
        lat1, lon1, lat2, lon2: Coordinate arrays in decimal degrees.
        radius: Earth radius in kilometers (default 6371 km).

    Returns:
        np.ndarray: Distances in kilometers.
    """
    lat1_r, lon1_r, lat2_r, lon2_r = map(np.radians, [lat1, lon1, lat2, lon2])

    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2.0) ** 2
    a = np.clip(a, 0.0, 1.0)
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return radius * c


def compute_timezone_diff(
    team_a: str,
    team_b: str,
    default_diff: float = 0.0,
) -> float:
    """
    Compute time zone difference in hours between two teams' home airport regions.

    Args:
        team_a (str): Primary team identifier.
        team_b (str): Venue / opponent team identifier.
        default_diff (float): Fallback difference if coordinates unavailable.

    Returns:
        float: Absolute time zone difference in hours.
    """
    info_a = get_team_airport_info(team_a)
    info_b = get_team_airport_info(team_b)

    if info_a and info_b and 'gmt_offset' in info_a and 'gmt_offset' in info_b:
        return abs(float(info_a['gmt_offset']) - float(info_b['gmt_offset']))
    return default_diff


def impute_rest_days(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute game gaps and impute missing rest days using expanding team averages.

    Args:
        df (pd.DataFrame): DataFrame containing 'team', 'game_date', and optionally 'rest_days'.

    Returns:
        pd.DataFrame: DataFrame with clean 'rest_days' column.
    """
    clean_df = df.copy()
    if 'game_date' not in clean_df.columns or 'team' not in clean_df.columns:
        return clean_df

    clean_df['game_date'] = pd.to_datetime(clean_df['game_date'])
    clean_df = clean_df.sort_values(by=['team', 'game_date']).reset_index(drop=True)

    def _impute_team(team_group: pd.DataFrame) -> pd.DataFrame:
        tg = team_group.copy()
        # Compute date gap from preceding game
        gap = tg['game_date'].diff().dt.days

        # If rest_days column exists, fill missing values from date gap
        if 'rest_days' in tg.columns:
            tg['rest_days'] = pd.to_numeric(tg['rest_days'], errors='coerce').fillna(gap)
        else:
            tg['rest_days'] = gap

        # Expanding average of rest days for first game imputation
        expanding_mean = tg['rest_days'].expanding().mean()
        overall_mean = tg['rest_days'].mean(skipna=True)
        fallback = overall_mean if not pd.isna(overall_mean) and overall_mean > 0 else DEFAULT_REST_DAYS

        # First game has NaN gap -> impute with team average or fallback
        tg['rest_days'] = tg['rest_days'].fillna(expanding_mean).fillna(fallback)
        return tg

    imputed_list = []
    for _, group in clean_df.groupby('team', sort=False):
        imputed_list.append(_impute_team(group))
    clean_df = pd.concat(imputed_list, ignore_index=True)
    return clean_df.sort_values(by=['game_date', 'game_id']).reset_index(drop=True)
