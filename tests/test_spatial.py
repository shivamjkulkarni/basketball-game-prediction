"""
Unit tests for spatial logistics, geodesic Haversine distance, airport lookups, and rest imputation.
"""

import math
import unittest
import numpy as np
import pandas as pd

from src.config import EARTH_RADIUS_KM
from src.data.spatial import (
    compute_timezone_diff,
    get_team_airport_info,
    haversine_distance,
    haversine_distance_vectorized,
    impute_rest_days,
)


class TestSpatial(unittest.TestCase):

    def test_haversine_distance_known_coordinates(self):
        """Test Haversine distance with known coordinates (NYC to LAX)."""
        # NYC (JFK): 40.6413, -73.7781
        # LA (LAX): 33.9416, -118.4085
        # Expected great-circle distance is ~3975-3985 km
        dist = haversine_distance(40.6413, -73.7781, 33.9416, -118.4085)
        self.assertGreater(dist, 3960.0)
        self.assertLess(dist, 4000.0)

    def test_haversine_distance_identical_points(self):
        """Distance between identical points must be 0.0."""
        dist = haversine_distance(35.9132, -79.0558, 35.9132, -79.0558)
        self.assertAlmostEqual(dist, 0.0, places=6)

    def test_haversine_nan_handling(self):
        """NaN coordinate should safely return 0.0."""
        dist = haversine_distance(float('nan'), -79.0558, 35.9132, -79.0558)
        self.assertEqual(dist, 0.0)

    def test_haversine_vectorized_matches_scalar(self):
        """Vectorized Haversine implementation must match scalar within numerical precision."""
        lats1 = np.array([40.6413, 35.9132, 41.8781])
        lons1 = np.array([-73.7781, -79.0558, -87.6298])
        lats2 = np.array([33.9416, 36.0014, 29.7604])
        lons2 = np.array([-118.4085, -78.9382, -95.3698])

        vec_dists = haversine_distance_vectorized(lats1, lons1, lats2, lons2)
        for i in range(len(lats1)):
            scalar_dist = haversine_distance(lats1[i], lons1[i], lats2[i], lons2[i])
            self.assertAlmostEqual(vec_dists[i], scalar_dist, places=5)

    def test_airport_info_lookup(self):
        """Check airport coordinate retrieval for collegiate programs."""
        duke_info = get_team_airport_info("Duke")
        self.assertIsNotNone(duke_info)
        self.assertIn("latitude", duke_info)
        self.assertIn("longitude", duke_info)
        self.assertIn("iata", duke_info)
        self.assertIn("gmt_offset", duke_info)

        # Case-insensitivity check
        self.assertEqual(get_team_airport_info("duke"), get_team_airport_info("DUKE"))

        # Nonexistent team returns None
        self.assertIsNone(get_team_airport_info("NonexistentUniversity12345"))

    def test_compute_timezone_diff(self):
        """Verify timezone difference calculation between regions."""
        # Duke (EST, UTC-5) vs Gonzaga (PST, UTC-8) -> 3 hours diff
        tz_diff = compute_timezone_diff("Duke", "Gonzaga")
        self.assertEqual(tz_diff, 3.0)

        # Same region teams -> 0 hours diff
        tz_diff_same = compute_timezone_diff("Duke", "North Carolina")
        self.assertEqual(tz_diff_same, 0.0)

    def test_impute_rest_days(self):
        """Verify rest day imputation from sequential game dates."""
        sample_data = pd.DataFrame({
            "game_id": [1, 2, 3, 4, 5, 6],
            "team": ["TeamA", "TeamA", "TeamA", "TeamB", "TeamB", "TeamB"],
            "game_date": [
                "2023-11-01", "2023-11-04", "2023-11-08",
                "2023-11-02", "2023-11-05", "2023-11-10",
            ],
            "rest_days": [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
        })

        imputed = impute_rest_days(sample_data)
        # Check no NaNs remain
        self.assertFalse(imputed["rest_days"].isna().any())

        # For TeamA, game 2 gap is 3 days, game 3 gap is 4 days
        team_a = imputed[imputed["team"] == "TeamA"].sort_values("game_date")
        self.assertEqual(team_a.iloc[1]["rest_days"], 3.0)
        self.assertEqual(team_a.iloc[2]["rest_days"], 4.0)
        # First game should have imputed positive value
        self.assertGreater(team_a.iloc[0]["rest_days"], 0.0)


if __name__ == "__main__":
    unittest.main()

