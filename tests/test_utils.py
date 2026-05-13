"""
Tests for src/utils.py — normalize_minmax and get_next_version_path.

setup_logging and save_parquet are excluded: they have I/O side effects
(create files, write to disk) that belong in integration tests, not unit tests.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils import normalize_minmax, get_next_version_path


# ---------------------------------------------------------------------------
# normalize_minmax
# ---------------------------------------------------------------------------

class TestNormalizeMinmax:
    def test_min_becomes_0_max_becomes_1(self):
        s = pd.Series([0.0, 5.0, 10.0])
        result = normalize_minmax(s)
        assert result.iloc[0] == pytest.approx(0.0)
        assert result.iloc[1] == pytest.approx(0.5)
        assert result.iloc[2] == pytest.approx(1.0)

    def test_constant_series_returns_all_zeros(self):
        s = pd.Series([3.0, 3.0, 3.0])
        result = normalize_minmax(s)
        assert (result == 0.0).all()

    def test_nan_values_preserved(self):
        s = pd.Series([0.0, np.nan, 10.0])
        result = normalize_minmax(s)
        assert np.isnan(result.iloc[1])
        assert result.iloc[0] == pytest.approx(0.0)
        assert result.iloc[2] == pytest.approx(1.0)

    def test_string_coerced_to_nan(self):
        s = pd.Series([0.0, "invalid", 10.0])
        result = normalize_minmax(s)
        assert np.isnan(result.iloc[1])

    def test_winsorize_clips_extremes(self):
        # With winsorize, extreme values (P1 and P99) are clipped before normalization.
        # A value at P99 becomes the new max → normalizes to 1.0.
        # A value well above P99 also becomes 1.0.
        s = pd.Series(list(range(100)) + [9999])  # outlier at the end
        result_plain = normalize_minmax(s, winsorize=False)
        result_wins  = normalize_minmax(s, winsorize=True)
        # With winsorize, the outlier (9999) is clipped → its normalized value < 1 vs plain
        assert result_wins.iloc[-1] == pytest.approx(1.0)
        assert result_plain.iloc[-1] == pytest.approx(1.0)
        # But interior values should be higher after winsorizing (range is compressed)
        assert result_wins.iloc[50] > result_plain.iloc[50]

    def test_output_between_0_and_1(self):
        s = pd.Series(np.random.uniform(-100, 100, 200))
        result = normalize_minmax(s)
        assert result.min() >= 0.0
        assert result.max() <= 1.0


# ---------------------------------------------------------------------------
# get_next_version_path
# ---------------------------------------------------------------------------

class TestGetNextVersionPath:
    def test_nonexistent_file_gets_v1(self, tmp_path):
        p = tmp_path / "result.parquet"
        result = get_next_version_path(p)
        assert result.name == "result_v1.parquet"

    def test_already_versioned_nonexistent_stays(self, tmp_path):
        p = tmp_path / "result_v3.parquet"
        result = get_next_version_path(p)
        assert result.name == "result_v3.parquet"

    def test_existing_file_increments_to_v2(self, tmp_path):
        p = tmp_path / "result_v1.parquet"
        p.touch()
        result = get_next_version_path(p)
        assert result.name == "result_v2.parquet"

    def test_skips_existing_versions(self, tmp_path):
        for name in ["result_v1.parquet", "result_v2.parquet", "result_v3.parquet"]:
            (tmp_path / name).touch()
        p = tmp_path / "result_v1.parquet"
        result = get_next_version_path(p)
        assert result.name == "result_v4.parquet"

    def test_unversioned_existing_file_gets_v1(self, tmp_path):
        p = tmp_path / "result.parquet"
        p.touch()
        result = get_next_version_path(p)
        assert result.name == "result_v1.parquet"
