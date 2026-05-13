"""
Tests for src/formulas.py — core index calculation functions.

Tests _nanmean_cols and calculate_simple_iic with a minimal fake config
so no external files or data directories are needed.
"""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.formulas import _nanmean_cols, calculate_simple_iic


# ---------------------------------------------------------------------------
# Fake config injected into calculate_simple_iic via patch
# ---------------------------------------------------------------------------
FAKE_DIMENSION_META = {
    "grupos_prioritarios": {"abbr": "IP", "invert": False},
    "vulnerabilidade":     {"abbr": "IV", "invert": False},
    "gestao_municipal":    {"abbr": "IG", "invert": True},
}
FAKE_DIMENSIONS = {
    "grupos_prioritarios": ["p1", "p2"],
    "vulnerabilidade":     ["v1", "v2"],
    "gestao_municipal":    ["g1", "g2"],
}


def _make_df(**kwargs) -> pd.DataFrame:
    """Build a minimal DataFrame with synthetic indicator values."""
    base = {
        "h3_id": ["hex_a", "hex_b", "hex_c"],
        "p1": [0.0, 0.5, 1.0],
        "p2": [0.0, 0.5, 1.0],
        "v1": [0.2, 0.4, 0.6],
        "v2": [0.2, 0.4, 0.6],
        "g1": [0.0, 0.5, 1.0],
        "g2": [0.0, 0.5, 1.0],
    }
    base.update(kwargs)
    return pd.DataFrame(base)


# ---------------------------------------------------------------------------
# _nanmean_cols
# ---------------------------------------------------------------------------

class TestNanmeanCols:
    def test_basic_mean(self):
        df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        result = _nanmean_cols(df, ["a", "b"])
        pd.testing.assert_series_equal(result, pd.Series([2.0, 3.0]), check_names=False)

    def test_nan_in_one_column_uses_remaining(self):
        df = pd.DataFrame({"a": [np.nan, 2.0], "b": [4.0, 4.0]})
        result = _nanmean_cols(df, ["a", "b"])
        assert result.iloc[0] == pytest.approx(4.0)   # only b available
        assert result.iloc[1] == pytest.approx(3.0)   # mean(2, 4)

    def test_all_nan_row_returns_nan(self):
        df = pd.DataFrame({"a": [np.nan], "b": [np.nan]})
        result = _nanmean_cols(df, ["a", "b"])
        assert np.isnan(result.iloc[0])

    def test_single_column_returns_column_values(self):
        df = pd.DataFrame({"a": [0.3, 0.7]})
        result = _nanmean_cols(df, ["a"])
        pd.testing.assert_series_equal(result, pd.Series([0.3, 0.7]), check_names=False)

    def test_preserves_index(self):
        df = pd.DataFrame({"a": [1.0, 2.0]}, index=[10, 20])
        result = _nanmean_cols(df, ["a"])
        assert list(result.index) == [10, 20]


# ---------------------------------------------------------------------------
# calculate_simple_iic
# ---------------------------------------------------------------------------

class TestCalculateSimpleIic:
    def _run(self, df=None):
        if df is None:
            df = _make_df()
        with patch("src.formulas.cfg") as mock_cfg:
            mock_cfg.DIMENSION_META = FAKE_DIMENSION_META
            mock_cfg.DIMENSIONS = FAKE_DIMENSIONS
            return calculate_simple_iic(df)

    def test_output_columns_created(self):
        result = self._run()
        for col in ["ip", "iv", "ig", "iic_final"]:
            assert col in result.columns, f"Missing column: {col}"

    def test_ig_is_inverted(self):
        # g1=g2=0 → raw mean=0 → after inversion ig=1
        # g1=g2=1 → raw mean=1 → after inversion ig=0
        result = self._run()
        assert result.loc[0, "ig"] == pytest.approx(1.0)
        assert result.loc[2, "ig"] == pytest.approx(0.0)

    def test_ip_is_not_inverted(self):
        # p1=p2=1 → ip=1 (no inversion)
        result = self._run()
        assert result.loc[2, "ip"] == pytest.approx(1.0)

    def test_all_output_values_between_0_and_1(self):
        result = self._run()
        for col in ["ip", "iv", "ig", "iic_final"]:
            assert result[col].between(0.0, 1.0).all(), f"{col} has values outside [0, 1]"

    def test_iic_final_is_mean_of_dimensions(self):
        result = self._run()
        expected = (result["ip"] + result["iv"] + result["ig"]) / 3
        pd.testing.assert_series_equal(
            result["iic_final"], expected, check_names=False, atol=1e-10
        )

    def test_nan_in_indicator_propagates_correctly(self):
        df = _make_df(p1=[np.nan, 0.5, 1.0])  # p1 missing for first row
        result = self._run(df)
        # ip for row 0 should use only p2=0.0, not be NaN
        assert not np.isnan(result.loc[0, "ip"])
        assert result.loc[0, "ip"] == pytest.approx(0.0)

    def test_missing_dimension_skipped_gracefully(self):
        # DataFrame with no g columns → IG dimension absent, iic_final from remaining
        df = _make_df()
        df = df.drop(columns=["g1", "g2"])
        with patch("src.formulas.cfg") as mock_cfg:
            mock_cfg.DIMENSION_META = FAKE_DIMENSION_META
            mock_cfg.DIMENSIONS = FAKE_DIMENSIONS
            result = calculate_simple_iic(df)
        assert "ig" not in result.columns
        assert "iic_final" in result.columns
