"""Unit tests for src/leakage.py — post-decision feature removal."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import pytest
from src.leakage import remove_leakage, POST_DECISION_FEATURES, get_proxy_risk_table


class TestRemoveLeakage:

    def _make_df(self, cols):
        return pd.DataFrame({c: [1, 2, 3] for c in cols})

    def test_removes_rate_spread(self):
        df = self._make_df(["income", "loan_amount", "rate_spread"])
        result = remove_leakage(df)
        assert "rate_spread" not in result.columns
        assert "income" in result.columns
        assert "loan_amount" in result.columns

    def test_removes_denial_reasons(self):
        cols = ["income", "denial_reason_1", "denial_reason_2", "denial_reason_3", "denial_reason_4"]
        df = self._make_df(cols)
        result = remove_leakage(df)
        for col in ["denial_reason_1", "denial_reason_2", "denial_reason_3", "denial_reason_4"]:
            assert col not in result.columns

    def test_removes_interest_rate(self):
        df = self._make_df(["income", "interest_rate"])
        result = remove_leakage(df)
        assert "interest_rate" not in result.columns

    def test_removes_all_post_decision(self):
        all_cols = ["income", "loan_amount"] + POST_DECISION_FEATURES
        df = self._make_df(all_cols)
        result = remove_leakage(df)
        for feat in POST_DECISION_FEATURES:
            assert feat not in result.columns
        assert "income" in result.columns
        assert "loan_amount" in result.columns

    def test_handles_absent_features_gracefully(self):
        df = self._make_df(["income", "loan_amount"])
        result = remove_leakage(df)
        assert "income" in result.columns
        assert len(result) == 3

    def test_extra_features_removed(self):
        df = self._make_df(["income", "my_extra_leakage_col"])
        result = remove_leakage(df, extra=["my_extra_leakage_col"])
        assert "my_extra_leakage_col" not in result.columns
        assert "income" in result.columns

    def test_preserves_row_count(self):
        df = self._make_df(["income", "rate_spread", "loan_amount"])
        result = remove_leakage(df)
        assert len(result) == 3


class TestProxyRiskTable:

    def test_returns_dataframe(self):
        result = get_proxy_risk_table()
        assert isinstance(result, pd.DataFrame)

    def test_has_required_columns(self):
        result = get_proxy_risk_table()
        assert "feature" in result.columns
        assert "risk_level" in result.columns
        assert "justification" in result.columns

    def test_risk_levels_valid(self):
        result = get_proxy_risk_table()
        valid_levels = {"High", "Medium", "Low"}
        assert set(result["risk_level"].unique()).issubset(valid_levels)

    def test_census_tract_is_high_risk(self):
        result = get_proxy_risk_table()
        ct_row = result[result["feature"] == "census_tract"]
        assert len(ct_row) > 0
        assert ct_row["risk_level"].values[0] == "High"
