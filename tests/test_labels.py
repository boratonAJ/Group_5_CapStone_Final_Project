"""Unit tests for src/labels.py — label construction logic."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import pytest
from src.labels import apply_label, label_distribution_table, LABEL_MAP


class TestApplyLabel:

    def test_basic_mapping(self):
        df = pd.DataFrame({"action_taken": [1, 2, 3]})
        result = apply_label(df)
        assert list(result["y"].values) == [1, 1, 0]

    def test_drops_values_4_through_8(self):
        df = pd.DataFrame({"action_taken": [4, 5, 6, 7, 8]})
        result = apply_label(df)
        assert len(result) == 0

    def test_mixed_input(self):
        df = pd.DataFrame({"action_taken": [1, 2, 3, 4, 5, 6, 7, 8]})
        result = apply_label(df)
        assert len(result) == 3
        assert set(result["action_taken"].values) == {1, 2, 3}

    def test_y_column_is_integer(self):
        df = pd.DataFrame({"action_taken": [1, 3]})
        result = apply_label(df)
        assert result["y"].dtype == int

    def test_positive_rate_correct(self):
        df = pd.DataFrame({"action_taken": [1, 1, 3]})
        result = apply_label(df)
        assert result["y"].mean() == pytest.approx(2 / 3)

    def test_preserves_other_columns(self):
        df = pd.DataFrame({"action_taken": [1, 3], "loan_amount": [200000, 150000]})
        result = apply_label(df)
        assert "loan_amount" in result.columns
        assert list(result["loan_amount"].values) == [200000, 150000]

    def test_raises_if_column_missing(self):
        df = pd.DataFrame({"some_other_col": [1, 2, 3]})
        with pytest.raises(ValueError, match="action_taken"):
            apply_label(df)

    def test_handles_string_action_taken(self):
        df = pd.DataFrame({"action_taken": ["1", "2", "3", "4"]})
        result = apply_label(df)
        assert len(result) == 3

    def test_all_denied(self):
        df = pd.DataFrame({"action_taken": [3, 3, 3]})
        result = apply_label(df)
        assert result["y"].sum() == 0

    def test_all_approved(self):
        df = pd.DataFrame({"action_taken": [1, 2, 1]})
        result = apply_label(df)
        assert result["y"].mean() == 1.0

    def test_resets_index(self):
        df = pd.DataFrame({"action_taken": [1, 5, 3]})
        result = apply_label(df)
        assert list(result.index) == [0, 1]


class TestLabelDistributionTable:

    def test_returns_dataframe(self):
        df = pd.DataFrame({"action_taken": [1, 2, 3, 4]})
        result = label_distribution_table(df)
        assert isinstance(result, pd.DataFrame)

    def test_all_values_covered(self):
        df = pd.DataFrame({"action_taken": [1, 2, 3, 4, 5, 6, 7, 8]})
        result = label_distribution_table(df)
        assert len(result) == 8

    def test_retained_flag_correct(self):
        df = pd.DataFrame({"action_taken": [1, 2, 3, 4]})
        result = label_distribution_table(df)
        retained = result[result["action_taken"].isin([1, 2, 3])]["retained"]
        not_retained = result[result["action_taken"] == 4]["retained"]
        assert retained.all()
        assert not not_retained.any()
