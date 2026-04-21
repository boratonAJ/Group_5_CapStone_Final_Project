import pandas as pd

from src.models.fairness import fairness_by_group


def test_fairness_by_group_outputs_expected_columns():
    y_true = pd.Series([1, 1, 0, 0, 1, 0, 1, 0])
    y_pred = pd.Series([1, 0, 0, 0, 1, 1, 1, 0])
    groups = pd.Series(["A", "A", "A", "A", "B", "B", "B", "B"])

    out = fairness_by_group(y_true=y_true, y_pred=y_pred, group_series=groups, min_group_size=1)

    required = {
        "group",
        "n",
        "selection_rate",
        "base_rate",
        "reference_group",
        "air",
        "me",
        "smd",
        "p_value_selection_rate_vs_ref",
        "fpr",
        "fnr",
    }

    assert not out.empty
    assert required.issubset(set(out.columns))
