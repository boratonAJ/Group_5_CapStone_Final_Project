import pandas as pd

from src.features.engineering import build_binary_target


def test_build_binary_target_filters_and_maps_values():
    df = pd.DataFrame({"action_taken": [1, 2, 3, 4, 7, None]})
    out = build_binary_target(df, target_col="action_taken")

    assert out["action_taken"].tolist() == [1, 1, 0]
    assert out["action_taken"].dtype == "int64"


def test_build_binary_target_keeps_existing_binary_labels():
    df = pd.DataFrame({"action_taken": [1, 0, 1, 0]})
    out = build_binary_target(df, target_col="action_taken")

    assert out["action_taken"].tolist() == [1, 0, 1, 0]
