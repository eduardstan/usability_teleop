import pandas as pd

from usability_teleop.features.ee_quat import (
    FeatureSetSpec,
    aggregate_user_level_features,
    build_feature_set,
    generate_ee_quat_feature_sets,
)
from usability_teleop.protocol.selection import SelectionConfig, select_full_features


def test_generate_ee_quat_feature_sets_count_and_order() -> None:
    specs = generate_ee_quat_feature_sets(include_average=True)
    assert len(specs) == 16
    assert specs[0].name == "x"
    assert specs[1].name == "y"
    assert specs[2].name == "z"
    assert specs[3].name == "w"
    assert specs[-1].name == "avg"


def test_build_average_feature_set_has_19_columns() -> None:
    df = pd.DataFrame(
        {
            "min_ee_quat.x": [1.0, 2.0],
            "min_ee_quat.y": [3.0, 4.0],
            "min_ee_quat.z": [5.0, 6.0],
            "min_ee_quat.w": [7.0, 8.0],
            "max_ee_quat.x": [1.0, 1.0],
            "max_ee_quat.y": [3.0, 3.0],
            "max_ee_quat.z": [5.0, 5.0],
            "max_ee_quat.w": [7.0, 7.0],
            "mean_ee_quat.x": [1.0, 1.0],
            "mean_ee_quat.y": [2.0, 2.0],
            "mean_ee_quat.z": [3.0, 3.0],
            "mean_ee_quat.w": [4.0, 4.0],
            "median_ee_quat.x": [1.0, 1.0],
            "median_ee_quat.y": [2.0, 2.0],
            "median_ee_quat.z": [3.0, 3.0],
            "median_ee_quat.w": [4.0, 4.0],
            "std_ee_quat.x": [1.0, 1.0],
            "std_ee_quat.y": [2.0, 2.0],
            "std_ee_quat.z": [3.0, 3.0],
            "std_ee_quat.w": [4.0, 4.0],
            "variance_ee_quat.x": [1.0, 1.0],
            "variance_ee_quat.y": [2.0, 2.0],
            "variance_ee_quat.z": [3.0, 3.0],
            "variance_ee_quat.w": [4.0, 4.0],
            "4th_moment_ee_quat.x": [1.0, 1.0],
            "4th_moment_ee_quat.y": [2.0, 2.0],
            "4th_moment_ee_quat.z": [3.0, 3.0],
            "4th_moment_ee_quat.w": [4.0, 4.0],
            "5th_moment_ee_quat.x": [1.0, 1.0],
            "5th_moment_ee_quat.y": [2.0, 2.0],
            "5th_moment_ee_quat.z": [3.0, 3.0],
            "5th_moment_ee_quat.w": [4.0, 4.0],
            "skewness_ee_quat.x": [1.0, 1.0],
            "skewness_ee_quat.y": [2.0, 2.0],
            "skewness_ee_quat.z": [3.0, 3.0],
            "skewness_ee_quat.w": [4.0, 4.0],
            "kurtosis_ee_quat.x": [1.0, 1.0],
            "kurtosis_ee_quat.y": [2.0, 2.0],
            "kurtosis_ee_quat.z": [3.0, 3.0],
            "kurtosis_ee_quat.w": [4.0, 4.0],
            "rms_ee_quat.x": [1.0, 1.0],
            "rms_ee_quat.y": [2.0, 2.0],
            "rms_ee_quat.z": [3.0, 3.0],
            "rms_ee_quat.w": [4.0, 4.0],
            "iqr_ee_quat.x": [1.0, 1.0],
            "iqr_ee_quat.y": [2.0, 2.0],
            "iqr_ee_quat.z": [3.0, 3.0],
            "iqr_ee_quat.w": [4.0, 4.0],
            "total_sum_ee_quat.x": [1.0, 1.0],
            "total_sum_ee_quat.y": [2.0, 2.0],
            "total_sum_ee_quat.z": [3.0, 3.0],
            "total_sum_ee_quat.w": [4.0, 4.0],
            "range_ee_quat.x": [1.0, 1.0],
            "range_ee_quat.y": [2.0, 2.0],
            "range_ee_quat.z": [3.0, 3.0],
            "range_ee_quat.w": [4.0, 4.0],
            "entropy_ee_quat.x": [1.0, 1.0],
            "entropy_ee_quat.y": [2.0, 2.0],
            "entropy_ee_quat.z": [3.0, 3.0],
            "entropy_ee_quat.w": [4.0, 4.0],
            "std_peaks_ee_quat.x": [1.0, 1.0],
            "std_peaks_ee_quat.y": [2.0, 2.0],
            "std_peaks_ee_quat.z": [3.0, 3.0],
            "std_peaks_ee_quat.w": [4.0, 4.0],
            "spectral_power_ee_quat.x": [1.0, 1.0],
            "spectral_power_ee_quat.y": [2.0, 2.0],
            "spectral_power_ee_quat.z": [3.0, 3.0],
            "spectral_power_ee_quat.w": [4.0, 4.0],
            "spectral_mean_ee_quat.x": [1.0, 1.0],
            "spectral_mean_ee_quat.y": [2.0, 2.0],
            "spectral_mean_ee_quat.z": [3.0, 3.0],
            "spectral_mean_ee_quat.w": [4.0, 4.0],
            "spectral_median_ee_quat.x": [1.0, 1.0],
            "spectral_median_ee_quat.y": [2.0, 2.0],
            "spectral_median_ee_quat.z": [3.0, 3.0],
            "spectral_median_ee_quat.w": [4.0, 4.0],
        }
    )

    avg_df = build_feature_set(df, FeatureSetSpec(name="avg", axes=("x", "y", "z", "w"), mode="average"))
    assert avg_df.shape == (2, 19)
    assert avg_df["min_ee_quat.avg"].tolist() == [4.0, 5.0]


def test_aggregate_user_level_features_means_by_user() -> None:
    raw = pd.DataFrame({"f1": [1.0, 3.0, 10.0, 20.0]})
    labels = pd.DataFrame(
        {
            "task_id": [1, 1, 1, 1],
            "user_id": [1, 1, 2, 2],
            "rep_id": [1, 2, 1, 2],
        }
    )

    out = aggregate_user_level_features(raw, labels)
    assert list(out.index) == [1, 2]
    assert out.loc[1, "f1"] == 2.0
    assert out.loc[2, "f1"] == 15.0


def test_top_k_per_axis_keeps_avg_feature_set_columns() -> None:
    avg_df = pd.DataFrame(
        {
            "min_ee_quat.avg": [0.1, 0.3, 0.2],
            "max_ee_quat.avg": [1.0, 0.8, 1.2],
            "mean_ee_quat.avg": [0.4, 0.5, 0.6],
        }
    )
    selected, cols = select_full_features(avg_df, SelectionConfig(top_k_per_axis=1))
    assert len(cols) == 1
    assert selected.shape[1] == 1
    assert cols[0] in set(avg_df.columns)
