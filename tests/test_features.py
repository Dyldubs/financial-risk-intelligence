"""
Unit tests for src/data/features.py

Tests the custom transformers, feature name detection,
pipeline construction, and training data preparation.
"""

import numpy as np
import pandas as pd
import pytest

from src.data.features import (
    LogAmountTransformer,
    IsNightTransformer,
    build_feature_pipeline,
    get_feature_names,
    prepare_training_data,
    SYNTHETIC_NUMERIC_FEATURES,
    KAGGLE_NUMERIC_FEATURES,
)


# ---------------------------------------------------------------------------
# LogAmountTransformer
# ---------------------------------------------------------------------------

class TestLogAmountTransformer:
    def test_adds_log_amount_column(self):
        df = pd.DataFrame({"amount": [10.0, 100.0, 1000.0]})
        result = LogAmountTransformer().fit_transform(df)
        assert "log_amount" in result.columns

    def test_log_amount_values_use_log1p(self):
        df = pd.DataFrame({"amount": [0.0, 9.0, 99.0]})
        result = LogAmountTransformer().fit_transform(df)
        expected = np.log1p(df["amount"])
        np.testing.assert_allclose(result["log_amount"].values, expected.values)

    def test_no_amount_column_leaves_df_unchanged(self):
        df = pd.DataFrame({"other": [1.0, 2.0]})
        result = LogAmountTransformer().fit_transform(df)
        assert "log_amount" not in result.columns
        assert list(result.columns) == ["other"]

    def test_does_not_mutate_original_dataframe(self):
        df = pd.DataFrame({"amount": [50.0]})
        LogAmountTransformer().transform(df)
        assert "log_amount" not in df.columns

    def test_fit_returns_self(self):
        t = LogAmountTransformer()
        result = t.fit(pd.DataFrame({"amount": [1.0]}))
        assert result is t


# ---------------------------------------------------------------------------
# IsNightTransformer
# ---------------------------------------------------------------------------

class TestIsNightTransformer:
    @pytest.mark.parametrize("hour", [23, 0, 1, 2, 3, 4, 5])
    def test_night_hours_flagged_as_1(self, hour):
        df = pd.DataFrame({"hour": [hour]})
        result = IsNightTransformer().fit_transform(df)
        assert result["is_night"].iloc[0] == 1

    @pytest.mark.parametrize("hour", [6, 7, 9, 12, 17, 20, 22])
    def test_day_hours_flagged_as_0(self, hour):
        df = pd.DataFrame({"hour": [hour]})
        result = IsNightTransformer().fit_transform(df)
        assert result["is_night"].iloc[0] == 0

    def test_no_hour_column_leaves_df_unchanged(self):
        df = pd.DataFrame({"amount": [100.0]})
        result = IsNightTransformer().fit_transform(df)
        assert "is_night" not in result.columns

    def test_is_night_column_is_integer(self):
        df = pd.DataFrame({"hour": [2, 14]})
        result = IsNightTransformer().fit_transform(df)
        assert result["is_night"].dtype in (int, np.int32, np.int64)

    def test_does_not_mutate_original_dataframe(self):
        df = pd.DataFrame({"hour": [2]})
        IsNightTransformer().transform(df)
        assert "is_night" not in df.columns


# ---------------------------------------------------------------------------
# get_feature_names
# ---------------------------------------------------------------------------

class TestGetFeatureNames:
    def test_returns_synthetic_features_by_default(self, sample_df):
        assert get_feature_names(sample_df) == SYNTHETIC_NUMERIC_FEATURES

    def test_returns_kaggle_features_when_v1_present(self):
        df = pd.DataFrame({"V1": [1.0], "V2": [2.0], "amount": [100.0]})
        assert get_feature_names(df) == KAGGLE_NUMERIC_FEATURES

    def test_synthetic_feature_list_is_non_empty(self):
        assert len(SYNTHETIC_NUMERIC_FEATURES) > 0

    def test_kaggle_feature_list_includes_amount(self):
        assert "amount" in KAGGLE_NUMERIC_FEATURES


# ---------------------------------------------------------------------------
# build_feature_pipeline
# ---------------------------------------------------------------------------

class TestBuildFeaturePipeline:
    def test_output_row_count_matches_input(self, sample_df, feature_names, fitted_pipeline):
        X_t = fitted_pipeline.transform(sample_df[feature_names])
        assert X_t.shape[0] == len(sample_df)

    def test_output_column_count_matches_feature_count(self, sample_df, feature_names, fitted_pipeline):
        X_t = fitted_pipeline.transform(sample_df[feature_names])
        assert X_t.shape[1] == len(feature_names)

    def test_pipeline_has_three_named_steps(self, feature_names):
        pipeline = build_feature_pipeline(feature_names)
        step_names = [name for name, _ in pipeline.steps]
        assert "log_amount" in step_names
        assert "is_night" in step_names
        assert "preprocessor" in step_names

    def test_output_is_numeric_array(self, sample_df, feature_names, fitted_pipeline):
        X_t = fitted_pipeline.transform(sample_df[feature_names])
        assert np.issubdtype(X_t.dtype, np.floating)

    def test_output_is_approximately_standardised(self, sample_df, feature_names, fitted_pipeline):
        X_t = fitted_pipeline.transform(sample_df[feature_names])
        # StandardScaler should bring column means close to zero on training data
        assert abs(X_t.mean()) < 2.0


# ---------------------------------------------------------------------------
# prepare_training_data
# ---------------------------------------------------------------------------

class TestPrepareTrainingData:
    def test_x_columns_match_feature_names(self, sample_df):
        X, _, feature_names = prepare_training_data(sample_df)
        assert list(X.columns) == feature_names

    def test_y_is_integer_dtype(self, sample_df):
        _, y, _ = prepare_training_data(sample_df)
        assert y is not None
        assert y.dtype == int

    def test_y_values_are_binary(self, sample_df):
        _, y, _ = prepare_training_data(sample_df)
        assert set(y.unique()).issubset({0, 1})

    def test_missing_is_fraud_returns_none_y(self, sample_df):
        df = sample_df.drop(columns=["is_fraud"])
        _, y, _ = prepare_training_data(df)
        assert y is None

    def test_row_count_preserved(self, sample_df):
        X, _, _ = prepare_training_data(sample_df)
        assert len(X) == len(sample_df)
