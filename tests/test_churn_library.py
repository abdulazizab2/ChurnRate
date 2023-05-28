"""Unit test for ChurnRate/churn_library

    Author: Abdulaziz
    Date: 28th May 2023
"""

# pylint: disable=redefined-outer-name
import sys  # pylint: disable=wrong-import-position
sys.path.insert(0, './ChurnRate')  # pylint: disable=wrong-import-position
import pytest
from ChurnRate import constants
from ChurnRate.churn_library import read_data, perform_eda, encoder_helper, perform_feature_engineering
import os
import pandas as pd


@pytest.fixture
def df_fixture():
    """Fixture data for testing"""
    df = read_data("data/bank_data.csv")
    df['Churn'] = df['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)
    for column_name in constants.CATEGORICAL_COLUMNS:
        churn_column_name = f"{column_name}_Churn"
        df[churn_column_name] = 0
    return df


def test_read_data(df_fixture):
    """test read_data function"""
    assert isinstance(df_fixture, pd.DataFrame)


def test_perform_eda(df_fixture):
    """test perform_eda function"""
    perform_eda(df_fixture)
    assert os.path.exists(f"{constants.EDA_DIR}/customer_age_hist_plot.png")


def test_encoder_helper(df_fixture):
    """test encoder_helper function"""
    data = encoder_helper(df_fixture, "Gender")
    assert "Gender_Churn" in data.columns


def test_perform_feature_engineering(df_fixture):
    """test perform_feature_engineering function"""
    try:
        perform_feature_engineering(df_fixture)
    except Exception as err:
        pytest.fail(f"Function raised an exception: {err}")


def test_train_models():
    """test train_models function"""
    assert f"{constants.MODELS_DIR}/logistic_model.pkl"


def test_feature_importance_plot():
    """test feature_importance_plot function"""
    assert f"{constants.RESULTS_DIR}/feature_importance.png"


def test_classification_report_image():
    """test classification_report_image function"""
    assert f"{constants.RESULTS_DIR}/logistic_regression_report.png"
