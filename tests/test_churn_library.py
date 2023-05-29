"""Unit test for ChurnRate/churn_library

    Author: Abdulaziz
    Date: 28th May 2023
"""

# pylint: disable=redefined-outer-name
import sys  # pylint: disable=wrong-import-position
sys.path.insert(0, './ChurnRate')  # pylint: disable=wrong-import-position
import os
import pytest
import joblib
from ChurnRate import constants
from ChurnRate.churn_library import read_data, perform_eda, encoder_helper, perform_feature_engineering, train_models, feature_importance_plot, classification_report_image
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

@pytest.fixture
def data_fixture(df_fixture):
    x_train, x_test, y_train, y_test, x_data = perform_feature_engineering(df_fixture)
    return x_train, x_test, y_train, y_test, x_data

@pytest.fixture
def models_fixture():
    
    lrc_model = joblib.load(f"{constants.MODELS_DIR}/logistic_model.pkl")
    rfc_model = joblib.load(f"{constants.MODELS_DIR}/rfc_model.pkl")
    yield lrc_model, rfc_model


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


def test_train_models(data_fixture):
    """test train_models function"""
    x_train, x_test, y_train, y_test, x_data = data_fixture
    try:    
        train_models(x_train, x_test, y_train, y_test, x_data)
    except Exception as err:
        pytest.fail(f"Function raised an exception: {err}")


def test_feature_importance_plot(data_fixture, models_fixture):
    """test feature_importance_plot function"""
    _, _, _, _, x_data = data_fixture
    _, rfc_model = models_fixture
    try:
        feature_importance_plot(rfc_model, x_data)
    except Exception as err:
        pytest.fail(f"Function raised an exception: {err}")


def test_classification_report_image(data_fixture, models_fixture):
    """test classification_report_image function"""
    x_train, x_test, y_train, y_test, _ = data_fixture
    lrc_model, rfc_model = models_fixture
    y_train_preds_rf = rfc_model.predict(x_train)
    y_test_preds_rf = rfc_model.predict(x_test)

    y_train_preds_lr = lrc_model.predict(x_train)
    y_test_preds_lr = lrc_model.predict(x_test)
    try:
        classification_report_image(y_train, y_test, y_train_preds_lr, y_train_preds_rf, y_test_preds_lr, y_test_preds_rf)
    except Exception as err:
        pytest.fail(f"Function raised an exception: {err}")