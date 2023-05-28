"""Library for performing end2end datascience on churn data

    Author: Abdulaziz
    Date: May 28th 2023
"""

from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import constants
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils.logger import logging
import seaborn as sns
sns.set()


def read_data(data_path):
    """
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            data: pandas dataframe
    """
    try:
        data = pd.read_csv(data_path)
        logging.info("SUCCESS: Data is read")
        return data
    except FileNotFoundError as err:
        raise RuntimeError(
            f"ERROR: Couldn't find file path for data: {data_path} - " +
            str(err)) from err


def perform_eda(data):
    """
    perform eda on data and save figures to images folder
    input:
            data: pandas dataframe

    output:
            data: Dataframe with churn column
    """

    data['Churn'] = data['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)
    plt.figure(figsize=(20, 10))
    data['Churn'].hist()
    plt.title("Churn Histogram Plot")
    plt.xlabel("Churn")
    plt.ylabel("Frequency")
    plt.savefig(f"{constants.EDA_DIR}/churn_hist_plot.png")

    plt.figure(figsize=(20, 10))
    data['Customer_Age'].hist()
    plt.title("Customer Age Histogram Plot")
    plt.xlabel("Customer Age")
    plt.ylabel("Frequency")
    plt.savefig(f"{constants.EDA_DIR}/customer_age_hist_plot.png")

    plt.figure(figsize=(20, 10))
    data.Marital_Status.value_counts(
        normalize=True).plot(
        kind='bar', color=[
            'black', 'b', 'r', 'g'])
    plt.title("Marital Status Bar Plot")
    plt.xlabel("Marital Status")
    plt.ylabel("Normalized Count")
    plt.savefig(f"{constants.EDA_DIR}/marital_status_bar_plot.png")

    plt.figure(figsize=(20, 10))
    sns.histplot(data['Total_Trans_Ct'], stat='density', kde=True)
    plt.savefig(f"{constants.EDA_DIR}/total_trans_plot.png")

    plt.figure(figsize=(20, 10))
    sns.histplot(data['Total_Trans_Ct'], stat='density', kde=True)
    sns.heatmap(data.corr(), annot=False, cmap='Dark2_r', linewidths=2)
    plt.title("Feature Correlation Heatmap")
    plt.savefig(f"{constants.EDA_DIR}/feature_correlation_heatmap.png")

    logging.info(f"INFO: Plots are saved in {constants.EDA_DIR}")

    return data


def encoder_helper(data, column_name):
    """
    helper function to turn each categorical column into a new column with
    propotion of churn for each category

    input:
            data: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could be used for naming variables or index target column]

    output:
            data: pandas dataframe with new columns
    """

    churn_column_name = f"{column_name}_Churn"
    data[churn_column_name] = data.groupby(column_name)["Churn"].transform("mean")
    return data


def perform_feature_engineering(data):
    """
    input:
              data: pandas dataframe
              response: string of response name [optional argument that could be used for naming variables or index target column]

    output:
              x_train: X training data
              x_test: X testing data
              y_train: target training data
              y_test: target testing data
              x_data: features before trian/test split
    """
    target = data[constants.TARGET]
    x_data = data[constants.KEEP_COLUMNS]
    x_train, x_test, y_train, y_test = train_test_split(
        x_data, target, test_size=0.3, random_state=42)
    return x_train, x_test, y_train, y_test, x_data


def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    """
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    """
    plt.rc('figure', figsize=(5, 5))
    plt.text(0.01, 1.25, str('Random Forest Train'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05, str(classification_report(y_test, y_test_preds_rf)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.text(0.01, 0.6, str('Random Forest Test'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report(y_train, y_train_preds_rf)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.axis('off')
    plt.savefig(f"{constants.RESULTS_DIR}/random_forest_report.png")

    plt.rc('figure', figsize=(5, 5))
    plt.text(0.01, 1.25, str('Logistic Regression Train'),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05, str(classification_report(y_train, y_train_preds_lr)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.text(0.01, 0.6, str('Logistic Regression Test'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report(y_test, y_test_preds_lr)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.axis('off')
    plt.savefig(f"{constants.RESULTS_DIR}/logistic_regression_report.png")

    logging.info(
        f"INFO: Classification report results are saved in {constants.RESULTS_DIR}")


def feature_importance_plot(model, x_data):
    """
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            x_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    """
    importances = model.best_estimator_.feature_importances_
    indices = np.argsort(importances)[::-1]
    names = [x_data.columns[i] for i in indices]

    plt.figure(figsize=(20, 5))
    plt.title("Feature Importance")
    plt.ylabel('Importance')
    plt.bar(range(x_data.shape[1]), importances[indices])
    plt.xticks(range(x_data.shape[1]), names, rotation=90)
    plt.savefig(f"{constants.RESULTS_DIR}/feature_importance.png")

    logging.info(
        f"INFO: Feature importance result plot is saved in {constants.RESULTS_DIR}")


def train_models(x_train, x_test, y_train, y_test, x_data):
    """
    train, store model results: images + scores, and store models
    input:
              x_train: X training data
              x_test: X testing data
              y_train: target training data
              y_test: target testing data
              x_data: features before trian/test split
    output:
              None
    """
    rfc = RandomForestClassifier(random_state=42)
    lrc = LogisticRegression(solver="lbfgs", max_iter=3000)

    param_grid = {
        "n_estimators": [200, 500],
        "max_features": ["auto", "sqrt"],
        "max_depth": [4, 5, 100],
        "criterion": ["gini", "entropy"]
    }

    logging.info("INFO: Started training models")
    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(x_train, y_train)

    lrc.fit(x_train, y_train)

    joblib.dump(
        cv_rfc.best_estimator_,
        f"{constants.MODELS_DIR}/rfc_model.pkl")
    joblib.dump(lrc, f"{constants.MODELS_DIR}/logistic_model.pkl")
    logging.info(
        f"SUCCESS: Models are trained and their serialized weights are saved in f{constants.RESULTS_DIR}")

    y_train_preds_rf = cv_rfc.best_estimator_.predict(x_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(x_test)

    y_train_preds_lr = lrc.predict(x_train)
    y_test_preds_lr = lrc.predict(x_test)

    classification_report_image(
        y_train,
        y_test,
        y_train_preds_lr,
        y_train_preds_rf,
        y_test_preds_lr,
        y_test_preds_rf)
    feature_importance_plot(cv_rfc, x_data)


def main():

    data = read_data("data/bank_data.csv")
    data = perform_eda(data)

    for column_name in constants.CATEGORICAL_COLUMNS:
        data = encoder_helper(data, column_name)

    x_train, x_test, y_train, y_test, x_data = perform_feature_engineering(data)
    train_models(x_train, x_test, y_train, y_test, x_data)


if __name__ == "__main__":
    main()
