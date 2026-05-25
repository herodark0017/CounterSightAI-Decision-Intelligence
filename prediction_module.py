
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import shap

def run_prediction_module(csv_path='loan_approval_dataset.csv'):
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    df = df.drop('loan_id', axis=1)

    df['education'] = df['education'].str.strip().map({'Graduate': 1, 'Not Graduate': 0})
    df['self_employed'] = df['self_employed'].str.strip().map({'Yes': 1, 'No': 0})
    df['loan_status'] = df['loan_status'].str.strip().map({'Approved': 1, 'Rejected': 0})

    X = df.drop('loan_status', axis=1)
    y = df['loan_status']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = xgb.XGBClassifier(eval_metric='logloss')
    model.fit(X_train, y_train)

    return model, X_train, X_test, y_train, y_test


def run_shap_module(model, X_train):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_train)
    return explainer, shap_values


def explain_single_prediction(explainer, X_data, row_index=0):
    shap_values = explainer(X_data)
    return shap_values[row_index]
