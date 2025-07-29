import joblib
from sklearn.linear_model import LinearRegression
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
import pytest

def test_data_load():
    X, y = fetch_california_housing(return_X_y=True)
    assert X.shape[0] > 0

def test_model_instance():
    model = LinearRegression()
    assert isinstance(model, LinearRegression)

def test_trained_model():
    model = joblib.load("artifacts/model.joblib")
    assert hasattr(model, "coef_")

def test_model_score():
    X, y = fetch_california_housing(return_X_y=True)
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2)
    model = joblib.load("artifacts/model.joblib")
    r2 = model.score(X_test, y_test)
    assert r2 > 0.5
    print("All tests passed.")