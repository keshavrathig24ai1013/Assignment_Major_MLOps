import numpy as np
import pytest
import os
import sys
from sklearn.linear_model import LinearRegression

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils import load_dataset, create_model, save_model, load_model


class TestTraining:
    

    def test_dataset_loading(self):
        X_train, X_test, y_train, y_test = load_dataset()

        assert X_train is not None
        assert X_test is not None
        assert y_train is not None
        assert y_test is not None

        assert X_train.shape[1] == 8  
        assert X_test.shape[1] == 8
        assert len(X_train) == len(y_train)
        assert len(X_test) == len(y_test)

        total_samples = len(X_train) + len(X_test)
        train_ratio = len(X_train) / total_samples
        assert 0.75 <= train_ratio <= 0.85

    def test_model_creation(self):
        model = create_model()

        assert isinstance(model, LinearRegression)
        assert hasattr(model, 'fit')
        assert hasattr(model, 'predict')

    def test_model_training(self):
        X_train, X_test, y_train, y_test = load_dataset()
        model = create_model()

        model.fit(X_train, y_train)

        assert hasattr(model, 'coef_')
        assert hasattr(model, 'intercept_')
        assert model.coef_ is not None
        assert model.intercept_ is not None

        assert model.coef_.shape == (8,) 
        assert isinstance(model.intercept_, (float, np.float64))

    def test_model_performance(self):
        from utils import calculate_metrics

        X_train, X_test, y_train, y_test = load_dataset()
        model = create_model()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        r2, mse = calculate_metrics(y_test, y_pred)

        assert r2 > 0.5, f"R² score {r2:.4f} is below the minimum threshold of 0.5"
        assert mse > 0, "MSE must be positive"

        print(f"Model R² Score: {r2:.4f}")
        print(f"Model MSE: {mse:.4f}")

    def test_model_save_load(self):
        X_train, X_test, y_train, y_test = load_dataset()
        model = create_model()
        model.fit(X_train, y_train)

        test_path = "test_model.joblib"
        save_model(model, test_path)

        assert os.path.exists(test_path)

        loaded_model = load_model(test_path)

        pred_original = model.predict(X_test[:5])
        pred_loaded = loaded_model.predict(X_test[:5])

        np.testing.assert_array_almost_equal(pred_original, pred_loaded)

        os.remove(test_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])