import numpy as np
from utils import load_model, load_dataset, calculate_metrics

def main():
    print("Loading trained model.")
    model = load_model("models/linear_regression_model.joblib")

    print("Loading test dataset.")
    X_train, X_test, y_train, y_test = load_dataset()

    print("Making predictions.")
    y_pred = model.predict(X_test)

    r2, mse = calculate_metrics(y_test, y_pred)

    print(f"Model Performance:")
    print(f"R² Score: {r2:.4f}")
    print(f"Mean Squared Error: {mse:.4f}")

    print("\n Sample Predictions (first 10):")
    for i in range(10):
        print(f"True: {y_test[i]:.2f} | Predicted: {y_pred[i]:.2f} | Diff: {abs(y_test[i] - y_pred[i]):.2f}")

    print("\n Prediction has been successfully completed.")
    return True


if __name__ == "__main__":
    main()