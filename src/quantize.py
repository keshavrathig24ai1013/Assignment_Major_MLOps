import numpy as np
import joblib
import os

from utils import (
    load_model,
    load_dataset,
    quantize_to_uint8,
    quantize_to_uint8_individual,
    dequantize_from_uint8,
    dequantize_from_uint8_individual
)

def main():
    print("Loading the trained linear regression model...")
    model = load_model("models/linear_regression_model.joblib")

    weights = model.coef_
    bias = model.intercept_

    print(f"Model weight shape: {weights.shape}")
    print(f"Model intercept: {bias}")
    print(f"Model weights: {weights}")

    # Save unquantized parameters
    raw_parameters = {
        "coef": weights,
        "intercept": bias
    }
    os.makedirs("models", exist_ok=True)
    joblib.dump(raw_parameters, "models/unquant_params.joblib")

    # Quantize model weights
    q_weights, weight_metadata = quantize_to_uint8_individual(weights)

    # Quantize intercept
    print("\nQuantizing intercept...")
    q_bias_array, min_val, max_val, scale = quantize_to_uint8(np.array([bias]))
    q_bias = q_bias_array[0]
    print(f"Intercept scale: {scale:.4f}")

    # Store quantized parameters
    quantized_params = {
        "quant_coef": q_weights,
        "coef_metadata": weight_metadata,
        "quant_intercept": q_bias,
        "int_min": min_val,
        "int_max": max_val,
        "int_scale": scale
    }
    joblib.dump(quantized_params, "models/quant_params.joblib")
    print("Quantized parameters saved to models/quant_params.joblib")

    # Dequantize weights and intercept
    dq_weights = dequantize_from_uint8_individual(q_weights, weight_metadata)
    dq_bias = dequantize_from_uint8(np.array([q_bias]), min_val, max_val, scale)[0]

    # Compare errors
    coef_error = np.max(np.abs(weights - dq_weights))
    bias_error = np.abs(bias - dq_bias)
    print(f"Maximum weight error after dequantization: {coef_error:.8f}")
    print(f"Intercept error after dequantization: {bias_error:.8f}")

    # Perform inference comparison
    X_train, X_test, y_train, y_test = load_dataset()
    x_sample = X_test[:1]

    pred_sklearn = model.predict(x_sample)[0]
    pred_manual = np.dot(x_sample[0], weights) + bias
    pred_dequant = np.dot(x_sample[0], dq_weights) + dq_bias

    preds_sklearn = model.predict(X_test[:10])
    preds_manual = X_test[:10] @ weights + bias
    preds_dequant = X_test[:10] @ dq_weights + dq_bias

    print("\nInference Results (first 10 samples):")
    print("Predictions (sklearn):", preds_sklearn)
    print("Manual predictions (original weights):", preds_manual)
    print("Manual predictions (dequantized weights):", preds_dequant)

    print("\nPrediction Differences:")
    print("Sklearn vs Manual Original:", np.abs(preds_sklearn - preds_manual))
    print("Original vs Dequantized Manual:", np.abs(preds_manual - preds_dequant))

    diffs = np.abs(preds_sklearn - preds_dequant)
    print("Absolute differences:", diffs)
    print("Max difference:", diffs.max())
    print("Mean difference:", diffs.mean())

    # Assess quantization quality
    max_diff = diffs.max()
    if max_diff < 0.1:
        print("Quantization quality: Excellent")
    elif max_diff < 1.0:
        print("Quantization quality: Acceptable")
    else:
        print("Quantization quality: Poor")

if __name__ == "__main__":
    main()
