import numpy as np
from utils import load_model, quantize_to_uint8, dequantize_from_uint8, quantize_to_uint8_individual, dequantize_from_uint8_individual
import joblib
import os

def main():
    print("Loading trained model.")
    model = load_model("models/linear_regression_model.joblib")

    coef = model.coef_
    intercept = model.intercept_

    print(f"Original coefficients shape is : {coef.shape}")
    print(f"Original intercept is : {intercept}")
    print(f"Original coef values is : {coef}")

    raw_params = {
        'coef': coef,
        'intercept': intercept
    }
    os.makedirs("models", exist_ok=True)
    joblib.dump(raw_params, "models/unquant_params.joblib")

    quant_coef, coef_metadata = quantize_to_uint8_individual(coef)

    print(f"\n Quantizing intercept.")
    print(f"Intercept value: {intercept:.8f}")
    quant_intercept, int_min, int_max, int_scale = quantize_to_uint8(np.array([intercept]))
    print(f"Intercept scale factor: {int_scale:.2f}")

    quant_params = {
        'quant_coef': quant_coef,
        'coef_metadata': coef_metadata,
        'quant_intercept': quant_intercept[0],
        'int_min': int_min,
        'int_max': int_max,
        'int_scale': int_scale
    }
    joblib.dump(quant_params, "models/quant_params.joblib")
    print("Quantized parameters saved to models/quant_params.joblib")

    dequant_coef = dequantize_from_uint8_individual(quant_coef, coef_metadata)
    dequant_intercept_array = dequantize_from_uint8(np.array([quant_params['quant_intercept']]), int_min, int_max,
                                                    int_scale)
    dequant_intercept = dequant_intercept_array[0]

    coef_error = np.abs(coef - dequant_coef).max()
    intercept_error = np.abs(intercept - dequant_intercept)
    print(f"Max coefficient error: {coef_error:.8f}")
    print(f"Intercept error: {intercept_error:.8f}")

    from utils import load_dataset
    X_train, X_test, y_train, y_test = load_dataset()

    test_sample = X_test[0:1] 

    original_pred_single = model.predict(test_sample)[0]
    manual_original = np.dot(test_sample[0], coef) + intercept
    manual_dequant = np.dot(test_sample[0], dequant_coef) + dequant_intercept
    original_pred = model.predict(X_test[:10])
    manual_original_pred = X_test[:10] @ coef + intercept
    manual_dequant_pred = X_test[:10] @ dequant_coef + dequant_intercept

    print("\n Inference Test (the first 10 samples are..):")
    print("Original predictions (sklearn):", original_pred)
    print("Manual original predictions:   ", manual_original_pred)
    print("Manual dequant predictions:    ", manual_dequant_pred)

    print("\n Differences:")
    print("Sklearn vs manual original:", np.abs(original_pred - manual_original_pred))
    print("Original vs dequant manual: ", np.abs(manual_original_pred - manual_dequant_pred))

    original_vs_dequant_diff = np.abs(original_pred - manual_dequant_pred)
    print("Absolute differences:", original_vs_dequant_diff)
    print("Max difference:", original_vs_dequant_diff.max())
    print("Mean difference:", original_vs_dequant_diff.mean())

    max_diff = original_vs_dequant_diff.max()
    if max_diff < 0.1:
        print(f"Quantization quality is good (max diff: {max_diff:.6f})")
    elif max_diff < 1.0:
        print(f"Quantization quality is acceptable (max diff: {max_diff:.6f})")
    else:
        print(f"Quantization quality is poor (max diff: {max_diff:.6f})")


if __name__ == "__main__":
    main()