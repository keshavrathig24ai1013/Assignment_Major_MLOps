# Major\_Exam\_MLOps_G24AI1013

# Major Exam – MLOps Pipeline with Linear Regression

## Objective
Develop a complete MLOps pipeline focused on **Linear Regression**, integrating the following components:
- Model training
- Unit testing
- Model quantization
- Dockerization
- CI/CD automation

All components are managed within a single `main` branch.

---

## Step 1: Project Initialization
- Initialized the `main` branch.
- Added the following project setup files:
  - `.gitignore` – to exclude unnecessary files from version control.
  - `requirements.txt` – to define Python dependencies.
  - `README.md` – for project documentation.

---

## Step 2: Model Training (`src/train.py`)
- Loaded the **California Housing Dataset** using `sklearn.datasets`.
- Created and trained a **Linear Regression** model with `sklearn.linear_model`.
- Saved the trained model using `joblib` into the `models/` directory.

### Console Output
## Model Evaluation Metrics

| Metric        | Value     |
|---------------|-----------|
| R² Score      | 0.5758    |
| MSE           | 0.5559    |
| Model Size    | 1.7 KB    |


| R² Score | Loss   |
|----------|--------|
| 0.5758   | 0.5559 |

## Step 3: Unit Testing (`tests/test_train.py`)

- Created unit tests to verify each component of the training pipeline.
- The test suite covers:
  - Dataset loading
  - Model initialization
  - Training process
  - Evaluation metrics
  - Model saving and reloading

### Running the Tests
Run the following command from the project root:
python tests/test_train.py
rootdir: /Documents/Github/Major_Exam_MLOps
collected 5 items

tests/test_train.py::TestTraining::test_dataset_loading PASSED    [ 20%]
tests/test_train.py::TestTraining::test_model_creation PASSED     [ 40%]
tests/test_train.py::TestTraining::test_model_training PASSED     [ 60%]
tests/test_train.py::TestTraining::test_model_performance PASSED  [ 80%]
tests/test_train.py::TestTraining::test_model_save_load PASSED    [100%]


========================= 5 passed in 0.28s =========================

## Step 4: Manual Quantization (`src/quantize.py`)

To optimize the model for deployment, **manual quantization** was applied to reduce model size and ensure efficient storage without sacrificing performance.

### What Was Done
- Loaded the trained `LinearRegression` model.
- Manually quantized the model's coefficients and intercept.
- Saved the quantized parameters to `models/quant_params.joblib`.
- Evaluated precision loss through a quality check and confirmed minimal error.

### Run Quantization
```bash```
python src/quantize.py

Loading trained model...
Original coefficients shape: (8,)
Original intercept: -37.023278
Original coef values: [ 4.48674910e-01  9.72425752e-03 -1.23323343e-01  7.83144907e-01
 -2.02962058e-06 -3.52631849e-03 -4.19792487e-01 -4.33708065e-01]
Quantizing intercept...
Intercept scale factor: 5.40
Quantized parameters saved to models/quant_params.joblib
Max coefficient error: 0.00000002
Intercept error: 0.00000042

## Evaluation Metrics

| Metric                   | Value        |
|--------------------------|--------------|
| R² Score                 | 0.5758       |
| Mean Squared Error (MSE) | 0.5559       |
| Quantized Model Size     | 0.9 KB       |

Inference Test (first 10 samples):

Original predictions (sklearn): [0.71912284 1.76401657 2.70965883 2.83892593 2.60465725 2.01175367

 2.64550005 2.16875532 2.74074644 3.91561473]

Manual original predictions:    [0.71912284 1.76401657 2.70965883 2.83892593 2.60465725 2.01175367

 2.64550005 2.16875532 2.74074644 3.91561473]

Manual dequant predictions:     [0.71912454 1.76401826 2.70966059 2.83892763 2.60465899 2.01175536

 2.64550173 2.168757   2.7407482  3.91561644]

Sklearn vs manual original:     [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]

Original vs dequant manual:     [1.69724930e-06 1.69119024e-06 1.75485833e-06 1.70069523e-06

 1.73872417e-06 1.68434987e-06 1.68467863e-06 1.67666992e-06

 1.75952921e-06 1.71352492e-06]




## 🐳 Step 5: Inference & Docker Integration

To run inference on the trained model:
- A `predict.py` script was created in the `src/` folder.
- A `Dockerfile` was added to containerize the complete pipeline.
- The workflow was executed via GitHub Actions on each push.

### ▶️ Run the Prediction Script
```bash```
python src/predict.py

Loading trained model.
Loading test dataset.
Making predictions.

Model Performance:
R² Score: 0.5758
Mean Squared Error: 0.5559

Sample Predictions (first 10):
True: 0.48 | Predicted: 0.72 | Diff: 0.24
True: 0.46 | Predicted: 1.76 | Diff: 1.31
True: 5.00 | Predicted: 2.71 | Diff: 2.29
True: 2.19 | Predicted: 2.84 | Diff: 0.65
True: 2.78 | Predicted: 2.60 | Diff: 0.18
True: 1.59 | Predicted: 2.01 | Diff: 0.42
True: 1.98 | Predicted: 2.65 | Diff: 0.66
True: 1.57 | Predicted: 2.17 | Diff: 0.59
True: 3.40 | Predicted: 2.74 | Diff: 0.66
True: 4.47 | Predicted: 3.92 | Diff: 0.55

## 6. Below is the performance comparison table

## Performance Comparison Table

| Metric     | Original Model | Quantized Model | Difference |
| ---------- | -------------- | --------------- | ---------- |
| R² Score   | 0.5758         | 0.5758          | 0.0000     |
| MSE        | 0.5559         | 0.5559          | 0.0000     |
| Model Size | 1.7 KB         | 0.9 KB          | -0.8 KB    |


## 7. Below are the Docker commands used in ci.yml and on every push this CI/CD workflow will start building the image then create and start the container.

docker build -t major-exam-mlops .
docker run --rm major-exam-mlops
