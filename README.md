# COMP3610 – Assignment 2

## NYC Taxi Data Analysis and Machine Learning

### Overview

This project analyzes the NYC Yellow Taxi dataset to predict taxi tipping behavior using machine learning models. Two predictive tasks were implemented:

- **Regression:** Predict the tip amount for a trip.
- **Classification:** Predict whether a trip will result in a high tip (tip > 20% of fare).

The project includes data preprocessing, feature engineering, model training, hyperparameter tuning, neural network implementation, and model interpretation.

### Dataset

NYC Yellow Taxi trip data (January 2024).

- Initial size: 2,964,624 rows, 19 columns
- After cleaning: 2,754,427 valid trips

Cleaning steps:

- Removed rows with missing values
- Removed invalid trips
- Removed trips where dropoff occurred before pickup

### Feature Engineering

Additional features created:

- trip_duration_minutes
- trip_speed_mph
- pickup_hour
- pickup_weekday
- pickup_day_of_week
- is_weekend
- log_trip_distance
- fare_per_mile
- fare_per_minute
- pickup_borough
- dropoff_borough

Categorical features were one-hot encoded.

### Target Variables

- **Regression:** tip_amount (continuous)
- **Classification:** high_tip (binary: 1 if tip_amount > 20% of fare_amount, else 0)

### Data Split

- Training: 1,607,057 samples
- Validation: 344,369 samples
- Test: 344,370 samples
- Stratified sampling for classification

### Models Implemented

**Regression:**

- Linear Regression
- Random Forest Regressor

**Classification:**

- Logistic Regression
- Random Forest Classifier
- Neural Network (PyTorch)

### Model Performance

#### Regression Results

| Model                   | MAE    | RMSE   | R²     |
| ----------------------- | ------ | ------ | ------ |
| Linear Regression       | 1.2435 | 2.3812 | 0.6150 |
| Random Forest Regressor | 1.2964 | 2.4553 | 0.5907 |

Best regression model: **Linear Regression**

#### Classification Results

| Model               | Accuracy | Precision | Recall | F1     | AUC-ROC |
| ------------------- | -------- | --------- | ------ | ------ | ------- |
| Logistic Regression | 0.7681   | 0.7688    | 0.9936 | 0.8668 | 0.6046  |
| Random Forest       | 0.7186   | 0.7728    | 0.8917 | 0.8280 | 0.5640  |
| Neural Network      | 0.7710   | 0.7685    | 0.9996 | 0.8690 | 0.6136  |

Best classification model: **Neural Network**

### Neural Network Architecture

- Input layer: 30 features
- Hidden layer 1: 128 neurons + ReLU, Dropout (0.3)
- Hidden layer 2: 64 neurons + ReLU, Dropout (0.3)
- Output layer: 1 neuron
- Loss: BCEWithLogitsLoss
- Optimizer: Adam
- Early stopping used

### Model Interpretation

**Logistic Regression Coefficients:**
Important predictors:

- log_trip_distance
- Airport_fee
- congestion_surcharge
- trip_duration_minutes
- fare_amount
- trip_speed_mph
- pickup/dropoff borough indicators

These features influence the probability of a trip being classified as a high-tip trip.

### Visualizations

- Training vs validation loss curves
- Training vs validation accuracy curves
- Predicted vs actual tip amounts (regression)
- Model comparison tables

---

**How to run:**

1. Install dependencies: `pip install -r requirements.txt`
2. Open and run `assignment2.ipynb` sequentially in Jupyter or VS Code.
