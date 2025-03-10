# Battery Life Prediction using Machine Learning

## Project Overview
This project aims to predict battery life for various electronic devices using machine learning techniques. The dataset is preprocessed to extract key features, and multiple regression models are trained to estimate battery life accurately. The project includes single-model training and pairwise ensemble model training for better predictive performance.

## Project Structure
- `device_battery_data.xlsx`: Dataset containing battery performance metrics for different devices.
- `Single_Models.py`: Script to train and evaluate individual machine learning models.
- `Pair_Models.py`: Script to train and evaluate ensemble models using pairwise stacking regressors.

## Features and Preprocessing
The dataset includes features such as:
- **Battery Capacity (mAh)** and **Battery Voltage (V)** to calculate total battery energy.
- **Charging Cycles**, **Device Age**, and **Battery Lifespan** to compute battery degradation.
- **Operating Temperature** and **Environmental Conditions** to adjust for external factors.
- **Power Consumption in Active and Sleep Modes** to estimate daily energy usage.

Feature engineering steps include:
- Computing `Battery_Degradation_Factor` using charging cycles and device age.
- Calculating `Average_Power_Consumption` based on usage patterns.
- Scaling features using `StandardScaler`.
- Handling missing values and removing outliers using IQR filtering.

## Models Implemented
### Single Model Training (`Single_Models.py`)
The following models are trained and evaluated using cross-validation:
- Linear Regression, Lasso, Ridge, ElasticNet
- Decision Tree, Random Forest, Gradient Boosting, XGBoost
- Support Vector Regression, K-Nearest Neighbors
- AdaBoost, Bagging, Extra Trees

Some models are fine-tuned using `RandomizedSearchCV` for hyperparameter optimization.

### Pairwise Ensemble Training (`Pair_Models.py`)
- Pairs of models are combined using `StackingRegressor` to improve performance.
- Ridge Regression is used as the final estimator.
- Each model pair is evaluated on MSE, MAE, and R² scores.

## Running the Code
Ensure you have the required dependencies installed:
```bash
pip install numpy pandas scikit-learn xgboost joblib openpyxl
```

To run single-model training:
```bash
python Single_Models.py
```
To run pairwise model training:
```bash
python Pair_Models.py
```

## Evaluation Metrics
Models are evaluated using:
- **Mean Squared Error (MSE)**: Measures prediction error.
- **Mean Absolute Error (MAE)**: Measures average absolute error.
- **R² Score**: Measures the proportion of variance explained by the model.

## Future Improvements
- Implement deep learning models using TensorFlow/PyTorch.
- Experiment with Quantum Neural Networks (QNN) using Qiskit.
- Incorporate real-time battery performance monitoring.


