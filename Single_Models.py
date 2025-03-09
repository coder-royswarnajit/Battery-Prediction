import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, ExtraTreesRegressor, StackingRegressor, AdaBoostRegressor, BaggingRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from joblib import Parallel, delayed
from scipy.optimize import minimize
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform

def get_alpha(category):
    alpha_values = {'Laptop': 0.80, 'Smartwatch': 0.70, 'Tablet': 0.70, 'Smartphone': 0.75, 'Gaming console': 0.85}
    return alpha_values.get(category, 0.75)

def get_beta(category):
    beta_values = {'Laptop': 0.20, 'Smartwatch': 0.30, 'Tablet': 0.30, 'Smartphone': 0.25, 'Gaming console': 0.15}
    return beta_values.get(category, 0.25)

def get_temperature_factor(temp):
    try:
        temp = float(temp)
    except (ValueError, TypeError):
        return 1.0
    if 20 <= temp <= 30:
        return 1.0
    elif temp > 35:
        return 0.85
    elif temp < 0:
        return 0.65
    else:
        return 1.0

def load_and_preprocess_data(file_path):
    data = pd.read_excel(file_path)
    data['alpha'] = data['Category'].apply(get_alpha)
    data['beta'] = data['Category'].apply(get_beta)
    data['Battery_Degradation_Factor'] = (
        1 - (data['alpha'] * (data['Charging_Cycles'] / data['Maximum Cycles'])) -
        (data['beta'] * (data['Device Age (years)'] / data['Battery Lifespan (years)']))
    )
    data['Average_Power_Consumption'] = (
        (data['Active Time per Day (hours)'] * data['Active Power Consumption (mW)']) +
        (data['Sleep Time per Day (hours)'] * data['Sleep Power Consumption (mW)'])
    ) / 24
    data['Battery_Capacity_mWh'] = data['Battery Capacity (mAh)'] * data['Battery Voltage (V)']
    data['Operating Temp (°C)'] = pd.to_numeric(data['Operating Temp (°C)'], errors='coerce')
    data['Temperature Factor'] = data['Operating Temp (°C)'].apply(get_temperature_factor)
    env_map = {'ideal': 1.0, 'low humidity/dust': 0.95, 'high humidity/dust': 0.925, 'extreme': 0.85}
    data['Environmental Conditions'] = data['Environmental Conditions'].str.lower().str.strip()
    data['Environmental Factor'] = data['Environmental Conditions'].map(env_map).fillna(1.0)
    data['Battery_Life'] = (
        data['Battery_Capacity_mWh'] * data['Battery_Degradation_Factor'] *
        data['Temperature Factor'] * data['Environmental Factor']
    ) / data['Average_Power_Consumption']
    Q1, Q3 = data['Battery_Life'].quantile([0.25, 0.75])
    IQR = Q3 - Q1
    data = data[(data['Battery_Life'] >= (Q1 - 1.5 * IQR)) & (data['Battery_Life'] <= (Q3 + 1.5 * IQR))]
    features = ['Battery_Degradation_Factor', 'Average_Power_Consumption', 'Battery_Capacity_mWh', 'Temperature Factor', 'Environmental Factor']
    target = 'Battery_Life'
    scaler = StandardScaler()
    X = scaler.fit_transform(data[features])
    y = data[target]
    return train_test_split(X, y, test_size=0.2, random_state=42)

def fine_tune_model(model, param_grid, X_train, y_train):
    random_search = RandomizedSearchCV(
        model, param_grid, n_iter=20, cv=5, n_jobs=-1, random_state=42, scoring='neg_mean_squared_error'
    )
    random_search.fit(X_train, y_train)
    return random_search.best_estimator_

def train_models(X_train, X_test, y_train, y_test):
    models = [
        ("Linear Regression", LinearRegression()),
        ("Lasso Regression", Lasso(alpha=0.01)),
        ("Ridge Regression", Ridge(alpha=1.0)),
        ("ElasticNet", fine_tune_model(ElasticNet(), {'alpha': [0.01, 0.1], 'l1_ratio': [0.2, 0.5, 0.8]}, X_train, y_train)),
        ("Decision Tree", DecisionTreeRegressor(random_state=42)),
        ("Random Forest", fine_tune_model(RandomForestRegressor(random_state=42), {'n_estimators': [100, 200], 'max_depth': [None, 10, 20]}, X_train, y_train)),
        ("Gradient Boosting", fine_tune_model(GradientBoostingRegressor(random_state=42), {'n_estimators': [100, 200], 'learning_rate': [0.01, 0.1]}, X_train, y_train)),
        ("XGBoost", fine_tune_model(XGBRegressor(random_state=42), {'n_estimators': [100, 200], 'learning_rate': [0.01, 0.1]}, X_train, y_train)),
        ("Support Vector Regressor", fine_tune_model(SVR(), {'C': [1, 10], 'gamma': ['scale', 'auto']}, X_train, y_train)),
        ("K-Nearest Neighbors", fine_tune_model(KNeighborsRegressor(), {'n_neighbors': [3, 5, 7]}, X_train, y_train)),
        ("AdaBoost Regressor", AdaBoostRegressor(n_estimators=50, random_state=42)),
        ("Bagging Regressor", BaggingRegressor(n_estimators=50, random_state=42)),
        ("Extra Trees", ExtraTreesRegressor(n_estimators=100, random_state=42)),
    ]
    Parallel(n_jobs=-1)(delayed(evaluate_model)(name, model, X_train, X_test, y_train, y_test) for name, model in models)

def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"{name}: MSE = {mse:.4f}, R² = {r2:.4f}")

def main():
    X_train, X_test, y_train, y_test = load_and_preprocess_data('device_battery_data.xlsx')
    train_models(X_train, X_test, y_train, y_test)

if __name__ == "__main__":
    main()
