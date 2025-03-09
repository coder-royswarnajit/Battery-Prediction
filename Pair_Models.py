import itertools
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor, AdaBoostRegressor, BaggingRegressor, StackingRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

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
    data['Temperature Factor'] = data['Operating Temp (°C)'].apply(get_temperature_factor)
    env_map = {'ideal': 1.0, 'low humidity/dust': 0.95, 'high humidity/dust': 0.925, 'extreme': 0.85}
    data['Environmental Factor'] = data['Environmental Conditions'].map(env_map).fillna(1.0)
    data['Battery_Life'] = (
        data['Battery_Capacity_mWh'] * data['Battery_Degradation_Factor'] *
        data['Temperature Factor'] * data['Environmental Factor']
    ) / data['Average_Power_Consumption']
    
    features = ['Battery_Degradation_Factor', 'Average_Power_Consumption', 'Battery_Capacity_mWh', 'Temperature Factor', 'Environmental Factor']
    target = 'Battery_Life'
    scaler = StandardScaler()
    X = scaler.fit_transform(data[features])
    y = data[target]
    
    return train_test_split(X, y, test_size=0.2, random_state=42)

def evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mse, r2

def train_combined_models(X_train, X_test, y_train, y_test):
    base_models = [
        ("Linear Regression", LinearRegression()),
        ("Lasso Regression", Lasso(alpha=0.01)),
        ("Ridge Regression", Ridge(alpha=1.0)),
        ("ElasticNet", ElasticNet(alpha=0.01, l1_ratio=0.5)),
        ("Decision Tree", DecisionTreeRegressor(random_state=42)),
        ("Random Forest", RandomForestRegressor(n_estimators=100, random_state=42)),
        ("Gradient Boosting", GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)),
        ("XGBoost", XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)),
        ("SVR", SVR(C=1.0, gamma='scale')),
        ("K-Nearest Neighbors", KNeighborsRegressor(n_neighbors=5)),
        ("AdaBoost Regressor", AdaBoostRegressor(n_estimators=50, random_state=42)),
        ("Bagging Regressor", BaggingRegressor(n_estimators=50, random_state=42)),
        ("Extra Trees", ExtraTreesRegressor(n_estimators=100, random_state=42)),
    ]
    
    for (name1, model1), (name2, model2) in itertools.combinations(base_models, 2):
        combined_model = StackingRegressor(estimators=[(name1, model1), (name2, model2)], final_estimator=Ridge(alpha=1.0))
        mse, r2 = evaluate_model(combined_model, X_train, X_test, y_train, y_test)
        print(f"Combined Model: {name1} + {name2}")
        print(f"  MSE = {mse:.4f}, R² = {r2:.4f}\n")

def main():
    X_train, X_test, y_train, y_test = load_and_preprocess_data('device_battery_data.xlsx')
    train_combined_models(X_train, X_test, y_train, y_test)

if __name__ == "__main__":
    main()