Here's a README file for your project:  

---

## Battery Life Prediction Models  

This project implements machine learning models to predict battery life for various electronic devices. The models use features such as battery degradation, environmental conditions, and power consumption to estimate battery lifespan.  

### Files  
- **`Pair_Models.py`**: Trains and evaluates combined (stacked) regression models.  
- **`Single_Models.py`**: Trains and evaluates individual regression models with hyperparameter tuning.  

### Features  
- **Preprocessing**: Loads and cleans data from an Excel file (`device_battery_data.xlsx`).  
- **Feature Engineering**: Computes battery degradation, environmental effects, and power consumption.  
- **Models Used**:  
  - Linear Regression, Lasso, Ridge, ElasticNet  
  - Decision Tree, Random Forest, Gradient Boosting, XGBoost  
  - SVR, K-Nearest Neighbors, Extra Trees, AdaBoost, Bagging  
  - Stacking (in `Pair_Models.py`)  

### Installation  
1. Install dependencies:  
   ```bash
   pip install numpy pandas scikit-learn xgboost joblib openpyxl
   ```  
2. Run a script:  
   ```bash
   python Single_Models.py
   ```  
   or  
   ```bash
   python Pair_Models.py
   ```  

### Dataset  
The scripts expect a dataset named **`device_battery_data.xlsx`** with features like device category, charging cycles, battery capacity, and environmental conditions.  

### Output  
Each script prints the Mean Squared Error (MSE) and R² score for each model.  

---

Would you like any modifications or additional sections?
