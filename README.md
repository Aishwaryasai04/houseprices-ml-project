# houseprices-ml-project
# 🏡 Housing Price Prediction - Machine Learning Project

This project is a complete end-to-end Machine Learning pipeline to predict housing prices using the [Ames Housing Dataset](https://www.kaggle.com/datasets/prevek18/house-price-prediction) from Kaggle. It includes exploratory data analysis (EDA), data preprocessing, feature engineering, model building, and evaluation.

---

## 📁 Dataset Used
- **Train & Test CSV**: From Kaggle (manually downloaded)
- **Shape of dataset**: 1460 rows × 81 columns

---

## 🧠 ML Algorithms Used
- Linear Regression
- Random Forest Regressor

---

## 📊 Workflow

### ✅ Step 1: Data Loading & Initial Exploration
- Checked shape and data types
- Printed sample rows using `.head()`

### ✅ Step 2: Data Cleaning
- Handled missing values using `.fillna()` (median or most frequent)
- Dropped unnecessary columns with high missing data

### ✅ Step 3: Feature Engineering
- Converted categorical variables using `pd.get_dummies`
- Removed ID column

### ✅ Step 4: Correlation Analysis
- Visualized top 10 features highly correlated with `SalePrice`

### ✅ Step 5: Model Training & Evaluation
- Trained:
  - `LinearRegression` model
  - `RandomForestRegressor` model
- Evaluated using:
  - R² Score
  - Mean Squared Error

### ✅ Step 6: Feature Importance (Random Forest)
- Visualized top 10 important features that influenced price prediction

---

## 📈 Results

| Model                  | R² Score | MSE (lower is better) |
|------------------------|----------|------------------------|
| Linear Regression      | ~0.85    | [Your_Value]           |
| Random Forest Regressor| ~0.89    | [Your_Value]           |

> ✅ Random Forest performed slightly better due to handling non-linearity and feature interactions.

---

## 🗃️ Files in This Repo

- `housing_model.ipynb` – Colab notebook with all code
- `rf_predictions.csv` – Actual vs Predicted results (optional)
- `README.md` – This file

---

## ✅ Skills Used
- Python
- Pandas, NumPy, Matplotlib, Seaborn
- Scikit-Learn (train_test_split, LinearRegression, RandomForestRegressor)
- Data Preprocessing & EDA
- GitHub Project Documentation

---

## 📚 Future Improvements
- Hyperparameter tuning
- Use of XGBoost/CatBoost
- Deployment via Streamlit or Flask
