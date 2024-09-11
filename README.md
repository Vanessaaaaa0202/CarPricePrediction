# 🚗Car Price Prediction
This repository contains code and documentation for participating in the Kaggle competition: Used Car Price Prediction. The goal of the competition is to predict the price of used cars based on various attributes using machine learning models. The project is scored based on the Root Mean Squared Error (RMSE).
Project Overview
The dataset includes information such as car brand, model, year, mileage, fuel type, and engine details, which are used to predict the target variable: car price.

**📊Datasets:**  
train.csv: Training data with car attributes and price (target).  
test.csv: Test data where the car price needs to be predicted.  
sample_submission.csv: A sample submission file that indicates the format for predictions.

**💡Key Steps:**  
Data Preprocessing: Handle missing values, clean and normalize data.   
Feature Engineering: Transform categorical features and extract meaningful features.   
Modeling: Train regression models like XGBoost, Random Forest, and Linear Regression.  
Evaluation: Measure model performance using RMSE.  
Prediction: Generate predictions for the test set and submit to Kaggle.

**🧑‍💻Technologies Used:**  
Python (pandas, numpy, scikit-learn, XGBoost)
Jupyter Notebook / PyCharm
Git for version control

**How to Run:**  
Clone the repository:
git clone https://github.com/yourusername/used-car-price-prediction.git  

**Install dependencies:**
pip install -r requirements.txt  
Run the notebook or Python scripts for data preprocessing, modeling, and prediction.
