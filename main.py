import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import datetime

# 1. 数据加载
train_path = 'train.csv'
test_path = 'test.csv'
sample_submission_path = 'sample_submission.csv'

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)
sample_submission_df = pd.read_csv(sample_submission_path)

# 2. 填补缺失值
train_df['fuel_type'] = train_df['fuel_type'].fillna(train_df['fuel_type'].mode()[0])
test_df['fuel_type'] = test_df['fuel_type'].fillna(test_df['fuel_type'].mode()[0])

train_df['accident'] = train_df['accident'].fillna('None reported')
test_df['accident'] = test_df['accident'].fillna('None reported')

train_df['clean_title'] = train_df['clean_title'].fillna('Unknown')
test_df['clean_title'] = test_df['clean_title'].fillna('Unknown')

# 3. One-Hot Encoding
train_df_encoded = pd.get_dummies(train_df, columns=['brand', 'model', 'fuel_type', 'transmission', 'ext_col', 'int_col', 'accident', 'clean_title'])
test_df_encoded = pd.get_dummies(test_df, columns=['brand', 'model', 'fuel_type', 'transmission', 'ext_col', 'int_col', 'accident', 'clean_title'])

# 4. 特征工程：创建每年行驶的平均里程数
current_year = datetime.datetime.now().year

# 处理 'milage_per_year' 列中可能出现的除以零情况
train_df_encoded['milage_per_year'] = train_df['milage'] / (current_year - train_df['model_year']).replace(0, 1)
test_df_encoded['milage_per_year'] = test_df['milage'] / (current_year - test_df['model_year']).replace(0, 1)

# 5. 提取 'engine' 列中的马力和引擎排量
train_df_encoded['horsepower'] = train_df['engine'].str.extract(r'(\d+\.\d+|\d+)HP').astype(float)
train_df_encoded['engine_size'] = train_df['engine'].str.extract(r'(\d+\.\d+)L').astype(float)

test_df_encoded['horsepower'] = test_df['engine'].str.extract(r'(\d+\.\d+|\d+)HP').astype(float)
test_df_encoded['engine_size'] = test_df['engine'].str.extract(r'(\d+\.\d+)L').astype(float)

# 填补提取的数值缺失值
train_df_encoded['horsepower'] = train_df_encoded['horsepower'].fillna(train_df_encoded['horsepower'].mean())
train_df_encoded['engine_size'] = train_df_encoded['engine_size'].fillna(train_df_encoded['engine_size'].mean())

test_df_encoded['horsepower'] = test_df_encoded['horsepower'].fillna(test_df_encoded['horsepower'].mean())
test_df_encoded['engine_size'] = test_df_encoded['engine_size'].fillna(test_df_encoded['engine_size'].mean())

# 6. 删除原始 'engine' 列（包含复杂字符串）
train_df_encoded = train_df_encoded.drop(columns=['engine'])
test_df_encoded = test_df_encoded.drop(columns=['engine'])

# 7. 分离特征和目标变量
X = train_df_encoded.drop(columns=['price', 'id'])
y = train_df_encoded['price']

# 8. 拆分训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 9. 转换为数值格式，确保所有列都是数值类型，并处理无穷大和 NaN 值
X_train = X_train.replace([np.inf, -np.inf], np.nan).dropna()
X_val = X_val.replace([np.inf, -np.inf], np.nan).dropna()

# 10. XGBoost 模型训练
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
xgb_model.fit(X_train, y_train)

# 验证集预测
y_pred_xgb = xgb_model.predict(X_val)

# 计算 XGBoost 的 RMSE
rmse_xgb = np.sqrt(mean_squared_error(y_val, y_pred_xgb))
print(f"XGBoost RMSE: {rmse_xgb}")

# 11. 线性回归模型训练（可选，之前已实现）
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

# 验证集预测
y_pred_linear = linear_model.predict(X_val)

# 计算线性回归的 RMSE
rmse_linear = np.sqrt(mean_squared_error(y_val, y_pred_linear))
print(f"Linear Regression RMSE: {rmse_linear}")

# 12. 随机森林模型训练（可选，之前已实现）
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# 验证集预测
y_pred_rf = rf_model.predict(X_val)

# 计算随机森林的 RMSE
rmse_rf = np.sqrt(mean_squared_error(y_val, y_pred_rf))
print(f"Random Forest RMSE: {rmse_rf}")

# 1. 使用 XGBoost 模型预测测试集价格
X_test = test_df_encoded.drop(columns=['id'])

# 预测价格
test_predictions = xgb_model.predict(X_test)

# 2. 创建提交文件
submission_df = pd.DataFrame({
    'id': test_df['id'],
    'price': test_predictions
})

# 3. 导出为 CSV 文件
submission_df.to_csv('submission.csv', index=False)

print("提交文件已生成: submission.csv")