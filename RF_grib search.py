import sklearn
from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import metrics
import numpy as np
import pandas as pd

# Load data
data = pd.read_csv("F:\Data\台风风雨训练数据集\Test for model.csv", delimiter=',', index_col=["ID", "Time"], low_memory=False)

# Drop missing values
data.dropna(axis=0, inplace=True)

# Define features and target variable
X = data[["Lat_sta", "Long_sta", "Lat", "Long", "cap", "mws", "Alt", "Distance", "GM", "GQ", "GX", "GD"]]
Y = data['PRCP']

# Split data into training and testing sets
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, train_size=0.9, random_state=1)

# Define the model
model = RandomForestRegressor(oob_score=True, random_state=0, n_jobs=-1)

# Define hyperparameters and grid search
param_grid = {
    'n_estimators': [50, 100],           # 决策树的数量
    'max_depth': [10, 30, 50, 70],    # 每棵树的最大深度
    'min_samples_split': [2, 5, 10],          # 分裂一个内部节点所需的最小样本数
    'min_samples_leaf': [1, 2, 4],            # 一个叶节点所需的最小样本数
}

grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(Xtrain, Ytrain)

# Get best model
best_model = grid_search.best_estimator_

# Predict and evaluate on test and training sets
Ytest_p = best_model.predict(Xtest)
Ytrain_p = best_model.predict(Xtrain)

# Prepare evaluation metrics
def calculate_metrics(Y, Y_p):
    mse = metrics.mean_squared_error(Y, Y_p)
    rmse = np.sqrt(mse)
    mae = metrics.mean_absolute_error(Y, Y_p)
    bias = np.sum(Y_p - Y) / Y.size
    r = np.corrcoef(Y, Y_p)[0, 1]
    sdo = np.std(Y)
    sdp = np.std(Y_p)
    ia = 1 - np.sum((Y_p - Y) ** 2) / np.sum((np.abs(Y_p - Y.mean()) + np.abs(Y - Y.mean())) ** 2)
    return round(mae, 3), round(rmse, 3), round(bias, 3), round(r, 3), round(sdo, 3), round(sdp, 3), round(ia, 3)

train_metrics = calculate_metrics(Ytrain, Ytrain_p)
test_metrics = calculate_metrics(Ytest, Ytest_p)

# Print metrics
print("训练集：")
print("mae={:.3f}\nrmse={:.3f}\nbias={:.3f}\nr={:.3f}\nsdo={:.3f}\nsdp={:.3f}\nia={:.3f}".format(*train_metrics))
print("测试集：")
print("mae={:.3f}\nrmse={:.3f}\nbias={:.3f}\nr={:.3f}\nsdo={:.3f}\nsdp={:.3f}\nia={:.3f}".format(*test_metrics))

# Save metrics to Excel
metrics_data = {
    "Dataset": ["Train", "Test"],
    "MAE": [train_metrics[0], test_metrics[0]],
    "RMSE": [train_metrics[1], test_metrics[1]],
    "Bias": [train_metrics[2], test_metrics[2]],
    "R": [train_metrics[3], test_metrics[3]],
    "SDO": [train_metrics[4], test_metrics[4]],
    "SDP": [train_metrics[5], test_metrics[5]],
    "IA": [train_metrics[6], test_metrics[6]]
}
metrics_df = pd.DataFrame(metrics_data)
metrics_df.to_excel("模型with_grid_search.xlsx", index=False)

# Save grid search results
results_df = pd.DataFrame(grid_search.cv_results_)
results_df.to_excel("grid_search_results.xlsx", index=False)
