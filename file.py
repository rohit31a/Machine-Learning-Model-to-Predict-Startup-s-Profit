import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV

data = pd.read_csv("50_Startups.csv")

print(data.head())

X = data[['R&D Spend', 'Administration', 'Marketing Spend']]
y = data['Profit']

#splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y , test_size = 0.2, random_state = 42)

# Scale the features (important for KNN)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)
lr_mae = mean_absolute_error(y_test, lr_pred)
lr_mse = mean_squared_error(y_test, lr_pred)
lr_r2 = r2_score(y_test, lr_pred)

# ElasticNet Regression
en_model = ElasticNet()
en_params = {'alpha': [0.1, 1, 10], 'l1_ratio': [0.1, 0.5, 0.9]}
en_grid = GridSearchCV(en_model, en_params, cv=5)
en_grid.fit(X_train, y_train)
en_pred = en_grid.predict(X_test)
en_mae = mean_absolute_error(y_test, en_pred)
en_mse = mean_squared_error(y_test, en_pred)
en_r2 = r2_score(y_test, en_pred)

# KNN Regression
knn_model = KNeighborsRegressor()
knn_params = {'n_neighbors': [3, 5, 7], 'weights': ['uniform', 'distance']}
knn_grid = GridSearchCV(knn_model, knn_params, cv=5)
knn_grid.fit(X_train_scaled, y_train)
knn_pred = knn_grid.predict(X_test_scaled)
knn_mae = mean_absolute_error(y_test, knn_pred)
knn_mse = mean_squared_error(y_test, knn_pred)
knn_r2 = r2_score(y_test, knn_pred)

# Print model performance
print("Linear Regression: MAE =", lr_mae, ", MSE =", lr_mse, ", R2 =", lr_r2 , "\n")
print("ElasticNet Regression: MAE =", en_mae, ", MSE =", en_mse, ", R2 =", en_r2, "\n")
print("KNN Regression: MAE =", knn_mae, ", MSE =", knn_mse, ",R2 =", knn_r2, "\n")

# Plot settings
plt.figure(figsize=(18, 5))

# Linear Regression
plt.subplot(1, 3, 1)
plt.scatter(y_test, lr_pred, alpha=0.7, color='b')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Actual Profit')
plt.ylabel('Predicted Profit')
plt.title('Linear Regression')

# ElasticNet Regression
plt.subplot(1, 3, 2)
plt.scatter(y_test, en_pred, alpha=0.7, color='r')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Actual Profit')
plt.ylabel('Predicted Profit')
plt.title('ElasticNet Regression')

# KNN Regression
plt.subplot(1, 3, 3)
plt.scatter(y_test, knn_pred, alpha=0.7, color='g')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Actual Profit')
plt.ylabel('Predicted Profit')
plt.title('KNN Regression')

plt.tight_layout()
plt.show()

# Combined residuals plot
plt.figure(figsize=(12, 8))

plt.scatter(y_test, lr_pred, alpha=0.6, color='b', s=50, label='Linear Regression')
plt.scatter(y_test, en_pred, alpha=0.6, color='r', s=50, label='ElasticNet Regression')
plt.scatter(y_test, knn_pred, alpha=0.6, color='g', s=50, label='KNN Regression')

plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)

plt.xlabel('Actual Profit', fontsize=14)
plt.ylabel('Predicted Profit', fontsize=14)
plt.title('Actual vs Predicted Profit', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.show()