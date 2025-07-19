# Inventory Forecasting Model using Python (Simple Linear Regression)
# Author: Abednego Ndegwa
# Purpose: Predict upcoming inventory needs to reduce overstock/understock scenarios
# Ready for push to GitHub with clear, educational structure

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load data (replace with your CSV path or integrate with your warehouse data source)
data = pd.read_csv('inventory_data.csv')  # Expected columns: ['date', 'item_id', 'units_sold']

# Preprocess: aggregate daily sales to weekly for smoother forecasting
data['date'] = pd.to_datetime(data['date'])
data['week'] = data['date'].dt.isocalendar().week
weekly_data = data.groupby(['item_id', 'week'])['units_sold'].sum().reset_index()

# For simplicity, select one item to forecast
item_id = weekly_data['item_id'].unique()[0]
item_data = weekly_data[weekly_data['item_id'] == item_id]

# Prepare X (week) and y (units sold)
X = item_data[['week']]
y = item_data['units_sold']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")
print(f"R^2 Score: {r2:.2f}")

# Visualize
plt.scatter(X_test, y_test, color='blue', label='Actual')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Predicted')
plt.title(f'Inventory Forecasting for Item {item_id}')
plt.xlabel('Week')
plt.ylabel('Units Sold')
plt.legend()
plt.show()

# Predict next week's demand
next_week = np.array([[item_data['week'].max() + 1]])
next_week_pred = model.predict(next_week)
print(f"Predicted units for week {next_week[0][0]}: {next_week_pred[0]:.0f}")

# Save the model for future integration with a warehouse dashboard
import joblib
joblib.dump(model, 'inventory_forecasting_model.pkl')


