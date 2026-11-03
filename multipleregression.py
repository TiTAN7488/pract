# Step 1: Import Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Step 2: Load the Dataset
from sklearn.datasets import load_diabetes
diabetes = load_diabetes()
X = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
y = pd.Series(diabetes.target, name="DiseaseProgression")
print(X.head())
print(y.head())

# Step 3: Check Dataset Shape
print("Dataset Shape:", X.shape)
print("Number of Features:", X.shape[1])

# Step 4: Check for Missing Values
print(X.isnull().sum())

# Step 5: Split the Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)
print("Training Data:", X_train.shape)
print("Testing Data:", X_test.shape)

# Step 6: Create the Multiple Linear Regression Model
model = LinearRegression()

# Step 7: Train the Model
model.fit(X_train, y_train)
print("Model Training Completed")

# Step 8: Display the Coefficients
print("Intercept:", model.intercept_)
print("\nCoefficients:")
for feature, coefficient in zip(X.columns, model.coef_):
    print(feature, ":", coefficient)

# Step 9: Make Predictions
y_pred = model.predict(X_test)
print("Predicted Values:")
print(y_pred[:10])

# Step 10: Calculate MSE
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# Step 11: Calculate RMSE
rmse = np.sqrt(mse)
print("Root Mean Squared Error:", rmse)

# Step 12: Calculate R² Score
r2 = r2_score(y_test, y_pred)
print("R2 Score:", r2)

# Step 13: Display Actual vs Predicted Values
results = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred
})
print(results.head(10))