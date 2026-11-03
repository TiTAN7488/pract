# Step 1: Import libraries
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler 
from sklearn.linear_model import LinearRegression, Ridge, Lasso 
from sklearn.metrics import mean_squared_error, r2_score 

# Step 2: Load the Diabetes dataset
data = load_diabetes() 
X = pd.DataFrame(data.data, columns=data.feature_names) 
y = data.target 
print("Dataset Shape:", X.shape) 
print("\nMissing Values:\n", X.isnull().sum()) 

# Step 3: Check Multicollinearity
plt.figure(figsize=(8,5))
sns.heatmap(X.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()

# Step 4: Split the Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)

# Step 5: Standardize the Data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 6: Create Regression Models
models = {
    "Linear": LinearRegression(),
    "Ridge": Ridge(alpha=1),
    "Lasso": Lasso(alpha=0.1)
}

# Step 7: Train and Test the Models
for name, model in models.items():
    model.fit(X_train, y_train)
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    print("\n", name)
    print("Training MSE:", mean_squared_error(y_train, train_pred))
    print("Testing MSE :", mean_squared_error(y_test, test_pred))
    print("Training R2 :", r2_score(y_train, train_pred))
    print("Testing R2  :", r2_score(y_test, test_pred))

# Step 8: Compare Coefficients
coef = pd.DataFrame({
    "Feature": data.feature_names,
    "Linear": models["Linear"].coef_,
    "Ridge": models["Ridge"].coef_,
    "Lasso": models["Lasso"].coef_
})
print("\nCoefficient Comparison:")
print(coef)