# Step 4: Import Required Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Step 5: Load the Dataset
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

# Step 6: Display the Dataset
df.head()

# Step 7: Display Dataset Information
df.info()

# Step 8: Check Missing Values
df.isnull().sum()

# Step 9: Select Required Columns
df = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Survived']]
df.head()

# Step 10: Separate Input and Output Variables
X = df.drop('Survived', axis=1)
y = df['Survived']

# Step 11: Identify Numerical Features
numeric_features = ['Age', 'Fare', 'SibSp', 'Parch']

# Step 12: Identify Categorical Features
categorical_features = ['Sex', 'Embarked', 'Pclass']

# Step 13: Handle Missing Values and Scale Numerical Data
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# Step 14: Encode Categorical Variables
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

# Step 15: Combine the Transformations
preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

# Step 16: Split the Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

# Step 17: Apply Feature Engineering
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Step 18: Display Dataset Shape
print("Original Shape :", X_train.shape)
print("Processed Shape :", X_train_processed.shape)

# Step 19: Display Success Message
print("Feature Engineering Completed Successfully")