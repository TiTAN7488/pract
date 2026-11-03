# Step 1: Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# Step 2: Load the Dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
columns = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
    "Outcome"
]
df = pd.read_csv(url, names=columns)
print(df.head())
print("\nDataset shape:", df.shape)

# Step 3: Separate Features and Target
X = df.drop("Outcome", axis=1)
y = df["Outcome"]
print("Features:")
print(X.columns)
print("\nTarget classes:")
print(y.unique())

# Step 4: Split the Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)
print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)

# Step 5: Standardize the Data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 6: Create Linear SVM
linear_svm = SVC(kernel="linear")
linear_svm.fit(X_train, y_train)
y_pred_linear = linear_svm.predict(X_test)

# Step 7: Create Nonlinear SVM (RBF Kernel)
rbf_svm = SVC(kernel="rbf")
rbf_svm.fit(X_train, y_train)
y_pred_rbf = rbf_svm.predict(X_test)

# Step 8: Evaluate Linear SVM
print("Linear SVM Performance")
print("----------------------")
print("Accuracy :", accuracy_score(y_test, y_pred_linear))
print("Precision:", precision_score(y_test, y_pred_linear))
print("Recall   :", recall_score(y_test, y_pred_linear))
print("F1-score :", f1_score(y_test, y_pred_linear))

# Step 9: Evaluate RBF SVM
print("\nRBF SVM Performance")
print("-------------------")
print("Accuracy :", accuracy_score(y_test, y_pred_rbf))
print("Precision:", precision_score(y_test, y_pred_rbf))
print("Recall   :", recall_score(y_test, y_pred_rbf))
print("F1-score :", f1_score(y_test, y_pred_rbf))

# Step 10: Compare Both Models
results = pd.DataFrame({
    "Metric": ["Accuracy", "Precision", "Recall", "F1-score"],
    "Linear SVM": [
        accuracy_score(y_test, y_pred_linear),
        precision_score(y_test, y_pred_linear),
        recall_score(y_test, y_pred_linear),
        f1_score(y_test, y_pred_linear)
    ],
    "RBF SVM": [
        accuracy_score(y_test, y_pred_rbf),
        precision_score(y_test, y_pred_rbf),
        recall_score(y_test, y_pred_rbf),
        f1_score(y_test, y_pred_rbf)
    ]
})
print(results)

# Step 11: Plot Performance Comparison
results.set_index("Metric").plot(
    kind="bar",
    figsize=(8, 5)
)
plt.title("Linear SVM vs RBF SVM")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.xticks(rotation=0)
plt.grid(axis="y")
plt.show()

# Step 12: Confusion Matrix – Linear SVM
cm_linear = confusion_matrix(y_test, y_pred_linear)
plt.figure(figsize=(5, 4))
sns.heatmap(
    cm_linear,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["No Diabetes", "Diabetes"],
    yticklabels=["No Diabetes", "Diabetes"]
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Linear SVM")
plt.show()

# Step 13: Confusion Matrix – RBF SVM
cm_rbf = confusion_matrix(y_test, y_pred_rbf)
plt.figure(figsize=(5, 4))
sns.heatmap(
    cm_rbf,
    annot=True,
    fmt="d",
    cmap="Greens",
    xticklabels=["No Diabetes", "Diabetes"],
    yticklabels=["No Diabetes", "Diabetes"]
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - RBF SVM")
plt.show()

# Step 14: Classification Reports
print("Linear SVM Classification Report")
print("--------------------------------")
print(classification_report(y_test, y_pred_linear))
print("\nRBF SVM Classification Report")
print("-----------------------------")
print(classification_report(y_test, y_pred_rbf))
