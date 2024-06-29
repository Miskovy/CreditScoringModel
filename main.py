import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Load the dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
columns = ["Status of existing checking account", "Duration in month", "Credit history", "Purpose", "Credit amount", 
           "Savings account/bonds", "Present employment since", "Installment rate in percentage of disposable income", 
           "Personal status and sex", "Other debtors / guarantors", "Present residence since", "Property", "Age in years", 
           "Other installment plans", "Housing", "Number of existing credits at this bank", "Job", "Number of people being liable to provide maintenance for", 
           "Telephone", "foreign worker", "Good/Bad"]

data = pd.read_csv(url, delimiter=' ', header=None, names=columns)

# Explore the data
print(data.head())

# Preprocess the data
# Encoding categorical variables
label_encoders = {}
for column in data.columns:
    if data[column].dtype == object:
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])
        label_encoders[column] = le

# Splitting the data
X = data.drop("Good/Bad", axis=1)
y = data["Good/Bad"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model Training and Evaluation
models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "Gradient Boosting": GradientBoostingClassifier()
}

results = {}

for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    results[model_name] = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred),
        "ROC AUC": roc_auc_score(y_test, y_pred)
    }

# Display the results
results_df = pd.DataFrame(results).T
print(results_df)

# Select the best model
best_model_name = results_df["Accuracy"].idxmax()
best_model = models[best_model_name]

print(f"The best model is: {best_model_name}")
