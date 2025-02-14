import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
import random

# 🔹 Step 1: Generate Dummy Data
np.random.seed(42)

num_customers = 5000

# Creating a customer dataset with key attributes
customer_data = pd.DataFrame({
    'customer_id': range(1, num_customers + 1),
    'age': np.random.randint(18, 70, num_customers),
    'income': np.random.randint(20000, 200000, num_customers),
    'credit_score': np.random.randint(300, 850, num_customers),
    'loan_amount': np.random.randint(5000, 500000, num_customers),
    'loan_term': np.random.choice([12, 24, 36, 48, 60], num_customers),
    'previous_defaults': np.random.randint(0, 5, num_customers),
    'loan_purpose': np.random.choice(['Personal', 'Home', 'Auto', 'Education', 'Business'], num_customers),
    'bureau_score': np.random.randint(300, 900, num_customers),
    'external_data_score': np.random.randint(1, 100, num_customers),
    'repayment_behavior': np.random.choice(['Good', 'Average', 'Poor'], num_customers)
})

# 🔹 Step 2: Define Target Variable
# 1 = High Risk (default), 0 = Low Risk (no default)
customer_data['default_risk'] = np.where(
    (customer_data['credit_score'] < 500) | 
    (customer_data['previous_defaults'] > 2) | 
    (customer_data['bureau_score'] < 450), 1, 0
)

# 🔹 Step 3: Convert Categorical Variables to Numeric
customer_data = pd.get_dummies(customer_data, columns=['loan_purpose', 'repayment_behavior'], drop_first=True)

# 🔹 Step 4: Split Data into Training & Testing Sets
X = customer_data.drop(columns=['customer_id', 'default_risk'])  # Features
y = customer_data['default_risk']  # Target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 🔹 Step 5: Standardize Features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 🔹 Step 6: Train a Machine Learning Model (Random Forest)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)

# 🔹 Step 7: Evaluate Model Performance
y_pred = rf_model.predict(X_test_scaled)
y_pred_prob = rf_model.predict_proba(X_test_scaled)[:, 1]

auc_score = roc_auc_score(y_test, y_pred_prob)
classification_rep = classification_report(y_test, y_pred)

# 🔹 Print Model Performance
print(f"📊 Model AUC Score: {auc_score:.2f}")
print("\n🔹 Classification Report:\n", classification_rep)

# 🔹 Step 8: Save Processed Data & Model Outputs
customer_data.to_csv("customer_data.csv", index=False)
X_train.to_csv("X_train.csv", index=False)
X_test.to_csv("X_test.csv", index=False)
pd.DataFrame(y_train, columns=['default_risk']).to_csv("y_train.csv", index=False)
pd.DataFrame(y_test, columns=['default_risk']).to_csv("y_test.csv", index=False)
