# Advanced-Analytical-Scorecards-for-Customer-Acquisition-Python

**Data Creation & Preprocessing**

A synthetic dataset is created with 5,000 customers, containing key attributes like age, income, credit score, loan amount, bureau score, repayment behavior, and external data scores.

The target variable (default_risk) is generated based on credit score, past defaults, and bureau scores, labeling customers as high risk (1) or low risk (0).

Categorical variables like loan purpose and repayment behavior are converted into numerical format.

**Splitting & Feature Scaling**

The dataset is split into training (80%) and testing (20%) sets.

Features are standardized using scaling techniques to improve model accuracy.

**Model Training (Random Forest)**

A Random Forest Classifier is used to train the model, leveraging an ensemble of decision trees to improve accuracy and robustness.

The model learns from customer financial history and repayment behavior to classify loan default risk.

**Model Evaluation**

The model predicts default risk for new customers.

Performance is evaluated using:

✔ AUC Score (measures model performance; higher is better).

✔ Classification Report (provides accuracy, precision, recall, and F1-score).
