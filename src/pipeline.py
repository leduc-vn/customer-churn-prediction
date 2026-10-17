import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
import joblib


def generate_synthetic_data(n_samples=1000):
    np.random.seed(42)
    tenure = np.random.randint(1, 72, n_samples)
    monthly_charges = np.random.uniform(20.0, 120.0, n_samples)
    contract_type = np.random.choice([0, 1, 2], n_samples)
    prob = 0.3 * (monthly_charges / 120.0) - 0.2 * (tenure / 72.0) + 0.1
    churn = (prob > 0.15).astype(int)
    df = pd.DataFrame({
        'tenure': tenure,
        'monthly_charges': monthly_charges,
        'contract_type': contract_type,
        'churn': churn,
    })
    return df


def train_model():
    df = generate_synthetic_data()
    X = df.drop('churn', axis=1)
    y = df['churn']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    proba = clf.predict_proba(X_test)[:, 1]
    print("Model Evaluation:")
    print(classification_report(y_test, preds))
    print(f"ROC-AUC: {roc_auc_score(y_test, proba):.3f}")
    joblib.dump(clf, 'model.joblib')
    print("Model saved to model.joblib")


if __name__ == "__main__":
    train_model()
