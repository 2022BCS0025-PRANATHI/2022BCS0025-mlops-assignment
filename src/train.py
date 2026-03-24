import pandas as pd
import mlflow
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import json

ROLL_NO = "2022BCS0025"
NAME = "Mantri Pranathi"

# ===== CONFIG =====
RUN_NAME = "final_model"
N_ESTIMATORS = 100
# ==================

# Load data
df = pd.read_csv("data/data.csv")

# Encode target
df["target"] = df["species"].astype("category").cat.codes
df = df.drop("species", axis=1)

# Use ALL FEATURES (IMPORTANT)
X = df.drop("target", axis=1)
y = df["target"]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# MLflow
mlflow.set_experiment(f"{ROLL_NO}_experiment")

with mlflow.start_run(run_name=RUN_NAME):

    model = RandomForestClassifier(n_estimators=N_ESTIMATORS)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="macro")

    # Log params
    mlflow.log_param("model", "RandomForest")
    mlflow.log_param("n_estimators", N_ESTIMATORS)
    mlflow.log_param("features", "all")

    # Log metrics
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1", f1)

    # Save model
    joblib.dump(model, "model.pkl")

    # Save metrics (MANDATORY for assignment)
    with open("metrics.json", "w") as f:
        json.dump({
            "accuracy": acc,
            "f1": f1,
            "name": NAME,
            "roll_no": ROLL_NO
        }, f)

print("Training complete ")