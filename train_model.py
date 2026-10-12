import os
import json
import joblib
import numpy as np
import pandas as pd
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import set_random_seed

# 1. Reproducibility
set_random_seed(42)
# MLflow experiment tracking
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Student Grade Prediction")
# 2. Load dataset
df = pd.read_csv("student_grades.csv")

required_columns = [
    "hours_studied",
    "attendance",
    "assignments",
    "grade"
]

if not all(column in df.columns for column in required_columns):
    raise ValueError(f"Dataset must contain these columns: {required_columns}")

if df.empty or df[required_columns].isnull().any().any():
    raise ValueError("Dataset is empty or contains missing values.")

grade_codes = [0, 1, 2, 3]
grade_names = ["A", "B", "C", "D"]

if not df["grade"].isin(grade_codes).all():
    raise ValueError("Grade must contain only 0, 1, 2, or 3.")

X = df[["hours_studied", "attendance", "assignments"]]
y = df["grade"].astype(int)

if y.value_counts().min() < 2:
    raise ValueError("Each grade needs at least two samples for stratified splitting.")

# 3. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

# 4. Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 5. Build ANN
model = Sequential([
    Input(shape=(3,)),
    Dense(16, activation="relu"),
    Dense(8, activation="relu"),
    Dense(4, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# 6. Train model
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=20,
    restore_best_weights=True
)

model.fit(
    X_train,
    y_train,
    validation_split=0.2,
    epochs=200,
    batch_size=4,
    callbacks=[early_stop],
    verbose=1
)

# 7. Evaluate model
predictions = model.predict(X_test, verbose=0)
predicted_classes = np.argmax(predictions, axis=1)

accuracy = accuracy_score(y_test, predicted_classes)
# Log evaluation metrics to MLflow
with mlflow.start_run():
    mlflow.log_metric("test_accuracy", float(accuracy))
    mlflow.log_param("test_samples", len(y_test))
    mlflow.log_param("random_seed", 42)
print(f"\nTest accuracy: {accuracy:.4f}")
print("\nClassification report:")
print(classification_report(
    y_test,
    predicted_classes,
    labels=grade_codes,
    target_names=grade_names,
    zero_division=0
))

# 8. Save model, scaler, and evaluation metrics
os.makedirs("model", exist_ok=True)

model.save("model/student_grade_model.keras")
joblib.dump(scaler, "model/scaler.pkl")

with open("metrics.json", "w", encoding="utf-8") as file:
    json.dump(
        {
            "accuracy": round(float(accuracy), 4),
            "test_samples": int(len(y_test))
        },
        file,
        indent=4
    )

print("\nModel, scaler, and metrics saved successfully.")