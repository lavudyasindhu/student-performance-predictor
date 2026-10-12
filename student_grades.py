import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import set_random_seed

# -----------------------------
# 1. Load Dataset
# -----------------------------

set_random_seed(42)

df = pd.read_csv("student_grades.csv")

print("Student Dataset:")
print(df)
print("\nGrade counts:")
print(df["grade"].value_counts())

# -----------------------------

# 2. Separate Input and Output
# -----------------------------

X = df[["hours_studied", "attendance", "assignments"]]

# CSV grade codes: 0=A, 1=B, 2=C, 3=D
grade_codes = [0, 1, 2, 3]
grade_names = ["A", "B", "C", "D"]

# Validate the dataset's grade values
if not df["grade"].isin(grade_codes).all():
    raise ValueError("Grade must contain only 0, 1, 2, or 3.")

y = df["grade"].astype(int)

print("Grade mapping:")
for code, name in zip(grade_codes, grade_names):
    print(f"{code} = {name}")


# -----------------------------
# 3. Train-Test Split
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

# -----------------------------
# 4. Data Preprocessing
# -----------------------------

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -----------------------------
# 5. Create ANN Model
# -----------------------------

model = Sequential([
    Input(shape=(3,)),

    Dense(16, activation="relu"),
    Dense(8, activation="relu"),

    Dense(4, activation="softmax")
])
# -----------------------------
# 6. Compile Model
# -----------------------------

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# -----------------------------
# 7. Train Model
# -----------------------------

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=20,
    restore_best_weights=True
)

history = model.fit(
    X_train,
    y_train,
    validation_split=0.2,
    epochs=200,
    batch_size=4,
    callbacks=[early_stop],
    verbose=1
)

# -----------------------------
# 8. Evaluate Model
# -----------------------------

loss, accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=0
)

print(f"\nTest Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")
print(f"Test Accuracy Percentage: {accuracy * 100:.2f}%")

# Save trained model and scaler
import os
import joblib

os.makedirs("model", exist_ok=True)

model.save("model/student_grade_model.keras")
joblib.dump(scaler, "model/scaler.pkl")

print("\nModel and scaler saved successfully!")

# Detailed test results
predictions = model.predict(X_test, verbose=0)
predicted_classes = np.argmax(predictions, axis=1)

# ADD THIS CODE

print("\nActual vs Predicted Grades:")

for actual, predicted in zip(y_test, predicted_classes):
    print(
        f"Actual: {grade_names[actual]}, "
        f"Predicted: {grade_names[predicted]}"
    )
print("\nClassification Report:")
print(classification_report(
    y_test,
    predicted_classes,
    labels=[0, 1, 2, 3],
    target_names=grade_names,
    zero_division=0
))

# -----------------------------
# 9. Predict New Student
# -----------------------------

while True:
    try:
        hours = float(input("\nEnter hours studied: "))
        attendance = float(input("Enter attendance percentage: "))
        assignments = float(input("Enter assignments completed: "))

        if hours < 0 or not 0 <= attendance <= 100 or assignments < 0:
            print("Enter valid non-negative values; attendance must be 0–100.")
            continue

        new_student = np.array([[hours, attendance, assignments]])
        new_student_scaled = scaler.transform(new_student)

        prediction = model.predict(new_student_scaled, verbose=0)
        grade_index = np.argmax(prediction, axis=1)[0]

        print("\nPredicted Grade:", grade_names[grade_index])
        print("Prediction probabilities:")
        for grade, probability in zip(grade_names, prediction[0]):
          print(f"{grade}: {probability * 100:.2f}%")

        break

    except ValueError:
        print("Please enter numeric values.")

