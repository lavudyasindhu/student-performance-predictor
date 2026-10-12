
import numpy as np
import joblib

from flask import Flask, render_template, request
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load the trained model and scaler
model = load_model("model/student_grade_model.keras")
scaler = joblib.load("model/scaler.pkl")

grades = ["A", "B", "C", "D"]


@app.route("/", methods=["GET", "POST"])
def home():
    predicted_grade = None
    error = None

    if request.method == "POST":
        try:
            hours = float(request.form["hours"])
            attendance = float(request.form["attendance"])
            assignments = float(request.form["assignments"])

            if hours < 0 or not 0 <= attendance <= 100 or assignments < 0:
                error = "Enter valid values. Attendance must be between 0 and 100."
            else:
                student = np.array([[hours, attendance, assignments]])
                student_scaled = scaler.transform(student)

                prediction = model.predict(student_scaled, verbose=0)
                grade_index = int(np.argmax(prediction))
                predicted_grade = grades[grade_index]

        except (ValueError, KeyError):
            error = "Please enter valid numeric values."

    return render_template(
        "index.html",
        predicted_grade=predicted_grade,
        error=error
    )


if __name__ == "__main__":
    app.run(debug=True)
