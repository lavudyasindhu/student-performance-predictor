
import unittest
from unittest.mock import MagicMock, patch

# Mock the saved model and scaler before importing the Flask app.
with patch(
    "tensorflow.keras.models.load_model",
    return_value=MagicMock()
), patch(
    "joblib.load",
    return_value=MagicMock()
):
    import app as student_app


class StudentAppTests(unittest.TestCase):
    def setUp(self):
        student_app.app.config["TESTING"] = True
        self.client = student_app.app.test_client()

    def test_home_page_loads(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)

    def test_invalid_attendance_is_rejected(self):
        response = self.client.post(
            "/",
            data={
                "hours": "5",
                "attendance": "150",
                "assignments": "3"
            }
        )

        self.assertEqual(response.status_code, 200)
        self.assertIn(
            b"Attendance must be between 0 and 100",
            response.data
        )


if __name__ == "__main__":
    unittest.main()
