
import os

# Limit TensorFlow CPU thread usage
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"

import numpy as np
import joblib
import tensorflow as tf

from flask import Flask, render_template, request
from tensorflow.keras.models import load_model

tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

app = Flask(__name__)

model = load_model("model/student_grade_model.keras")
scaler = joblib.load("model/scaler.pkl")

grades = ["A", "B", "C", "D"]

