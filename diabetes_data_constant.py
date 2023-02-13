
import numpy as np
import tensorflow as tf
import tensorflow_transform as tft

LABEL_KEY = "Outcome"
FEATURE_KEY = ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']


def transformed_name(key):
    """Rename transformed features"""

    return key + "_tn"
