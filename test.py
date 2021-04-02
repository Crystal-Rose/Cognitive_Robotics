import subprocess
import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
import cv2
import random
from variables import ANNOTATIONS_PATH
from model import loadData

def predict():
    # Load model
    model = keras.models.load_model("trained_model")
    # Load data
    features, labels = loadData()
    print(features.shape)
    # Choose a file for a test
    index = random.randint(0, features.shape[0]-1)
    print("Chosen features: {}".format(features[index]))
    # Make a prediction
    predictions = model.predict(np.array([features[index]]))
    print("Expected label: {}".format(labels[index]))
    print("Predicted label: {}".format(predictions))

if __name__ == "__main__":
    predict()
