import subprocess
import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
import cv2
import json
from variables import ANNOTATIONS_PATH

def loadData():
    with open(ANNOTATIONS_PATH) as f:
        annotations = json.loads(f)
    
    features = list()
    labels = list()
    for data in annotations["data"]:
        labels.append([data["class"],data["side"]])
        features.append(data["features"])

    return np.array(features), np.array(labels)

# From Tensorflow documentation (https://www.tensorflow.org/tutorials/keras/regression)
def plot_loss(history):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.ylim([0, 10])
    plt.xlabel('Epoch')
    plt.ylabel('Error (mean absolute)')
    plt.legend()
    plt.grid(True)
    plt.show()


def train(csv, model_name):
    '''
    Returns the model trained on a provided dataset (.csv)
    '''
    features = loadData()

    # Data normalizer
    normalizer = preprocessing.Normalization()
    normalizer.adapt(np.array(features))

    # Visualize normalization effect
    # with np.printoptions(precision=2, suppress=True):
    #     print('First example:', np.array(df[:1]))
    #     print('Normalized:', normalizer(np.array(df[:1])).numpy())


    # Train the regression
    model = tf.keras.Sequential([
        # When the layer is called it returns the input data, with each feature independently normalized
        normalizer,
        layers.Dense(32, activation="relu")
        layers.Dense(2, activation="softmax")   # 2 output units: class, direction

    ])

    # Configure the trainng procedure
    model.compile(
        optimizer=tf.optimizers.Adam(learning_rate=0.01),
        loss='sparse_categorical_crossentropy',
        metrics=["accuracy"]
    )

    # Get a print-out summary of the model
    # model.summary()
    
    # Train and save the history (training process)
    history = model.fit(
        features, labels,
        epochs=100, 
        verbose = 1, # 0 to suppress logging
        # Calculate validation results on 20% of the training data
        validation_split = 0.2
    )
    print("Training Complete!\n")

    # Visualize training process as a table
    # hist = pd.DataFrame(history.history)
    # hist['epoch'] = history.epoch
    # print("Hist:")
    # print(hist.tail())

    # Visualize training process as a graph
    print("Graph:")
    plot_loss(history)

    # Get test results
    # test_results = {}
    # test_results['model'] = model.evaluate(
    # test_features['Horsepower'],
    # test_labels, verbose=0)


    # Save the model for documentation and later testing
    model.save(model_name)
