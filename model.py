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
        annotations = json.load(f)
    
    features = list()
    labels = list()
    for data in annotations["data"]:
        labels.append(data["label"])
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

def train(model_name):
    '''
    Returns the model trained on a provided dataset (.csv)
    '''
    features, labels = loadData()
    print(features.shape)

    # Data normalizer
    normalizer = preprocessing.Normalization(axis=1)
    normalizer.adapt(features)

    # Visualize normalization effect
    print('First row of features (before norm):', features[0])
    print('Normalized:', normalizer(np.array([features[0]])))

    # Train the regression
    model = tf.keras.Sequential([
        layers.Flatten(input_shape=(1, features.shape[1])),
        # When the layer is called it returns the input data, with each feature independently normalized
        normalizer,
        # layers.Dense(32, activation="relu")
        layers.Dense(3, activation="softmax")   # 3 classes: no rumble, left rumble, right rumble
    ])

    # Configure the trainng procedure
    model.compile(
        optimizer=tf.optimizers.Adam(learning_rate=0.01),
        loss='sparse_categorical_crossentropy',
        metrics=["accuracy"]
    )

    # Get a print-out summary of the model
    model.summary()
    
    # Train and save the history (training process)
    history = model.fit(
        features, labels,
        epochs = 50, 
        verbose = 1,                # 0 to suppress logging
        validation_split = 0.2,      # Calculate validation results on 20% of the training data
        batch_size = 1
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


if __name__ == "__main__":
    train("trained_model")