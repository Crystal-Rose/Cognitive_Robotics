import opensmile
import numpy as np
import tensorflow as tf
from tensorflow import keras
import random
from variables import ANNOTATIONS_PATH
from model import loadData
import pyaudio
import wave
import time
import sys
import audiofile
import matplotlib.pyplot as plt
import struct

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

class C:
    def __init__(self):
        self.count = 1
        self.a = list()
        self.features = str()
        self.features += "pcm_fftMag_spectralRollOff90.0,pcm_fftMag_spectralFlux,pcm_fftMag_spectralCentroid,pcm_fftMag_spectralMaxPos,pcm_fftMag_spectralMinPos,pcm_LOGenergy,pcm_zcr\n"
    def inc(self, plus):
        self.a += list(plus)
    def print(self):
        self.count += 1
        print(self.count, end="\r")
        # print(len(self.a), end="\r")
    def plot(self):
        plt.plot(range(len(self.a)), self.a)
        plt.show()
    def appendFeatures(self, features):
        new_features = np.array(features)
        for row in new_features:
            str_row = str()
            for i in row[:-1]:
                str_row = str_row + str(i) + ","
            str_row = str_row + str(row[-1]) + "\n"
            self.features += str_row
    def printFeatures(self, features):
        print(features)
    def saveCSV(self):
        with open("outout.csv", "w") as f:
            f.write(self.features)

# PyAudio tutorial: https://people.csail.mit.edu/hubert/pyaudio/docs/
# Here see the settings for audio recording: we can set them ourselves to fit the training data format
# Convertion stereo->mono in ffmpeg: ffmpeg -i audio/1s.wav -ac 1 audio/mono.wav
def liveTest():
    wf = wave.open("audio/mono.wav", 'rb')
    print("samplewidth, nchannels, framerate, nframes")
    print(wf.getsampwidth())    #2
    print(wf.getnchannels())    #2 (now - 1)
    print(wf.getframerate())    #44100  # sampling frequency
    print(wf.getnframes())      #441000 -> 10sec audio (True)

    # signal, sampling_rate = audiofile.read("audio/1s.wav", always_2d=True)
    smile = opensmile.Smile(
        feature_set='conf/alqudah_live.conf',
        feature_level='features',
        num_channels=1  #wf.getnchannels()
    )

    c = C()

    p = pyaudio.PyAudio()
    def callback(in_data, frame_count, time_info, status):
        # print(frame_count)
        data = wf.readframes(frame_count)
        c.inc(np.frombuffer(data, dtype="int16")/pow(2,15))
        c.print()
        features = smile.process_signal(
            np.frombuffer(data, dtype="int16")/pow(2,15),
            wf.getframerate()
        )
        c.appendFeatures(features)
        return (data, pyaudio.paContinue)
    
    # todo: set the format manually like "format=pyaudio.paInt16"
    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
        channels=wf.getnchannels(),
        rate=wf.getframerate(),
        # input=True,
        output=True,
        stream_callback=callback,
        frames_per_buffer=int(wf.getframerate()/10), # 0.1sec
    )
    
    stream.start_stream()
    while stream.is_active():
        time.sleep(0.1)
    print("Done")
    
    stream.stop_stream()
    stream.close()
    wf.close()
    p.terminate()

    c.plot()
    c.saveCSV()

def opensmileTrial():
    signal, sampling_rate = audiofile.read("audio/1s.wav", always_2d=True)
    # wf = wave.open("audio/1s.wav", 'rb')
    # signal = wf.readframes(4096)
    smile = opensmile.Smile(
        feature_set='conf/alqudah_live.conf',
        feature_level='features',
        num_channels=2,
    )
    print(signal.shape)
    result = smile.process_signal(
        signal[:, :4096],
        sampling_rate
    )
    print(result)

if __name__ == "__main__":
    liveTest()
    # opensmileTrial()
