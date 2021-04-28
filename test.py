import opensmile
import numpy as np
import tensorflow as tf
from tensorflow import keras
import random
from variables import ANNOTATIONS_PATH, LABELS, TIME_STEP
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
        self.a = list()         # stores audio signal
        self.ch1 = list()       # stores audio signal for 1st channel
        self.ch2 = list()
        self.features = str()
        self.features += "pcm_fftMag_spectralRollOff90.0,pcm_fftMag_spectralFlux,pcm_fftMag_spectralCentroid,pcm_fftMag_spectralMaxPos,pcm_fftMag_spectralMinPos,pcm_LOGenergy,pcm_zcr\n"
        self.time = 0
        self.false_predictions = 0
        self.true_predictions = 0

        # prepare model for prediction
        self.model = keras.models.load_model("trained_model")

        # read test labels
        with open("test_labels.txt", "r") as f:
            content = f.readlines()
        self.labels = np.array(list(map(lambda x: np.array([x.split(" ",1)[0], x.split(" ",1)[1].strip()]), content)))

    def inc(self, plus):
        self.a += list(plus)
    
    def addtime(self, time):
        self.time += time

    def addToOneChannel(self, plus, channel):
        if channel == 1:
            self.ch1 += list(plus)
        elif channel == 2:
            self.ch2 += list(plus)

    def print(self):
        self.count += 1
        print(self.count, end="\r")
        # print(len(self.a), end="\r")

    def plot(self):
        plt.plot(range(len(self.a)), self.a)
        plt.show()
    
    def plotStereo(self):
        fig, axs = plt.subplots(2, 1)
        axs[0].plot(range(len(self.ch1)), self.ch1)
        axs[1].plot(range(len(self.ch2)), self.ch2)
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

    def predict(self, features1, features2):
        features1 = np.array(features1)
        features2 = np.array(features2)
        features1 = np.mean(features1, axis=0)
        features2 = np.mean(features2, axis=0)
        features = np.hstack((features1, features2))
        # print(features)

        # Make a prediction
        predictions = self.model.predict(np.array([features]))
        if self.time/TIME_STEP < len(self.labels):
            expected = self.labels[int(self.time/TIME_STEP), 1]
            predicted = LABELS[np.argmax(predictions)]
            print("Prediction: {}, Expected: {}".format(predicted, expected))
            if predicted == expected:
                self.true_predictions += 1
            else:
                self.false_predictions += 1
        else:
            print("Prediction: {}".format(LABELS[np.argmax(predictions)]))

    def saveCSV(self):
        with open("outout.csv", "w") as f:
            f.write(self.features)
    
    def showStats(self):
        accuracy = self.true_predictions / (self.false_predictions + self.true_predictions)
        print("accuracy: {}".format(accuracy))

# PyAudio tutorial: https://people.csail.mit.edu/hubert/pyaudio/docs/
# Here see the settings for audio recording: we can set them ourselves to fit the training data format
# Convertion stereo->mono in ffmpeg: ffmpeg -i audio/1s.wav -ac 1 audio/mono.wav
def liveTest(filepath):
    wf = wave.open(filepath, 'rb')
    print("samplewidth, nchannels, framerate, nframes")
    print(wf.getsampwidth())    #2 (bytes per sample)
    print(wf.getnchannels())    #2 (now - 1)
    print(wf.getframerate())    #44100  # sampling frequency
    print(wf.getnframes())      #441000 -> 10sec audio (True)

    # signal, sampling_rate = audiofile.read("audio/1s.wav", always_2d=True)
    smile = opensmile.Smile(
        feature_set='conf/alqudah_live.conf',
        feature_level='features',
        num_channels = 1    # audio will be broken down into 2 channels
    )

    c = C()

    p = pyaudio.PyAudio()
    def callback(in_data, frame_count, time_info, status):
        c.addtime(0.1)

        # print(frame_count)
        data = wf.readframes(frame_count)
        buff = np.frombuffer(data, dtype="int16")/pow(2,15) # int16 I guess because sampwidth is 2 bytes = 18 bit
        # c.inc(buff)     
        # print(len(buff))  the buffer size is frame_count * 2 because 2 channels were recorded

        buff_stereo = np.reshape(buff, (-1,2))
        channel1 = buff_stereo[:,0]
        channel2 = buff_stereo[:,1]
        c.addToOneChannel(channel1, 1)
        c.addToOneChannel(channel2, 2)

        c.print()
        features1 = smile.process_signal(channel1, wf.getframerate())
        features2 = smile.process_signal(channel2, wf.getframerate())
        c.predict(features1, features2)

        c.appendFeatures(features1)
        return (data, pyaudio.paContinue)
    
    # todo: set the format manually like "format=pyaudio.paInt16"
    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
        channels=2, #wf.getnchannels(),
        rate=wf.getframerate(),
        # input=True,
        output=True,
        stream_callback=callback,
        frames_per_buffer=int(wf.getframerate()/10), # 0.1sec - a processing step for OpenSMILE. 2 - num channels
    )
    
    stream.start_stream()
    while stream.is_active():
        time.sleep(0.1)
    print("Done")
    
    stream.stop_stream()
    stream.close()
    wf.close()
    p.terminate()

    # c.plot()
    c.plotStereo()
    # c.saveCSV()
    c.showStats()

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
    liveTest("audio/rec.wav")
    # opensmileTrial()


