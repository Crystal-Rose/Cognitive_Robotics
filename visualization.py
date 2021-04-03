import librosa
import matplotlib.pyplot as plt
import numpy as np

# Load audio
signal, sr = librosa.load("audio/1s.wav", sr=44100)
print(np.array(signal))
print(signal.dtype)

# Display audiowave
x = range(len(signal))
plt.plot(x, signal)
plt.show()
