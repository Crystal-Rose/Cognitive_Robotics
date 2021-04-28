import librosa
import matplotlib.pyplot as plt
import numpy as np

# # Load audio
# signal, sr = librosa.load("audio/1s.wav", sr=44100)
# print(np.array(signal))
# print(signal.dtype)

# # Display audiowave
# x = range(len(signal))
# plt.plot(x, signal)
# plt.show()

# Display stereo
y, sr = librosa.load("audio/rec.wav", mono=False)
print(y.shape)
fig, axs = plt.subplots(2,1)
axs[0].plot(range(y.shape[1]), y[0])
axs[1].plot(range(y.shape[1]), y[1])
plt.show()