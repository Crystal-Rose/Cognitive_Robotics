import librosa
import matplotlib.pyplot as plt
import seaborn
import numpy as np

# Load audio
signal, sr = librosa.load("audio.wav", sr=22050)

# Display audiowave
x = range(len(signal))
seaborn.lineplot(x=x, y=signal)
# plt.show()

# Get features
zcr = librosa.feature.zero_crossing_rate(signal)
print(zcr)

# Visualize features
zcr = np.array(zcr).flatten()
x = range(len(zcr))
seaborn.lineplot(x=x, y=zcr)
plt.show()
