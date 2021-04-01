import librosa
import librosa.display
import matplotlib.pyplot as plt#
import seaborn
import numpy as np
import subprocess

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

def extractAudioFeatures(audio_path, config_path, output_path):
    subprocess.call(
        ["SMILExtract", "-C", config_path,
            "-I", audio_path,
            # "-lldcsvoutput", output_path
            # "-frameModeFunctionalsConf", "openSMILE_conf/egemaps/FrameModeFunctionals.conf.inc",
            "-O", output_path
        ]
    )

'''
Features by Alqudah:
zero crossing rate, energy entropy, short time energy, spectral rolloff, spectral centroid and spectral flux

SMILExtract -C "opensmile/alqudah.conf" -I "audio/1s.wav" -O "out.csv"
'''

if __name__ == "__main__":
    pass