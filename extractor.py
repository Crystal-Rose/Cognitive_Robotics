import librosa
import librosa.display
import matplotlib.pyplot as plt#
import seaborn
import numpy as np
import subprocess
from variables import ANNOTATIONS_PATH
import json

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


'''
Features by Alqudah:
zero crossing rate, energy entropy, short time energy, spectral rolloff, spectral centroid and spectral flux

SMILExtract -C "opensmile/alqudah.conf" -I "audio/1s.wav" -O "out.csv"
'''

def extractAudioFeatures(audio_path, config_path, output_path):
    subprocess.call(
        ["SMILExtract", "-C", config_path,
            "-I", audio_path,
            # "-lldcsvoutput", output_path
            # "-frameModeFunctionalsConf", "openSMILE_conf/egemaps/FrameModeFunctionals.conf.inc",
            "-O", output_path
        ]
    )

def buildJSON():
    obj = {}
    data = list()
    dataobj = {}
    dataobj["class"] = 0
    dataobj["label"] = 0
    dataobj["features"] = []
    data.append()
    obj.append("data": data)
    objstr = json.dumps(obj)
    with open(ANNOTATIONS_PATH, "w") as f:
        f.write(objstr)

if __name__ == "__main__":
    pass