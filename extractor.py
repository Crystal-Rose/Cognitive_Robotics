import numpy as np
import subprocess
from variables import ANNOTATIONS_PATH
import json
import os

'''
Features by Alqudah:
zero crossing rate, energy entropy, short time energy, spectral rolloff, spectral centroid and spectral flux

SMILExtract -C "opensmile/alqudah.conf" -I "audio/1s.wav" -O "out.csv"
'''

def extractAudioFeatures(audio_path, output_path, config_path="conf/alqudah.conf"):
    subprocess.call(
        ["SMILExtract", "-C", config_path,
            "-I", audio_path,
            # "-lldcsvoutput", output_path
            # "-frameModeFunctionalsConf", "openSMILE_conf/egemaps/FrameModeFunctionals.conf.inc",
            "-O", output_path
        ]
    )

def extractForAllFiles():
    # create features folder if not created yet
    if not os.path.exists("Extracted features"):
        os.mkdir("Extracted features")

    # for left microphone
    audio_paths = ["Extracted audio/Left_microphone", "Extracted audio/Right_microphone"]
    feature_paths = ["Extracted features/Left_microphone", "Extracted features/Right_microphone"]
    
    for i in range(2):
        if not os.path.exists(feature_paths[i]):
            os.mkdir(feature_paths[i])
        for audio_name in os.listdir(audio_paths[i]):
            extractAudioFeatures(os.path.join(audio_paths[i], audio_name), os.path.join(feature_paths[i], audio_name.replace(".wav", ".csv")))

'''
Reads a csv containing features and returns means of features and labels
returns: 
    features : [[t0, t1, ...], [t0, t1, ...], [t0, t1, ...]],
    label: label      # either 0 (no rumble), 1 (left rumble) or 2 (right rumble) 
returns 0 if there is an issue
'''
def readCSV(csv_path):
    print("Reading a CSV file: {}".format(csv_path))

    # label information is contained in the name of the file
    label = None
    if "nr" in csv_path:
        label = 0
    elif "lr" in csv_path:
        label = 1
    elif "rr" in csv_path:
        label = 2
    else:
        print("None of lr, rr or nr found in {}. Ignoring".format(csv_path))
        return None, None
    
    try:
        with open(csv_path, "r") as f:
            lines = f.readlines()
    except FileNotFoundError:
        print("File {} not found. Ignoring".format(csv_path))
        return None, None

    num_features = len(lines[0].split(","))
    features = np.zeros((0,num_features))   # creates an empty nxnum_features matrix to store all features

    # fist line is a header, so we skip it
    for line in lines[1:]:
        parts = list(map(float,line.split(",")))
        assert len(parts) == num_features
        features = np.vstack((features, parts))
    
    assert features.ndim == 2
    assert features.shape[0] > 0
    assert features.shape[1] > 0
    return features, label


'''
Structure of a JSON file with features:
{
    data: [
        {
            "audio": "audio_name.wav",
            "class": 0,                                   # 0 - no rumble, 1 - left rumble strip, 2 - right rumble strip
            "features": [[0.98,1.34,2.56,...],...]        # matrix of features (column - cetain feature, row - timestamp)
            or "features": [[1.4],[2.3],...]              # or a vector of funcionals (means for each feature across an audio track)
        },
        ...
        {...},
        ...
    ]
}
'''

def buildJSON(meanEachSide=True, concatenateSides=True):
    obj = {}
    obj["data"] = list()

    left_features = "Extracted features/Left_microphone/"
    right_features = "Extracted features/Right_microphone/"
    left_audio = "Extracted audio/Left_microphone/"
    right_audio = "Extracted audio/Right_microphone/"
    for csv_name in os.listdir(left_features):
        # Getting 
        features_left, label_left = readCSV(os.path.join(left_features, csv_name))
        features_right, label_right = readCSV(os.path.join(right_features, csv_name.replace("_left", "_right")))

        # Ignoring all failed files
        if features_left is None or label_left is None:
            print("Left features / label is None")
            continue
        elif features_right is None or label_right is None:
            print("Right features / label is None")
            continue
        
        # Getting means for each feature
        if meanEachSide:
            features_left = features_left.mean(axis=0)  # this will calculate the means for each column
            features_right = features_right.mean(axis=0)
            assert features_left.ndim == 1 and features_right.ndim == 1
        
        left_audio_path = str(os.path.join(left_audio, csv_name.replace(".csv", ".wav")))
        right_audio_path = str(os.path.join(right_audio, csv_name.replace("_left.csv", "_right.wav")))

        # Combining the 
        if concatenateSides:
            dataobj = {}
            dataobj["files"] = [left_audio_path, right_audio_path]
            dataobj["label"] = label_left
            dataobj["features"] = list(np.hstack((features_left, features_right)))
            obj["data"].append(dataobj)
        else:
            print("Function without sides concatenation is not implemented yet")

    objstr = json.dumps(obj)
    with open(ANNOTATIONS_PATH, "w") as f:
        f.write(objstr)

if __name__ == "__main__":
    # extractAudioFeatures("audio/1s.wav", "0.cvs")

    ## 2 stages that are started separately:
    ## 1st: extract csv files with OpenSMILE
    # extractForAllFiles()

    ## 2nd: generate a JSON file
    # buildJSON()