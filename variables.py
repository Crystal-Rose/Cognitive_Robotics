ANNOTATIONS_PATH = "annotations.json"

'''
Structure of a JSON file with features:
{
    data: [
        {
            "class": 0,                             # 1 - rumble strip, 1 - everything else
            "side": 0,                              # 1 - right, 0 - left
            "features": [0.98,1.34,2.56,...]        # list of features
        },
        ...
        {...},
        ...
    ]
}
'''