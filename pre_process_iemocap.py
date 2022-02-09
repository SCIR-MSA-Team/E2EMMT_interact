import pickle
import json
import glob
import os

def use_left(utteranceId):
    entries = utteranceId.split('_')
    return entries[0][-1] == entries[-1][0]


train_lis = []
with open("/users10/zyzhang/multimodel/Multimodal-End2end-Sparse-onlyVideo/data/IEMOCAP_SPLIT/train_split.txt", "r") as f:
    lines = f.readlines()
    for line in lines:
        train_lis.append(line.strip())

valid_lis = []
with open("/users10/zyzhang/multimodel/Multimodal-End2end-Sparse-onlyVideo/data/IEMOCAP_SPLIT/valid_split.txt", "r") as f:
    lines = f.readlines()
    for line in lines:
        valid_lis.append(line.strip())

test_lis = []
with open("/users10/zyzhang/multimodel/Multimodal-End2end-Sparse-onlyVideo/data/IEMOCAP_SPLIT/test_split.txt", "r") as f:
    lines = f.readlines()
    for line in lines:
        test_lis.append(line.strip()) 

utteranceFolders = {
    folder.split('/')[-1]: folder
    for folder in glob.glob(os.path.join("/users10/zyzhang/multimodel/Multimodal-End2end-Sparse-onlyVideo/data/IEMOCAP_RAW_PROCESSED", '**/*'))
}

train_iemocap_dataset = []
valid_iemocap_dataset = []
test_iemocap_dataset = []

emotion_dic = {'ang': [1,0,0,0,0,0], \
               'exc': [0,1,0,0,0,0], \
               'fru': [0,0,1,0,0,0], \
               'hap': [0,0,0,1,0,0], \
               'neu': [0,0,0,0,1,0], \
               'sad': [0,0,0,0,0,1]}

with open("/users10/zyzhang/multimodel/Multimodal-End2end-Sparse-onlyVideo/data/IEMOCAP_RAW_PROCESSED/meta.pkl", "rb") as f:
        data = pickle.load(f)
        for iid in data.keys():
            if iid in train_lis:
                flag = use_left(iid)
                audio_suffix = 'L' if flag else 'R'
                train_iemocap_dataset.append({"id": iid, \
                            "labels": emotion_dic[data[iid]["label"]], \
                            "text": data[iid]["text"], \
                            "wav": utteranceFolders[iid]+"/audio_"+audio_suffix+".wav"})

            elif iid in valid_lis:
                flag = use_left(iid)
                audio_suffix = 'L' if flag else 'R'
                valid_iemocap_dataset.append({"id": iid, \
                            "labels": emotion_dic[data[iid]["label"]], \
                            "text": data[iid]["text"], \
                            "wav": utteranceFolders[iid]+"/audio_"+audio_suffix+".wav"})

            elif iid in test_lis:
                flag = use_left(iid)
                audio_suffix = 'L' if flag else 'R'
                test_iemocap_dataset.append({"id": iid, \
                            "labels": emotion_dic[data[iid]["label"]], \
                            "text": data[iid]["text"], \
                            "wav": utteranceFolders[iid]+"/audio_"+audio_suffix+".wav"})

json.dump(train_iemocap_dataset, open("train_iemocap_dataset.json", "w"))
json.dump(valid_iemocap_dataset, open("valid_iemocap_dataset.json", "w"))
json.dump(test_iemocap_dataset, open("test_iemocap_dataset.json", "w"))


 
