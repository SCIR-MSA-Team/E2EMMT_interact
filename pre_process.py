import pickle
import json

train_lis = []
with open("/users10/zyzhang/multimodel/Multimodal-End2end-Sparse-onlyVideo/data/MOSEI_SPLIT/train_split.txt", "r") as f:
    lines = f.readlines()
    for line in lines:
        train_lis.append(line.strip())

valid_lis = []
with open("/users10/zyzhang/multimodel/Multimodal-End2end-Sparse-onlyVideo/data/MOSEI_SPLIT/valid_split.txt", "r") as f:
    lines = f.readlines()
    for line in lines:
        valid_lis.append(line.strip())

test_lis = []
with open("/users10/zyzhang/multimodel/Multimodal-End2end-Sparse-onlyVideo/data/MOSEI_SPLIT/test_split.txt", "r") as f:
    lines = f.readlines()
    for line in lines:
        test_lis.append(line.strip()) 


train_mosei_dataset = []
valid_mosei_dataset = []
test_mosei_dataset = []

with open("/users10/zyzhang/multimodel/Multimodal-End2end-Sparse-onlyVideo/data/MOSEI_RAW_PROCESSED/meta.pkl", "rb") as f:
        data = pickle.load(f)
        for iid in data.keys():
            if iid in train_lis:
                train_mosei_dataset.append({"id": iid, \
                            "labels": data[iid]["label"], \
                            "text": data[iid]["text"], \
                            "wav": "/users10/zyzhang/multimodel/Multimodal-End2end-Sparse-onlyVideo/data/MOSEI_RAW_PROCESSED/"+iid+"/audio.wav"})

            elif iid in valid_lis:
                valid_mosei_dataset.append({"id": iid, \
                            "labels": data[iid]["label"], \
                            "text": data[iid]["text"], \
                            "wav": "/users10/zyzhang/multimodel/Multimodal-End2end-Sparse-onlyVideo/data/MOSEI_RAW_PROCESSED/"+iid+"/audio.wav"})

            elif iid in test_lis:
                test_mosei_dataset.append({"id": iid, \
                            "labels": data[iid]["label"], \
                            "text": data[iid]["text"], \
                            "wav": "/users10/zyzhang/multimodel/Multimodal-End2end-Sparse-onlyVideo/data/MOSEI_RAW_PROCESSED/"+iid+"/audio.wav"})

json.dump(train_mosei_dataset, open("train_mosei_dataset.json", "w"))
json.dump(valid_mosei_dataset, open("valid_mosei_dataset.json", "w"))
json.dump(test_mosei_dataset, open("test_mosei_dataset.json", "w"))


 
