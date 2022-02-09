from transformers import BeitFeatureExtractor, BeitModel
from PIL import Image
import numpy as np
import requests
import torch
import copy

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

feature_extractor = BeitFeatureExtractor.from_pretrained("microsoft/beit-base-patch16-224-pt22k-ft22k")
model = BeitModel.from_pretrained("microsoft/beit-base-patch16-224-pt22k-ft22k")

# print('image.shape',image.shape)
image=torch.LongTensor(np.array(image).transpose(2,1,0))
print('image.shape',image.shape)
# print('image',image)
image2=copy.deepcopy(image)
a=[image,image2]
# a=torch.stack(a)
# print('a.shape',a.shape)
inputs = feature_extractor(images=a, return_tensors="pt")
print("inputs['pixel_values'].shape",inputs['pixel_values'].shape)
print("type(inputs['pixel_values'])",type(inputs['pixel_values']))
# print("inputs['pixel_values']",inputs['pixel_values'])
# inputs['pixel_values']=
print('inputs.keys()',inputs.keys())

outputs = model(**inputs)
last_hidden_states = outputs.last_hidden_state
print('outputs.keys()',outputs.keys())
print("outputs['pooler_output'].shape",outputs['pooler_output'].shape)
print('last_hidden_states.shape',last_hidden_states.shape)