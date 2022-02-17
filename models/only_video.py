# -*- coding: utf-8 -*-
# @Time    : 6/10/21 5:04 PM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : ast_models.py

from audioop import mul
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
import os
import wget
os.environ['TORCH_HOME'] = '../pretrained_models'
import timm
import torch.nn.functional as F
from timm.models.layers import to_2tuple,trunc_normal_
from transformers import BeitModel, BeitConfig, BertModel,BeitFeatureExtractor

# override the timm package to relax the input shape constraint.
class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

class VTModel(nn.Module):
    """
    The VT model.
    :param label_dim: the label dimension, i.e., the number of total classes, it is 527 for AudioSet, 50 for ESC-50, and 35 for speechcommands v2-35
    :param fstride: the stride of patch spliting on the frequency dimension, for 16*16 patchs, fstride=16 means no overlap, fstride=10 means overlap of 6
    :param tstride: the stride of patch spliting on the time dimension, for 16*16 patchs, tstride=16 means no overlap, tstride=10 means overlap of 6
    :param input_fdim: the number of frequency bins of the input spectrogram
    :param input_tdim: the number of time frames of the input spectrogram
    :param imagenet_pretrain: if use ImageNet pretrained model
    :param audioset_pretrain: if use full AudioSet and ImageNet pretrained model
    :param model_size: the model size of AST, should be in [tiny224, small224, base224, base384], base224 and base 384 are same model, but are trained differently during ImageNet pretraining.
    """
    def __init__(self, label_dim=527, fstride=10, tstride=10, input_fdim=128, input_tdim=1024, imagenet_pretrain=True, audioset_pretrain=False, model_size='base384', patch_num=256, verbose=True):

        super(VTModel, self).__init__()
        # self.v = BeitModel.from_pretrained('microsoft/beit-base-patch16-224-pt22k-ft22k')
        self.v = BeitModel.from_pretrained('/users10/zyzhang/graduationProject/data/pretrain_model/beit-base-patch16-224-pt22k-ft22k')
        self.original_embedding_dim = 768
        self.mlp_head = nn.Sequential(nn.LayerNorm(self.original_embedding_dim), nn.Linear(self.original_embedding_dim, label_dim))
        # self.feature_extract = BeitFeatureExtractor.from_pretrained("microsoft/beit-base-patch16-224-pt22k-ft22k")
        self.feature_extract = BeitFeatureExtractor.from_pretrained('/users10/zyzhang/graduationProject/data/pretrain_model/beit-base-patch16-224-pt22k-ft22k')

    def get_shape(self, fstride, tstride, input_fdim=384, input_tdim=384):
        test_input = torch.randn(1, 3, input_fdim, input_tdim)
        test_proj = nn.Conv2d(3, self.original_embedding_dim, kernel_size=(16, 16), stride=(fstride, tstride))
        test_out = test_proj(test_input)
        f_dim = test_out.shape[2]
        t_dim = test_out.shape[3]
        return f_dim, t_dim

    @autocast()
    def forward(self, x):
        """
        :param x: the input spectrogram, expected shape: (batch_size, time_frame_num, frequency_bins), e.g., (12, 1024, 128)
        :return: prediction
        """
        # expect input x = (batch_size, 3, H, W)
        B = x.shape[0]
        x=x.cpu().numpy()
        x=[torch.Tensor(a) for a in x]
        # print('x.shape',x.shape)#[8, 3, 384, 384]
        # print('x[0]',x[0])
        x = self.feature_extract(images=x, return_tensors="pt")

        # x = self.v.patch_embed(x)
        x['pixel_values']=x['pixel_values'].to(torch.device('cuda:0'))
        # print("x['pixel_values']",x['pixel_values'])
        outputs = self.v(**x,output_hidden_states=True)
        last_hidden_state = outputs.last_hidden_state #[8, 197, 768]
        pooler_output = outputs.pooler_output #[8, 768]
        hidden_states = outputs.hidden_states #length=13 [8, 197, 768]
        #这里的hidden_states包括了最开始的embedding，去掉第一个
        hidden_states = hidden_states[1:]
        hidden_states = torch.stack(hidden_states)
        return last_hidden_state, pooler_output, hidden_states



class only_video(nn.Module):
    def __init__(self, label_dim, video_model):
        super(only_video, self).__init__()
        self.video_model = video_model
        self.predict = nn.Linear(768, label_dim)
        
        
    @autocast()
    def forward(self, audio_input, video_input, text_input):
        video_last_hidden_state, video_pooler_output, video_hidden_states = self.video_model(video_input)
        cls_embedding = video_last_hidden_state[:, 0, :]
        
        result = self.predict(cls_embedding)
        return result