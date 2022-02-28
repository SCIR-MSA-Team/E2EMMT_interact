#消融实验1：不用interact，只用3个模态的cls进行融合

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

class ASTModel(nn.Module):
    def __init__(self, label_dim=527, fstride=10, tstride=10, input_fdim=128, input_tdim=1024, imagenet_pretrain=True, audioset_pretrain=False, model_size='base384', patch_num=256, verbose=True):

        super(ASTModel, self).__init__()
        assert timm.__version__ == '0.4.5', 'Please use timm == 0.4.5, the code might not be compatible with newer versions.'

        if verbose == True:
            print('---------------AST Model Summary---------------')
            print('ImageNet pretraining: {:s}, AudioSet pretraining: {:s}'.format(str(imagenet_pretrain),str(audioset_pretrain)))
        # override timm input shape restriction
        timm.models.vision_transformer.PatchEmbed = PatchEmbed

        # Reduce the number of the tokens

        self.token_nums = patch_num #128 * 2

        # if AudioSet pretraining is not used (but ImageNet pretraining may still apply)
        if audioset_pretrain == False:
            if model_size == 'tiny224':
                self.v = timm.create_model('vit_deit_tiny_distilled_patch16_224', pretrained=imagenet_pretrain)
            elif model_size == 'small224':
                self.v = timm.create_model('vit_deit_small_distilled_patch16_224', pretrained=imagenet_pretrain)
            elif model_size == 'base224':
                self.v = timm.create_model('vit_deit_base_distilled_patch16_224', pretrained=imagenet_pretrain)
            elif model_size == 'base384':
                self.v = timm.create_model('vit_deit_base_distilled_patch16_384', pretrained=imagenet_pretrain)
            else:
                raise Exception('Model size must be one of tiny224, small224, base224, base384.')
            self.original_num_patches = self.v.patch_embed.num_patches
            self.oringal_hw = int(self.original_num_patches ** 0.5)
            self.original_embedding_dim = self.v.pos_embed.shape[2]
            self.mlp_head = nn.Sequential(nn.LayerNorm(self.original_embedding_dim), nn.Linear(self.original_embedding_dim, label_dim))

            # automatcially get the intermediate shape
            num_patches = int(input_tdim/2)
            self.v.patch_embed.num_patches = num_patches
            if verbose == True:
                print('frequncey stride={:d}, time stride={:d}'.format(fstride, tstride))
                print('number of patches={:d}'.format(num_patches))

            # the linear projection layer
            new_proj = torch.nn.Conv2d(1, self.original_embedding_dim, kernel_size=(16, 16), stride=(fstride, tstride))
            if imagenet_pretrain == True:
                new_proj.weight = torch.nn.Parameter(torch.sum(self.v.patch_embed.proj.weight, dim=1).unsqueeze(1))
                new_proj.bias = self.v.patch_embed.proj.bias
            self.v.patch_embed.proj = new_proj

            # the positional embedding
            if imagenet_pretrain == True:
                # concatenate the above positional embedding with the cls token and distillation token of the deit model.
                self.v.pos_embed = nn.Parameter(self.v.pos_embed[:, :int(input_tdim/2 + 2), :].detach())
            else:
                # if not use imagenet pretrained model, just randomly initialize a learnable positional embedding
                # TODO can use sinusoidal positional embedding instead
                new_pos_embed = nn.Parameter(torch.zeros(1, self.v.patch_embed.num_patches + 2, self.original_embedding_dim))
                self.v.pos_embed = new_pos_embed
                trunc_normal_(self.v.pos_embed, std=.02)
        for param in self.v.parameters():
            param.requires_grad = False


    def get_shape(self, fstride, tstride, input_fdim=128, input_tdim=1024):
        test_input = torch.randn(1, 1, input_fdim, input_tdim)
        test_proj = nn.Conv2d(1, self.original_embedding_dim, kernel_size=(16, 16), stride=(fstride, tstride))
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
        # expect input x = (batch_size, time_frame_num, frequency_bins), e.g., (12, 1024, 128)
    
        x = x.unsqueeze(1)
        x = x.transpose(2, 3)

        B = x.shape[0]
        x = self.v.patch_embed(x)

        # -1 means do not change
        cls_tokens = self.v.cls_token.expand(B, -1, -1)
        dist_token = self.v.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)
        x = x + self.v.pos_embed
        x = self.v.pos_drop(x)
        hidden_states = []
        for blk in self.v.blocks:
            x = blk(x)
            hidden_states.append(x)
        x = self.v.norm(x)
        x = (x[:, 0] + x[:, 1]) / 2
        x_out = self.mlp_head(x)
        hidden_states = torch.stack(hidden_states) #[12, 8, 514, 768]
        return hidden_states


class VTModel_deit(nn.Module):
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

        super(VTModel_deit, self).__init__()
        assert timm.__version__ == '0.4.5', 'Please use timm == 0.4.5, the code might not be compatible with newer versions.'

        if verbose == True:
            print('---------------AST Model Summary---------------')
            print('ImageNet pretraining: {:s}, AudioSet pretraining: {:s}'.format(str(imagenet_pretrain),str(audioset_pretrain)))
        # override timm input shape restriction
        timm.models.vision_transformer.PatchEmbed = PatchEmbed

        # Reduce the number of the tokens

        self.token_nums = patch_num #128 * 2
        self.token_attention = nn.Linear(768, self.token_nums)
        self.t_token_attention = nn.Linear(768*2, self.token_nums)
        self.ta_token_attention = nn.Linear(768*3, self.token_nums)

        # if AudioSet pretraining is not used (but ImageNet pretraining may still apply)
        if audioset_pretrain == False:
            if model_size == 'tiny224':
                self.v = timm.create_model('vit_deit_tiny_distilled_patch16_224', pretrained=imagenet_pretrain)
            elif model_size == 'small224':
                self.v = timm.create_model('vit_deit_small_distilled_patch16_224', pretrained=imagenet_pretrain)
            elif model_size == 'base224':
                self.v = timm.create_model('vit_deit_base_distilled_patch16_224', pretrained=imagenet_pretrain)
            elif model_size == 'base384':
                self.v = timm.create_model('vit_deit_base_distilled_patch16_384', pretrained=imagenet_pretrain)
            else:
                raise Exception('Model size must be one of tiny224, small224, base224, base384.')
            self.original_num_patches = self.v.patch_embed.num_patches
            self.oringal_hw = int(self.original_num_patches ** 0.5)
            self.original_embedding_dim = self.v.pos_embed.shape[2]
            self.mlp_head = nn.Sequential(nn.LayerNorm(self.original_embedding_dim), nn.Linear(self.original_embedding_dim, label_dim))

            # automatcially get the intermediate shape
            num_patches = self.token_nums
            self.v.patch_embed.num_patches = num_patches
            if verbose == True:
                print('frequncey stride={:d}, time stride={:d}'.format(fstride, tstride))
                print('number of patches={:d}'.format(num_patches))

            # the positional embedding
            if imagenet_pretrain == True:
                # concatenate the above positional embedding with the cls token and distillation token of the deit model.
                self.v.pos_embed = nn.Parameter(self.v.pos_embed[:, :, :].detach())
            else:
                # if not use imagenet pretrained model, just randomly initialize a learnable positional embedding
                # TODO can use sinusoidal positional embedding instead
                new_pos_embed = nn.Parameter(torch.zeros(1, self.v.patch_embed.num_patches + 2, self.original_embedding_dim))
                self.v.pos_embed = new_pos_embed
                trunc_normal_(self.v.pos_embed, std=.02)
        # self.v.requires_grad = False
        for param in self.v.parameters():
            param.requires_grad = False

    def get_shape(self, fstride, tstride, input_fdim=384, input_tdim=384):
        test_input = torch.randn(1, 3, input_fdim, input_tdim)
        test_proj = nn.Conv2d(3, self.original_embedding_dim, kernel_size=(16, 16), stride=(fstride, tstride))
        test_out = test_proj(test_input)
        f_dim = test_out.shape[2]
        t_dim = test_out.shape[3]
        return f_dim, t_dim

    @autocast()
    def forward(self, x, text_cls=None, audio_cls=None):
        """
        :param x: the input spectrogram, expected shape: (batch_size, time_frame_num, frequency_bins), e.g., (12, 1024, 128)
        :return: prediction
        """
        # expect input x = (batch_size, 3, H, W)
    

        B = x.shape[0]
        x = self.v.patch_embed(x)
        # -1 means do not change
        cls_tokens = self.v.cls_token.expand(B, -1, -1)
        dist_token = self.v.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)
        x = x + self.v.pos_embed
        x = self.v.pos_drop(x)
        hidden_states = []
        for blk in self.v.blocks:
            x = blk(x)
            hidden_states.append(x)
        x = self.v.norm(x)
        x = (x[:, 0] + x[:, 1]) / 2

        x_out = self.mlp_head(x)
        hidden_states = torch.stack(hidden_states)
        return None, None, hidden_states

class TTModel(nn.Module):
    def __init__(self, num_classes):
        super(TTModel, self).__init__()
        self.bert = BertModel.from_pretrained("/users10/zyzhang/graduationProject/data/pretrain_model/bert_base_uncased")
        self.num_classes = num_classes
        self.mlp_head = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, self.num_classes))
        # self.mlp = nn.Linear(768, self.num_classes)
        for param in self.bert.parameters():
            param.requires_grad = False

    @autocast()
    def forward(self, text):
        outputs = self.bert(**text,output_hidden_states=True)
        last_hidden_state = outputs.last_hidden_state
        pooler_output = outputs.pooler_output
        hidden_states = outputs.hidden_states
        #这里的hidden_states包括了最开始的embedding，去掉第一个
        hidden_states = hidden_states[1:]
        hidden_states = torch.stack(hidden_states)
        return last_hidden_state, pooler_output, hidden_states



class MTModel(nn.Module):
    def __init__(self, label_dim, audio_model, video_model, text_model, modals="tav"):
        super(MTModel, self).__init__()
        self.audio_model = audio_model
        self.video_model = video_model
        self.text_model = text_model
        self.modals = modals
        self.text_predict = nn.Linear(768, label_dim)
        self.video_predict = nn.Linear(768, label_dim)
        self.audio_predict = nn.Linear(768, label_dim)
        self.fusion = nn.Linear(768*3, label_dim)
        self.weighted_fusion = nn.Linear(4, 1, bias=False)
        
    @autocast()
    def forward(self, audio_input, video_input, text_input):
        text_last_hidden_state, text_pooler_output, text_hidden_states = self.text_model(text_input)

        video_last_hidden_state, video_pooler_output, video_hidden_states = self.video_model(video_input)
        audio_hidden_states = self.audio_model(audio_input)
        
        text_cls = text_hidden_states[-1, :, 0, :]
        audio_cls = audio_hidden_states[-1, :, 0, :]
        video_cls = video_hidden_states[-1, :, 0, :]
        text_pred = self.text_predict(text_cls)
        video_pred = self.video_predict(video_cls)
        audio_pred = self.audio_predict(audio_cls)
        multimode_pred = self.fusion(torch.cat((text_cls, video_cls, audio_cls),-1))
        result = self.weighted_fusion(torch.stack((text_pred, video_pred, audio_pred, multimode_pred),-1)).squeeze(-1)
        return result