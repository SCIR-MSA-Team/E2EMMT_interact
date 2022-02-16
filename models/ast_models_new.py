# -*- coding: utf-8 -*-

from audioop import mul
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
import os
import wget
os.environ['TORCH_HOME'] = '/users5/ywu/LLF/pretrained_models'
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
        pre_hidden_states = torch.stack(hidden_states) #[12, 8, 514, 768]
        post_cls = (pre_hidden_states[:, :, 0, :] + pre_hidden_states[:, :, 1, :]) / 2 
        hidden_states = torch.cat([post_cls.unsqueeze(2), pre_hidden_states[:, :, 2:, :]], 2)
        assert hidden_states.size(2) == 513
        return hidden_states


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
        self.v = BeitModel.from_pretrained('/users5/ywu/LLF/pretrained_models/beit-base-patch16-224-pt22k-ft22k')
        self.original_embedding_dim = 768
        self.mlp_head = nn.Sequential(nn.LayerNorm(self.original_embedding_dim), nn.Linear(self.original_embedding_dim, label_dim))
        self.feature_extract = BeitFeatureExtractor.from_pretrained('/users5/ywu/LLF/pretrained_models/beit-base-patch16-224-pt22k-ft22k')

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
        x = x.cpu().numpy()
        x = [torch.Tensor(a) for a in x]
        # print('x.shape',x.shape)#[8, 3, 384, 384]
        # print('x[0]',x[0])
        x = self.feature_extract(images=x, return_tensors="pt")

        # x = self.v.patch_embed(x)
        x['pixel_values'] = x['pixel_values'].to(torch.device('cuda:0'))
        # print("x['pixel_values']",x['pixel_values'])
        outputs = self.v(**x,output_hidden_states=True)
        last_hidden_state = outputs.last_hidden_state #[8, 197, 768]
        pooler_output = outputs.pooler_output #[8, 768]
        hidden_states = outputs.hidden_states #length=13 [8, 197, 768]
        #这里的hidden_states包括了最开始的embedding，去掉第一个
        hidden_states = hidden_states[1:]
        hidden_states = torch.stack(hidden_states)
        return last_hidden_state, pooler_output, hidden_states

class TTModel(nn.Module):
    def __init__(self, num_classes):
        super(TTModel, self).__init__()
        self.bert = BertModel.from_pretrained("/users5/ywu/LLF/pretrained_models/bert_en")
        self.num_classes = num_classes
        self.mlp_head = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, self.num_classes))
        # self.mlp = nn.Linear(768, self.num_classes)

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
        self.final_dims = len(self.modals)*768
        self.layerNorm = nn.LayerNorm(768)
        self.layerNorm_text = nn.LayerNorm(768)
        self.layerNorm_video = nn.LayerNorm(768)
        self.layerNorm_audio = nn.LayerNorm(768)
        self.layerNorm2 = nn.LayerNorm(768)
        self.text_map1 = nn.Linear(768*2, 768)
        self.text_attention = nn.Linear(768*2, 1)
        self.text_map2_text = nn.Linear(768*3, 768)
        self.text_map2_audio = nn.Linear(768*3, 768)
        self.text_map2_video = nn.Linear(768*3, 768)
        self.audio_map1 = nn.Linear(768*2, 768)
        self.audio_attention = nn.Linear(768*2, 1)
        self.audio_map2_text = nn.Linear(768*3, 768)
        self.audio_map2_audio = nn.Linear(768*3, 768)
        self.audio_map2_video = nn.Linear(768*3, 768)
        self.video_map1 = nn.Linear(768*2, 768)
        self.video_attention = nn.Linear(768*2, 1)
        self.video_map2_text = nn.Linear(768*3, 768)
        self.video_map2_audio = nn.Linear(768*3, 768)
        self.video_map2_video = nn.Linear(768*3, 768)
        self.text_predict = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, label_dim))
        self.video_predict = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, label_dim))
        self.audio_predict = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, label_dim))
        self.fusion = nn.Sequential(nn.LayerNorm(768*6), nn.Linear(768*6, label_dim))
        self.weighted_fusion = nn.Linear(4, 1, bias=False)

        # each fusion step
        # self.layer_classifier_t = nn.Linear(768, label_dim)       
        # self.layer_classifier_a = nn.Linear(768, label_dim)       
        # self.layer_classifier_v = nn.Linear(768, label_dim)     
        # force same space  
        self.layer_classifier = nn.Linear(768, label_dim)  

    
    def interaction(self, mode1, mode2, mode3, map1, linear_a, linear_b, linear_c, attention, layerNorm1, layerNorm2, layerNorm3, layer_classifier):
        batch_size = mode1.shape[1]
        embedding_size = mode1.shape[3]
        curr_embedding = torch.randn(batch_size, embedding_size).to(mode1.device)

        layer_pred_results = []

        # last six layers
        for step in [6, 7, 8, 9, 10, 11]:
        # for step in range(mode1.shape[0]):
            curr_embedding = self.layerNorm(curr_embedding)
            mode1_cls_embedding = layerNorm1(mode1[step, :, 0, :])
            mode1_other_embedding = layerNorm1(mode1[step, :, 1:, :])
            mode2_cls_embedding = layerNorm2(mode2[step, :, 0, :])
            mode3_cls_embedding = layerNorm3(mode3[step, :, 0, :])
            
            map1_embedding = self.layerNorm(map1(torch.cat((mode2_cls_embedding,mode3_cls_embedding),1)))
            # map1_embedding.shape torch.Size([8, 768])
            map2_embedding_a = torch.sigmoid(linear_a(torch.cat((map1_embedding, curr_embedding, mode1_cls_embedding),1))) * map1_embedding
            # map2_embedding_a.shape torch.Size([8, 768])
            map2_embedding_b = torch.sigmoid(linear_b(torch.cat((map1_embedding, curr_embedding, mode1_cls_embedding),1))) * curr_embedding
            # map2_embedding_b.shape torch.Size([8, 768])
            map2_embedding_c = torch.sigmoid(linear_c(torch.cat((map1_embedding, curr_embedding, mode1_cls_embedding),1))) * mode1_cls_embedding 
            # map2_embedding_c.shape torch.Size([8, 768])
            map2_embedding = map2_embedding_a + map2_embedding_b + map2_embedding_c
            # map2_embedding.shape torch.Size([8, 768])
            new_t_attention = attention(torch.cat((mode1_other_embedding,map2_embedding.unsqueeze(1).expand(-1,mode1_other_embedding.shape[1],-1)),-1)).squeeze(-1)
            # new_t_attention = torch.bmm(mode1_other_embedding,map2_embedding.unsqueeze(-1)).permute(0,2,1)
            new_t_attention = torch.softmax(new_t_attention,-1)
            new_t_attention = new_t_attention.unsqueeze(1)
            # new_t_attention.shape torch.Size([8, 1, 299])
            new_t = torch.bmm(new_t_attention,mode1_other_embedding).squeeze(1)
            # new_t.shape torch.Size([8, 768])
            curr_embedding = new_t + map2_embedding
            curr_embedding = self.layerNorm2(curr_embedding)

            layer_pred_results.append(layer_classifier(curr_embedding))

        return curr_embedding, torch.cat(layer_pred_results, dim=0)

    @autocast()
    def forward(self, audio_input, video_input, text_input):
        text_last_hidden_state, text_pooler_output, text_hidden_states = self.text_model(text_input)
        video_last_hidden_state, video_pooler_output, video_hidden_states = self.video_model(video_input)
        audio_hidden_states = self.audio_model(audio_input)
        
        text_inte_embedding, text_layer_pred_results = self.interaction(text_hidden_states, audio_hidden_states, video_hidden_states, self.text_map1, self.text_map2_text, self.text_map2_audio, self.text_map2_video, self.text_attention, self.layerNorm_text, self.layerNorm_audio, self.layerNorm_video, self.layer_classifier)
        audio_inte_embedding, audio_layer_pred_results = self.interaction(audio_hidden_states, text_hidden_states, video_hidden_states, self.audio_map1, self.audio_map2_audio, self.audio_map2_text, self.audio_map2_video, self.audio_attention, self.layerNorm_audio, self.layerNorm_text, self.layerNorm_video, self.layer_classifier)
        video_inte_embedding, video_layer_pred_results = self.interaction(video_hidden_states, text_hidden_states, audio_hidden_states, self.video_map1, self.video_map2_video, self.video_map2_text, self.video_map2_audio, self.video_attention, self.layerNorm_video, self.layerNorm_text, self.layerNorm_audio, self.layer_classifier)
        text_cls = text_hidden_states[-1, :, 0, :]
        audio_cls = audio_hidden_states[-1, :, 0, :]
        video_cls = video_hidden_states[-1, :, 0, :]
        text_pred = self.text_predict(text_cls)
        video_pred = self.video_predict(video_cls)
        audio_pred = self.audio_predict(audio_cls)
        multimode_pred = self.fusion(torch.cat((text_inte_embedding, text_cls, video_inte_embedding, video_cls, audio_inte_embedding, audio_cls),-1))
        result = self.weighted_fusion(torch.stack((text_pred, video_pred, audio_pred, multimode_pred),-1)).squeeze(-1)
        return result, text_layer_pred_results, audio_layer_pred_results, video_layer_pred_results