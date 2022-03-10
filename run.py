# -*- coding: utf-8 -*-
# @Time    : 6/11/21 12:57 AM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : run.py

import argparse
import os
import shutil
import ast
import pickle
import sys
import time
import torch
from torch.utils.data import WeightedRandomSampler
basepath = os.path.dirname(os.path.dirname(sys.path[0]))
sys.path.append(basepath)
print('basepath',basepath)
import dataloader
from models import ast_models_new, only_video, video_text, ast_models_video_deit, ast_models_cls, text_video, text_audio, video_audio, test_layer_k_last, test_layer_k_select, ast_interact_no_attention, ast_models_video_deit_transformer_freeze, ast_models_cls_transformer_freeze, ast_models_video_deit_transformer_freeze_no_text, ast_models_cls_transformer_freeze_no_text, ast_models_audio_interact, ast_models_text_interact, ast_models_video_interact
import numpy as np
from traintest import train, validate
from tabulate import tabulate
import torch.nn as nn
from facenet_pytorch import MTCNN
from transformers import BertTokenizer

import warnings
warnings.filterwarnings("ignore", category=Warning)

# import wandb
# wandb.init(project='ast_model_video_deit_iemocap')

print("I am process %s, running on %s: starting (%s)" % (os.getpid(), os.uname()[1], time.asctime()))

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# parser.add_argument("--data-train", type=str, default='train_iemocap_dataset.json', help="training data json")
# parser.add_argument("--data-val", type=str, default='valid_iemocap_dataset.json', help="validation data json")
# parser.add_argument("--data-eval", type=str, default='test_iemocap_dataset.json', help="evaluation data json")
parser.add_argument("--label-csv", type=str, default='', help="csv with class labels")
parser.add_argument("--n_class", type=int, default=6, help="number of classes")
parser.add_argument("--model", type=str, default='ast', help="the model used")
parser.add_argument("--dataset", type=str, default="iemocap", help="the dataset used", choices=["iemocap", "mosei"])
parser.add_argument("--modal", type=str, default='tav', help="the modality used")

parser.add_argument("--text_lr_factor", type=int, default=100, help="text_lr_factor")
parser.add_argument("--exp-dir", type=str, default="exp1202", help="directory to dump experiments")
parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float, metavar='LR', help='initial learning rate')
parser.add_argument("--optim", type=str, default="adam", help="training optimizer", choices=["sgd", "adam"])
parser.add_argument('--batch-size', default=8, type=int, metavar='N', help='mini-batch size')
parser.add_argument('-w', '--num-workers', default=10, type=int, metavar='NW', help='# of workers for dataloading (default: 32)')
parser.add_argument("--n-epochs", type=int, default=20, help="number of maximum training epochs")
# not used in the formal experiments
parser.add_argument("--lr_patience", type=int, default=2, help="how many epoch to wait to reduce lr if acc doesn't improve")
parser.add_argument("--early_stop", type=int, default=5, help="how many epoch to wait to stop training if acc doesn't improve")
parser.add_argument("--face_size", type=int, default=128, help="face size")
parser.add_argument('--text-max-len', type=int, default=300, help='Max length of text after tokenization')
parser.add_argument("--scale", type=float, default=0, help="sclae value for attention")
parser.add_argument("--v_patch_num", type=int, default=256, help="the number of video patch")
parser.add_argument("--a_patch_num", type=int, default=256, help="the num of audio patch")


parser.add_argument("--n-print-steps", type=int, default=100, help="number of steps to print statistics")
parser.add_argument('--save_model', help='save the model or not', type=ast.literal_eval)

parser.add_argument('--freqm', help='frequency mask max length', type=int, default=0)
parser.add_argument('--timem', help='time mask max length', type=int, default=0)
parser.add_argument("--mixup", type=float, default=0, help="how many (0-1) samples need to be mixup during training")
parser.add_argument("--bal", type=str, default=None, help="use balanced sampling or not")
# the stride used in patch spliting, e.g., for patch size 16*16, a stride of 16 means no overlapping, a stride of 10 means overlap of 6.
parser.add_argument("--fstride", type=int, default=16, help="soft split freq stride, overlap=patch_size-stride")
parser.add_argument("--tstride", type=int, default=16, help="soft split time stride, overlap=patch_size-stride")
parser.add_argument('--v_imagenet_pretrain', help='if use ImageNet pretrained audio spectrogram transformer model', type=ast.literal_eval, default='True')
parser.add_argument('--a_imagenet_pretrain', help='if use ImageNet pretrained audio spectrogram transformer model', type=ast.literal_eval, default='True')
parser.add_argument('--audioset_pretrain', help='if use ImageNet and audioset pretrained audio spectrogram transformer model', type=ast.literal_eval, default='False')
parser.add_argument('--time_dim_split', help='if use time-dim split', type=ast.literal_eval, default='True')
parser.add_argument("--seed",type=int,required=True)
parser.add_argument('--num_height', required=True, type=int)
parser.add_argument('--num_width',required=True,type=int)
parser.add_argument('--k',default=-1,type=int)

args = parser.parse_args()
seed = args.seed
torch.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if args.dataset == 'mosei':
    args.data_train = 'train_mosei_dataset.json'
    args.data_val =  'valid_mosei_dataset.json'
    args.data_eval = 'test_mosei_dataset.json'
elif args.dataset == 'iemocap':
    args.data_train = 'train_iemocap_dataset.json'
    args.data_val =  'valid_iemocap_dataset.json'
    args.data_eval = 'test_iemocap_dataset.json'

print('args.batch_size',args.batch_size)
print('args',args)

# dataset spectrogram mean and std, used to normalize the input
norm_stats = {'mosei':[-4.292258, 4.324287], 'iemocap':[-3.714969, 4.8446374]}
# image, audio
target_length = {'mosei':1024, 'iemocap':1024}
# if add noise for data augmentation, only use for speech commands
noise = {'mosei': False, 'iemocap': False}
label_maps = {"mosei": ['anger', 'disgust', 'fear', 'happy', 'sad', 'surprise'],
                  "iemocap": ['ang', 'exc', 'fru', 'hap', 'neu', 'sad']}

conf = {'num_mel_bins': 128, 'num_height': args.num_height, 'num_width': args.num_width, 'target_length': target_length[args.dataset], 'freqm': args.freqm, 'timem': args.timem, 'mixup': args.mixup, 'dataset': args.dataset, 'mode':'train', 'mean':norm_stats[args.dataset][0], 'std':norm_stats[args.dataset][1],
                  'noise':noise[args.dataset], 'time_dim_split':args.time_dim_split}
val_conf = {'num_mel_bins': 128, 'num_height': args.num_height, 'num_width': args.num_width, 'target_length': target_length[args.dataset], 'freqm': 0, 'timem': 0, 'mixup': 0, 'dataset': args.dataset, 'mode':'evaluation', 'mean':norm_stats[args.dataset][0], 'std':norm_stats[args.dataset][1], 'noise':False,
                'time_dim_split':args.time_dim_split}

# mtcnn = MTCNN(image_size=args.face_size, margin=0, post_process=False, device="cpu")
mtcnn = MTCNN(image_size=args.face_size, margin=10, selection_method="probability", post_process=False, device='cpu')
tokenizer_model = BertTokenizer.from_pretrained("/users10/zyzhang/graduationProject/data/pretrain_model/bert_base_uncased")
# tokenizer_model = BertTokenizer.from_pretrained("/users5/ywu/MMSA/pretrained_model/bert_en")
trainset = dataloader.MultimodalDataset(args.data_train, label_map=label_maps[args.dataset], conf=conf, face_model=mtcnn, face_size=args.face_size, tokenizer_model=tokenizer_model)
train_loader = torch.utils.data.DataLoader(
    trainset,
    collate_fn=dataloader.collate_fn,
    batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
val_loader = torch.utils.data.DataLoader(
    dataloader.MultimodalDataset(args.data_val, label_map=label_maps[args.dataset], conf=val_conf, face_model=mtcnn, face_size=args.face_size, tokenizer_model=tokenizer_model),
    collate_fn=dataloader.collate_fn,
    batch_size=args.batch_size*2, shuffle=False, num_workers=args.num_workers, pin_memory=True)

# transformer based model
if args.model == 'ast':
    print('now train a audio spectrogram transformer model')
    audio_model = ast_models_new.ASTModel(label_dim=args.n_class, fstride=args.fstride, tstride=args.tstride, input_fdim=128,
                                  input_tdim=target_length[args.dataset], imagenet_pretrain=args.a_imagenet_pretrain,
                                  audioset_pretrain=False, model_size='base384', patch_num=args.a_patch_num)

    video_model = ast_models_new.VTModel(label_dim=args.n_class, fstride=args.fstride, tstride=args.tstride, input_fdim=args.num_height,
                                  input_tdim=args.num_width, imagenet_pretrain=args.v_imagenet_pretrain,
                                  audioset_pretrain=False, model_size='base384', patch_num=args.v_patch_num)

    text_model = ast_models_new.TTModel(num_classes=args.n_class)

    mt_model = ast_models_new.MTModel(args.n_class, audio_model, video_model, text_model)

    args.PosWeight = trainset.getPosWeight()
elif args.model == 'only_video':
    video_model = only_video.VTModel(label_dim=args.n_class, fstride=args.fstride, tstride=args.tstride, input_fdim=args.num_height,
                                  input_tdim=args.num_width, imagenet_pretrain=args.v_imagenet_pretrain,
                                  audioset_pretrain=False, model_size='base384', patch_num=args.v_patch_num)
    mt_model = only_video.only_video(args.n_class,video_model)
elif args.model == 'video_text':
    video_model = video_text.VTModel(label_dim=args.n_class, fstride=args.fstride, tstride=args.tstride, input_fdim=args.num_height,
                                  input_tdim=args.num_width, imagenet_pretrain=args.v_imagenet_pretrain,
                                  audioset_pretrain=False, model_size='base384', patch_num=args.v_patch_num)

    text_model = video_text.TTModel(num_classes=args.n_class)
    mt_model = video_text.video_text(label_dim=args.n_class, audio_model=None, video_model=video_model, text_model=text_model)
elif args.model == 'ast_model_video_deit':
    audio_model = ast_models_video_deit.ASTModel(label_dim=args.n_class, fstride=args.fstride, tstride=args.tstride, input_fdim=128,
                                  input_tdim=target_length[args.dataset], imagenet_pretrain=args.a_imagenet_pretrain,
                                  audioset_pretrain=False, model_size='base384', patch_num=args.a_patch_num)

    video_model = ast_models_video_deit.VTModel_deit(label_dim=args.n_class, fstride=args.fstride, tstride=args.tstride, input_fdim=args.num_height,
                                  input_tdim=args.num_width, imagenet_pretrain=args.v_imagenet_pretrain,
                                  audioset_pretrain=False, model_size='base384', patch_num=args.v_patch_num)

    text_model = ast_models_video_deit.TTModel(num_classes=args.n_class)

    mt_model = ast_models_video_deit.MTModel(label_dim=args.n_class, audio_model=audio_model, video_model=video_model, text_model=text_model)
elif args.model == 'ast_model_cls':
    audio_model = ast_models_cls.ASTModel(label_dim=args.n_class, fstride=args.fstride, tstride=args.tstride, input_fdim=128,
                                  input_tdim=target_length[args.dataset], imagenet_pretrain=args.a_imagenet_pretrain,
                                  audioset_pretrain=False, model_size='base384', patch_num=args.a_patch_num)
    video_model = ast_models_cls.VTModel_deit(label_dim=args.n_class, fstride=args.fstride, tstride=args.tstride, input_fdim=args.num_height,
                                  input_tdim=args.num_width, imagenet_pretrain=args.v_imagenet_pretrain,
                                  audioset_pretrain=False, model_size='base384', patch_num=args.v_patch_num)
    text_model = ast_models_cls.TTModel(num_classes=args.n_class)
    mt_model = ast_models_cls.MTModel(label_dim=args.n_class, audio_model=audio_model, video_model=video_model, text_model=text_model)
elif args.model == 'text_video':
    text_model = text_video.TTModel(num_classes=args.n_class)
    video_model = text_video.VTModel_deit(label_dim=args.n_class, fstride=args.fstride, tstride=args.tstride, input_fdim=args.num_height,
                                  input_tdim=args.num_width, imagenet_pretrain=args.v_imagenet_pretrain,
                                  audioset_pretrain=False, model_size='base384', patch_num=args.v_patch_num)
    mt_model = text_video.MTModel(label_dim=args.n_class, video_model=video_model, text_model=text_model)
elif args.model == 'text_audio':
    text_model = text_audio.TTModel(num_classes=args.n_class)
    audio_model = text_audio.ASTModel(label_dim=args.n_class, fstride=args.fstride, tstride=args.tstride, input_fdim=128,
                                  input_tdim=target_length[args.dataset], imagenet_pretrain=args.a_imagenet_pretrain,
                                  audioset_pretrain=False, model_size='base384', patch_num=args.a_patch_num)
    mt_model = text_audio.MTModel(label_dim=args.n_class, audio_model=audio_model, text_model=text_model)
elif args.model == 'video_audio':
    video_model = video_audio.VTModel_deit(label_dim=args.n_class, fstride=args.fstride, tstride=args.tstride, input_fdim=args.num_height,
                                  input_tdim=args.num_width, imagenet_pretrain=args.v_imagenet_pretrain,
                                  audioset_pretrain=False, model_size='base384', patch_num=args.v_patch_num)
    audio_model = video_audio.ASTModel(label_dim=args.n_class, fstride=args.fstride, tstride=args.tstride, input_fdim=128,
                                  input_tdim=target_length[args.dataset], imagenet_pretrain=args.a_imagenet_pretrain,
                                  audioset_pretrain=False, model_size='base384', patch_num=args.a_patch_num)
    mt_model = video_audio.MTModel(label_dim=args.n_class, audio_model=audio_model, video_model=video_model)
elif args.model == 'test_layer_k_last':
    audio_model = test_layer_k_last.ASTModel(label_dim=args.n_class, fstride=args.fstride, tstride=args.tstride, input_fdim=128,
                                  input_tdim=target_length[args.dataset], imagenet_pretrain=args.a_imagenet_pretrain,
                                  audioset_pretrain=False, model_size='base384', patch_num=args.a_patch_num)

    video_model = test_layer_k_last.VTModel_deit(label_dim=args.n_class, fstride=args.fstride, tstride=args.tstride, input_fdim=args.num_height,
                                  input_tdim=args.num_width, imagenet_pretrain=args.v_imagenet_pretrain,
                                  audioset_pretrain=False, model_size='base384', patch_num=args.v_patch_num)

    text_model = test_layer_k_last.TTModel(num_classes=args.n_class)

    mt_model = test_layer_k_last.MTModel(label_dim=args.n_class, audio_model=audio_model, video_model=video_model, text_model=text_model, k=args.k)
elif args.model == 'test_layer_k_select':
    audio_model = test_layer_k_select.ASTModel(label_dim=args.n_class, fstride=args.fstride, tstride=args.tstride, input_fdim=128,
                                  input_tdim=target_length[args.dataset], imagenet_pretrain=args.a_imagenet_pretrain,
                                  audioset_pretrain=False, model_size='base384', patch_num=args.a_patch_num)

    video_model = test_layer_k_select.VTModel_deit(label_dim=args.n_class, fstride=args.fstride, tstride=args.tstride, input_fdim=args.num_height,
                                  input_tdim=args.num_width, imagenet_pretrain=args.v_imagenet_pretrain,
                                  audioset_pretrain=False, model_size='base384', patch_num=args.v_patch_num)

    text_model = test_layer_k_select.TTModel(num_classes=args.n_class)

    mt_model = test_layer_k_select.MTModel(label_dim=args.n_class, audio_model=audio_model, video_model=video_model, text_model=text_model, k=args.k)
elif args.model == 'ast_interact_no_attention':
    audio_model = ast_interact_no_attention.ASTModel(label_dim=args.n_class, fstride=args.fstride, tstride=args.tstride, input_fdim=128,
                                  input_tdim=target_length[args.dataset], imagenet_pretrain=args.a_imagenet_pretrain,
                                  audioset_pretrain=False, model_size='base384', patch_num=args.a_patch_num)
    video_model = ast_interact_no_attention.VTModel_deit(label_dim=args.n_class, fstride=args.fstride, tstride=args.tstride, input_fdim=args.num_height,
                                  input_tdim=args.num_width, imagenet_pretrain=args.v_imagenet_pretrain,
                                  audioset_pretrain=False, model_size='base384', patch_num=args.v_patch_num)
    text_model = ast_interact_no_attention.TTModel(num_classes=args.n_class)
    mt_model = ast_interact_no_attention.MTModel(label_dim=args.n_class, audio_model=audio_model, video_model=video_model, text_model=text_model)
elif args.model == 'ast_models_video_deit_transformer_freeze':
    audio_model = ast_models_video_deit_transformer_freeze.ASTModel(label_dim=args.n_class, fstride=args.fstride, tstride=args.tstride, input_fdim=128,
                                  input_tdim=target_length[args.dataset], imagenet_pretrain=args.a_imagenet_pretrain,
                                  audioset_pretrain=False, model_size='base384', patch_num=args.a_patch_num)
    video_model = ast_models_video_deit_transformer_freeze.VTModel_deit(label_dim=args.n_class, fstride=args.fstride, tstride=args.tstride, input_fdim=args.num_height,
                                  input_tdim=args.num_width, imagenet_pretrain=args.v_imagenet_pretrain,
                                  audioset_pretrain=False, model_size='base384', patch_num=args.v_patch_num)
    text_model = ast_models_video_deit_transformer_freeze.TTModel(num_classes=args.n_class)
    mt_model = ast_models_video_deit_transformer_freeze.MTModel(label_dim=args.n_class, audio_model=audio_model, video_model=video_model, text_model=text_model)
elif args.model == 'ast_models_cls_transformer_freeze':
    audio_model = ast_models_cls_transformer_freeze.ASTModel(label_dim=args.n_class, fstride=args.fstride, tstride=args.tstride, input_fdim=128,
                                  input_tdim=target_length[args.dataset], imagenet_pretrain=args.a_imagenet_pretrain,
                                  audioset_pretrain=False, model_size='base384', patch_num=args.a_patch_num)
    video_model = ast_models_cls_transformer_freeze.VTModel_deit(label_dim=args.n_class, fstride=args.fstride, tstride=args.tstride, input_fdim=args.num_height,
                                  input_tdim=args.num_width, imagenet_pretrain=args.v_imagenet_pretrain,
                                  audioset_pretrain=False, model_size='base384', patch_num=args.v_patch_num)
    text_model = ast_models_cls_transformer_freeze.TTModel(num_classes=args.n_class)
    mt_model = ast_models_cls_transformer_freeze.MTModel(label_dim=args.n_class, audio_model=audio_model, video_model=video_model, text_model=text_model)
elif args.model == 'ast_models_video_deit_transformer_freeze_no_text':
    audio_model = ast_models_video_deit_transformer_freeze_no_text.ASTModel(label_dim=args.n_class, fstride=args.fstride, tstride=args.tstride, input_fdim=128,
                                  input_tdim=target_length[args.dataset], imagenet_pretrain=args.a_imagenet_pretrain,
                                  audioset_pretrain=False, model_size='base384', patch_num=args.a_patch_num)
    video_model = ast_models_video_deit_transformer_freeze_no_text.VTModel_deit(label_dim=args.n_class, fstride=args.fstride, tstride=args.tstride, input_fdim=args.num_height,
                                  input_tdim=args.num_width, imagenet_pretrain=args.v_imagenet_pretrain,
                                  audioset_pretrain=False, model_size='base384', patch_num=args.v_patch_num)
    text_model = ast_models_video_deit_transformer_freeze_no_text.TTModel(num_classes=args.n_class)
    mt_model = ast_models_video_deit_transformer_freeze_no_text.MTModel(label_dim=args.n_class, audio_model=audio_model, video_model=video_model, text_model=text_model)
elif args.model == 'ast_models_cls_transformer_freeze_no_text':
    audio_model = ast_models_cls_transformer_freeze_no_text.ASTModel(label_dim=args.n_class, fstride=args.fstride, tstride=args.tstride, input_fdim=128,
                                  input_tdim=target_length[args.dataset], imagenet_pretrain=args.a_imagenet_pretrain,
                                  audioset_pretrain=False, model_size='base384', patch_num=args.a_patch_num)
    video_model = ast_models_cls_transformer_freeze_no_text.VTModel_deit(label_dim=args.n_class, fstride=args.fstride, tstride=args.tstride, input_fdim=args.num_height,
                                  input_tdim=args.num_width, imagenet_pretrain=args.v_imagenet_pretrain,
                                  audioset_pretrain=False, model_size='base384', patch_num=args.v_patch_num)
    text_model = ast_models_cls_transformer_freeze_no_text.TTModel(num_classes=args.n_class)
    mt_model = ast_models_cls_transformer_freeze_no_text.MTModel(label_dim=args.n_class, audio_model=audio_model, video_model=video_model, text_model=text_model)
elif args.model == 'ast_models_audio_interact':
    audio_model = ast_models_audio_interact.ASTModel(label_dim=args.n_class, fstride=args.fstride, tstride=args.tstride, input_fdim=128,
                                  input_tdim=target_length[args.dataset], imagenet_pretrain=args.a_imagenet_pretrain,
                                  audioset_pretrain=False, model_size='base384', patch_num=args.a_patch_num)
    video_model = ast_models_audio_interact.VTModel_deit(label_dim=args.n_class, fstride=args.fstride, tstride=args.tstride, input_fdim=args.num_height,
                                  input_tdim=args.num_width, imagenet_pretrain=args.v_imagenet_pretrain,
                                  audioset_pretrain=False, model_size='base384', patch_num=args.v_patch_num)
    text_model = ast_models_audio_interact.TTModel(num_classes=args.n_class)
    mt_model = ast_models_audio_interact.MTModel(label_dim=args.n_class, audio_model=audio_model, video_model=video_model, text_model=text_model)
elif args.model == 'ast_models_text_interact':
    audio_model = ast_models_text_interact.ASTModel(label_dim=args.n_class, fstride=args.fstride, tstride=args.tstride, input_fdim=128,
                                  input_tdim=target_length[args.dataset], imagenet_pretrain=args.a_imagenet_pretrain,
                                  audioset_pretrain=False, model_size='base384', patch_num=args.a_patch_num)
    video_model = ast_models_text_interact.VTModel_deit(label_dim=args.n_class, fstride=args.fstride, tstride=args.tstride, input_fdim=args.num_height,
                                  input_tdim=args.num_width, imagenet_pretrain=args.v_imagenet_pretrain,
                                  audioset_pretrain=False, model_size='base384', patch_num=args.v_patch_num)
    text_model = ast_models_text_interact.TTModel(num_classes=args.n_class)
    mt_model = ast_models_text_interact.MTModel(label_dim=args.n_class, audio_model=audio_model, video_model=video_model, text_model=text_model)
elif args.model == 'ast_models_video_interact':
    audio_model = ast_models_video_interact.ASTModel(label_dim=args.n_class, fstride=args.fstride, tstride=args.tstride, input_fdim=128,
                                  input_tdim=target_length[args.dataset], imagenet_pretrain=args.a_imagenet_pretrain,
                                  audioset_pretrain=False, model_size='base384', patch_num=args.a_patch_num)
    video_model = ast_models_video_interact.VTModel_deit(label_dim=args.n_class, fstride=args.fstride, tstride=args.tstride, input_fdim=args.num_height,
                                  input_tdim=args.num_width, imagenet_pretrain=args.v_imagenet_pretrain,
                                  audioset_pretrain=False, model_size='base384', patch_num=args.v_patch_num)
    text_model = ast_models_video_interact.TTModel(num_classes=args.n_class)
    mt_model = ast_models_video_interact.MTModel(label_dim=args.n_class, audio_model=audio_model, video_model=video_model, text_model=text_model)

if args.n_epochs > 0:
    print("\nCreating experiment directory: %s" % args.exp_dir)
    os.makedirs("%s/models" % args.exp_dir)
    with open("%s/args.pkl" % args.exp_dir, "wb") as f:
        pickle.dump(args, f)

print('Now starting training for {:d} epochs'.format(args.n_epochs))
train(mt_model, train_loader, val_loader, args, tokenizer_model)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sd = torch.load(args.exp_dir + '/models/best_mmt_model.pth', map_location=device)
mt_model = torch.nn.DataParallel(mt_model)
mt_model.load_state_dict(sd)

# best model on the validation set
stats, _ = validate(mt_model, val_loader, args, 'valid_set', tokenizer_model)
accs, recalls, precisions, f1s, aucs, best_thresholds = stats

if args.dataset == 'mosei':
    annotations = ['anger', 'disgust', 'fear', 'happy', 'sad', 'surprise']
elif args.dataset == 'iemocap':
    annotations = ['ang', 'exc', 'fru', 'hap', 'neu', 'sad']

headers = [
        ['phase (acc)', *annotations, 'average'],
        ['phase (recall)', *annotations, 'average'],
        ['phase (precision)', *annotations, 'average'],
        ['phase (f1)', *annotations, 'average'],
        ['phase (auc)', *annotations, 'average']
    ]
for i in range(len(headers)):
    content_str = ["{:.4f}".format(content) for content in stats[i]]
    print(tabulate([
                    ['Valid', *content_str],
    ], headers=headers[i]))

# test the model on the evaluation set
eval_loader = torch.utils.data.DataLoader(
    dataloader.MultimodalDataset(args.data_eval, label_map=label_maps[args.dataset], conf=val_conf, face_model=mtcnn, face_size=args.face_size, tokenizer_model=tokenizer_model),
    collate_fn=dataloader.collate_fn,
    batch_size=args.batch_size*2, shuffle=False, num_workers=args.num_workers, pin_memory=True)
stats, test_loss = validate(mt_model, eval_loader, args, 'eval_set', tokenizer_model, best_thresholds)
accs, recalls, precisions, f1s, aucs, best_thresholds = stats

if args.dataset == 'mosei':
    annotations = ['anger', 'disgust', 'fear', 'happy', 'sad', 'surprise']
elif args.dataset == 'iemocap':
    annotations = ['ang', 'exc', 'fru', 'hap', 'neu', 'sad']

headers = [
        ['phase (acc)', *annotations, 'average'],
        ['phase (recall)', *annotations, 'average'],
        ['phase (precision)', *annotations, 'average'],
        ['phase (f1)', *annotations, 'average'],
        ['phase (auc)', *annotations, 'average']
    ]

print('---------------evaluate on the test set---------------')

for i in range(len(headers)):
    content_str = ["{:.4f}".format(content) for content in stats[i]]
    print(tabulate([
                    ['Test', *content_str],
    ], headers=headers[i]))
tmp = [accs[-1], recalls[-1], precisions[-1], f1s[-1], aucs[-1], test_loss]
tmp.extend(best_thresholds)
np.savetxt(args.exp_dir + '/eval_result.csv', tmp)

