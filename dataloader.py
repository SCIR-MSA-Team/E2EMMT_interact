# -*- coding: utf-8 -*-
# @Time    : 6/19/21 12:23 AM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : dataloader.py

# modified from:
# Author: David Harwath
# with some functions borrowed from https://github.com/SeanNaren/deepspeech.pytorch

import csv
import json
import torchaudio
import numpy as np
import torch
import glob
import os
import torch.nn.functional
from torch.utils.data import Dataset
import random
from torchvision import transforms
from PIL import Image

def make_index_dict(label_csv):
    index_lookup = {}
    with open(label_csv, 'r') as f:
        csv_reader = csv.DictReader(f)
        line_count = 0
        for row in csv_reader:
            index_lookup[row['mid']] = row['index']
            line_count += 1
    return index_lookup

def make_name_dict(label_csv):
    name_lookup = {}
    with open(label_csv, 'r') as f:
        csv_reader = csv.DictReader(f)
        line_count = 0
        for row in csv_reader:
            name_lookup[row['index']] = row['display_name']
            line_count += 1
    return name_lookup

def lookup_list(index_list, label_csv):
    label_list = []
    table = make_name_dict(label_csv)
    for item in index_list:
        label_list.append(table[item])
    return label_list

def preemphasis(signal,coeff=0.97):
    """perform preemphasis on the input signal.

    :param signal: The signal to filter.
    :param coeff: The preemphasis coefficient. 0 is none, default 0.97.
    :returns: the filtered signal.
    """
    return np.append(signal[0],signal[1:]-coeff*signal[:-1])

class MultimodalDataset(Dataset):
    def __init__(self, dataset_json_file, conf, label_map, face_model, face_size, tokenizer_model):
        """
        Dataset that manages video recordings
        :param conf: Dictionary containing the audio loading and preprocessing settings
        :param dataset_json_file
        """
        self.datapath = dataset_json_file
        with open(dataset_json_file, 'r') as fp:
            data_json = json.load(fp)

        self.data = data_json
        self.conf = conf
        self.dataset = self.conf.get('dataset')
        print('now process ' + self.dataset)

        print('---------------the {:s} dataloader---------------'.format(self.conf.get('mode')))
        self.melbins = self.conf.get('num_mel_bins')
        self.norm_mean = self.conf.get('mean')
        self.norm_std = self.conf.get('std')
        self.time_dim_split = self.conf.get('time_dim_split')

        self.label_map = label_map
        self.label_num = len(self.label_map)
        print('number of classes is {:d}'.format(self.label_num))
        self.crop = transforms.CenterCrop(360)
        self.face_size = face_size
        self.mtcnn = face_model
        self.limited_W = self.conf.get('num_width')
        self.limited_H = self.conf.get('num_height')
        self.limited_patch_num = (self.limited_W // 16) * (self.limited_W // 16)
        self.tokenizer = tokenizer_model


    def getPosWeight(self):
        sums = None
        for index in range(len(self.data)):
            label = np.array(self.data[index]['labels'])
            if sums is None:
                sums = label
            else:
                sums += label
        pos_nums = sums
        neg_nums = len(self.data) - pos_nums
        pos_weight = neg_nums / pos_nums
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.FloatTensor(pos_weight).to(device)

    def _wav2fbank(self, filename):

        waveform, sr = torchaudio.load(filename)
        waveform = waveform - waveform.mean()
        fbank = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=sr, use_energy=False,
                                                  window_type='hanning', num_mel_bins=self.melbins, dither=0.0, frame_shift=10)

        target_length = self.conf.get('target_length')
        n_frames = fbank.shape[0]
        p = target_length - n_frames

        # cut and pad
        if p > 0:
            m = torch.nn.ZeroPad2d((0, 0, 0, p))
            fbank = m(fbank)
        elif p < 0:
            fbank = fbank[0:target_length, :]

        return fbank


    def _video2img(self, wav_file):
        wav_name = wav_file.split("/")[-1]
        if "R" in wav_name or "L" in wav_name:
            dir_name = wav_file[:-12]
            prefix_name = wav_file[-5]
            files = glob.glob(f'{dir_name}/*')
            nums = (len(files) - 5) // 2
            step = int(500 / 1000 * 30)
            sampled = [os.path.join(dir_name, f'image_{prefix_name}_{i}.jpg') for i in list(range(0, nums, step))]
 
            sampledImgs = []
            for imgPath in sampled:
                this_img = Image.open(imgPath)
                sampledImgs.append(np.float32(this_img))

        else:
            dir_name = wav_file[:-10]
            files = glob.glob(f'{dir_name}/*')
            nums = len(files) - 1
            step = int(500 / 1000 * 30)
            sampled = [os.path.join(dir_name, f'image_{i}.jpg') for i in list(range(0, nums, step))]
            if len(sampled) == 0:
                step = int(500 / 1000 * 30) // 4
                sampled = [os.path.join(dir_name, f'image_{i}.jpg') for i in list(range(0, nums, step))]

            sampledImgs = []
            for imgPath in sampled:
                this_img = Image.open(imgPath)
                H = np.float32(this_img).shape[0]
                W = np.float32(this_img).shape[1]
                if H > 360:
                    resize = transforms.Resize([H // 2, W // 2])
                    this_img = resize(this_img)
                this_img = self.crop(this_img)
                sampledImgs.append(np.float32(this_img))

        sampledImgs = np.array(sampledImgs)
        if len(sampledImgs) > 0:
            faces = self.mtcnn(sampledImgs)
        else:
            return None

        output_faces = []
        for i, face in enumerate(faces):
                if face is not None:
                    face = face.cpu()
                    face = face.view(3, self.face_size//16, 16, self.face_size//16, 16)
                    face = face.permute(1, 3, 2, 4, 0)
                    output_faces.append(face.contiguous().view(-1, 16, 16, 3))
        if len(output_faces) > 0:
            patch_num = (self.face_size//16) * (self.face_size//16) * len(output_faces)
            output_faces = torch.cat(output_faces, dim=0)
            
            if patch_num > self.limited_patch_num:
                output_faces = output_faces[:self.limited_patch_num, :, :, :]
            else:
                output_faces = torch.cat([output_faces, torch.zeros(self.limited_patch_num-patch_num, 16, 16, 3)], dim=0)
        else:
            patch_num = 0
            output_faces = torch.zeros(self.limited_patch_num-patch_num, 16, 16, 3)
        
        output_faces = output_faces.view((self.limited_H // 16), (self.limited_W // 16), 16, 16, 3)
        # channels * H//16 * 16 * W//16 * 16
        output_faces = output_faces.permute(4, 0, 2, 1, 3)
        output_faces = output_faces.contiguous().view(3, self.limited_H, self.limited_W)

        return output_faces
        


    def __getitem__(self, index):
        """
        returns: image, audio, nframes
        where image is a FloatTensor of size (3, H, W)
        audio is a FloatTensor of size (N_freq, N_frames) for spectrogram, or (N_frames) for waveform
        nframes is an integer
        """
        
        datum = self.data[index]
        fbank = self._wav2fbank(datum['wav'])
        text = datum['text']
        fbank = (fbank - self.norm_mean) / (self.norm_std * 2)

        # adapt multi-faces to a single image 256 * 256
        images = self._video2img(datum['wav'])
        label_indices = np.array(datum['labels'])
        label_indices = torch.FloatTensor(label_indices)

        # use 2*128 patch
        # self.time_dim_split = True
        if self.time_dim_split == True:
            pre_t, pre_f = fbank.size()
            groups = torch.split(fbank, 2, 0)
            _patchs = []
            for item in groups:
                _patchs.append(item.view(16, 16))
            _patchs = torch.stack(_patchs)
            _patchs = _patchs.view(pre_t//16, pre_f//16, 16 ,16)
            transfer_img = _patchs.permute(0,2,1,3).reshape(pre_t, pre_f)
            fbank = transfer_img

        # the output audio shape is [time_frame_num, frequency_bins]
        # the output images shape is [3, self.limited_H, self.limited_W]
        
        return fbank, images, text, label_indices

    def __len__(self):
        return len(self.data)

def collate_fn(batch):
    fbanks = []
    imgs = []
    labels = []
    texts = []
    for dp in batch:
        fbank, sampledImgs, text, label = dp
        if sampledImgs is None:
            continue
        fbanks.append(fbank)
        imgs.append(sampledImgs)
        labels.append(label)
        texts.append(text)
    return torch.stack(fbanks), torch.stack(imgs), texts, torch.stack(labels)
