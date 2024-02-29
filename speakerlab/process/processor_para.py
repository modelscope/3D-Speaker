# Copyright 3D-Speaker (https://github.com/alibaba-damo-academy/3D-Speaker). All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import random
import pickle
import torch
import numpy as np
import torchaudio
import torch.nn.functional as F
import torchaudio.compliance.kaldi as Kaldi

from speakerlab.process.augmentation import NoiseReverbCorrupter
from speakerlab.utils.fileio import load_data_csv


def apply_lfr(inputs, lfr_m, lfr_n):
    LFR_inputs = []
    T = inputs.shape[0]
    T_lfr = int(np.ceil(T / lfr_n))
    left_padding = inputs[0].repeat((lfr_m - 1) // 2, 1)
    inputs = torch.vstack((left_padding, inputs))
    T = T + (lfr_m - 1) // 2
    for i in range(T_lfr):
        if lfr_m <= T - i * lfr_n:
            LFR_inputs.append((inputs[i * lfr_n:i * lfr_n + lfr_m]).view(1, -1))
        else:  # process last LFR frame
            num_padding = lfr_m - (T - i * lfr_n)
            frame = (inputs[i * lfr_n:]).view(-1)
            for _ in range(num_padding):
                frame = torch.hstack((frame, inputs[-1]))
            LFR_inputs.append(frame)
    LFR_outputs = torch.vstack(LFR_inputs)
    return LFR_outputs.type(torch.float32)

def apply_cmvn(inputs, cmvn):  # noqa
    """
    Apply CMVN with mvn data
    """
    device = inputs.device
    dtype = inputs.dtype
    frame, dim = inputs.shape

    means = cmvn[0:1, :dim]
    vars = cmvn[1:2, :dim]
    inputs += means.to(device)
    inputs *= vars.to(device)

    return inputs.type(torch.float32)

def load_cmvn(cmvn_file):
    with open(cmvn_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    means_list = []
    vars_list = []
    for i in range(len(lines)):
        line_item = lines[i].split()
        if line_item[0] == '<AddShift>':
            line_item = lines[i + 1].split()
            if line_item[0] == '<LearnRateCoef>':
                add_shift_line = line_item[3:(len(line_item) - 1)]
                means_list = list(add_shift_line)
                continue
        elif line_item[0] == '<Rescale>':
            line_item = lines[i + 1].split()
            if line_item[0] == '<LearnRateCoef>':
                rescale_line = line_item[3:(len(line_item) - 1)]
                vars_list = list(rescale_line)
                continue
    means = np.array(means_list).astype(np.float32)
    vars = np.array(vars_list).astype(np.float32)
    cmvn = np.array([means, vars])
    cmvn = torch.as_tensor(cmvn, dtype=torch.float32)
    return cmvn

cmvn_data = load_cmvn('pretrained/am.mvn')


class WavReader(object):
    def __init__(self,
        sample_rate = 16000,
        duration: float = 3.0,
        speed_pertub: bool = False,
        lm: bool = True,
    ):
        self.duration = duration
        self.sample_rate = sample_rate
        self.speed_pertub = speed_pertub
        self.lm = lm

    def __call__(self, wav_path):
        wav, sr = torchaudio.load(wav_path)
        assert sr == self.sample_rate
        wav = wav[0]

        if self.speed_pertub and self.lm:
            speeds = [1.0, 0.9, 1.1]
            speed_idx = random.randint(0, 2)
            if speed_idx > 0:
                wav, _ = torchaudio.sox_effects.apply_effects_tensor(
                    wav.unsqueeze(0), self.sample_rate, [['speed', str(speeds[speed_idx])], ['rate', str(self.sample_rate)]])
        else:
            speed_idx = 0

        wav = wav.squeeze(0)
        data_len = wav.shape[0]

        chunk_len = int(self.duration * sr)
        if data_len >= chunk_len:
            start = random.randint(0, data_len - chunk_len)
            end = start + chunk_len
            wav = wav[start:end]
        else:
            wav = F.pad(wav, (0, chunk_len - data_len))

        return wav, speed_idx

class SpkLabelEncoder(object):
    def __init__(self, data_file):
        self.lab2ind = {}
        self.ind2lab = {}
        self.starting_index = -1
        self.load_from_csv(data_file)

    def __call__(self, spk, speed_idx=0):
        spkid = self.lab2ind[spk]
        spkid = spkid + len(self.lab2ind) * speed_idx
        return spkid

    def load_from_csv(self, path):
        self.data = load_data_csv(path)
        for key in self.data:
            self.add(self.data[key]['spk'])

    def add(self, label):
        if label in self.lab2ind:
            return
        index = self._next_index()
        self.lab2ind[label] = index
        self.ind2lab[index] = label

    def _next_index(self):
        self.starting_index += 1
        return self.starting_index

    def __len__(self):
        return len(self.lab2ind)

    def save(self, path, device=None):
        with open(path, 'wb') as f:
            pickle.dump(self.lab2ind, f)

    def load(self, path, device=None):
        self.lab2ind = {}
        self.ind2lab = {}
        with open(path, 'rb') as f:
            self.lab2ind = pickle.load(f)
        for label in self.lab2ind:
            self.ind2lab[self.lab2ind[label]] = label


class SpkVeriAug(object):
    def __init__(
        self,
        aug_prob: float = 0.0,
        noise_file: str = None,
        reverb_file: str = None,
    ):
        self.aug_prob = aug_prob
        if aug_prob > 0:
            self.add_noise = NoiseReverbCorrupter(
                noise_prob=1.0,
                noise_file=noise_file,
                )
            self.add_rir = NoiseReverbCorrupter(
                reverb_prob=1.0,
                reverb_file=reverb_file,
                )
            self.add_rir_noise = NoiseReverbCorrupter(
                noise_prob=1.0,
                reverb_prob=1.0,
                noise_file=noise_file,
                reverb_file=reverb_file,
                )

            self.augmentations = [self.add_noise, self.add_rir, self.add_rir_noise]

    def __call__(self, wav):
        sample_rate = 16000
        if self.aug_prob > random.random():
            aug = random.choice(self.augmentations)
            wav = aug(wav, sample_rate)

        return wav


class FBank(object):
    def __init__(self,
        n_mels,
        sample_rate,
        mean_nor: bool = False,
    ):
        self.n_mels = n_mels
        self.sample_rate = sample_rate
        self.mean_nor = mean_nor

    def __call__(self, wav, dither=0):

        sr = 16000
        assert sr==self.sample_rate, f"The sample rate of wav is {self.sample_rate} and inconsistent with model."
        if len(wav.shape) == 1:
            wav = wav.unsqueeze(0)
        # select single channel
        if wav.shape[0] > 1:
            wav = wav[0, :]
        assert len(wav.shape) == 2 and wav.shape[0]==1

        wav = wav * (1 << 15)
        mat = Kaldi.fbank(wav,
                num_mel_bins=80,
                frame_length=25,
                frame_shift=10,
                dither=1.0,
                energy_floor=0.0,
                window_type='hamming',
                sample_frequency=16000,
                snip_edges=True)
        
        mat = apply_lfr(mat, 7, 6)
        feat = apply_cmvn(mat, cmvn_data)

        return feat
