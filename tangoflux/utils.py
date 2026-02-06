import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd

import torchaudio
import random
import itertools
import numpy as np


import numpy as np


def normalize_wav(waveform):
    waveform = waveform - torch.mean(waveform)
    waveform = waveform / (torch.max(torch.abs(waveform)) + 1e-8)
    return waveform * 0.5


def pad_wav(waveform, segment_length):
    waveform_length = len(waveform)

    if segment_length is None or waveform_length == segment_length:
        return waveform
    elif waveform_length > segment_length:
        return waveform[:segment_length]
    else:
        padded_wav = torch.zeros(segment_length - waveform_length).to(waveform.device)
        waveform = torch.cat([waveform, padded_wav])
        return waveform


def read_wav_file(filename, duration_sec):
    info = torchaudio.info(filename)
    sample_rate = info.sample_rate

    # Calculate the number of frames corresponding to the desired duration
    num_frames = int(sample_rate * duration_sec)

    waveform, sr = torchaudio.load(filename, num_frames=num_frames)  # Faster!!!

    if waveform.shape[0] == 2:  ## Stereo audio
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=44100)
        resampled_waveform = resampler(waveform)
        # print(resampled_waveform.shape)
        padded_left = pad_wav(
            resampled_waveform[0], int(44100 * duration_sec)
        )  ## We pad left and right seperately
        padded_right = pad_wav(resampled_waveform[1], int(44100 * duration_sec))

        return torch.stack([padded_left, padded_right])
    else:
        waveform = torchaudio.functional.resample(
            waveform, orig_freq=sr, new_freq=44100
        )[0]
        waveform = pad_wav(waveform, int(44100 * duration_sec)).unsqueeze(0)

        return waveform


class DPOText2AudioDataset(Dataset):
    def __init__(
        self,
        dataset,
        prefix,
        text_column,
        audio_w_column,
        audio_l_column,
        duration,
        num_examples=-1,
    ):

        inputs = list(dataset[text_column])
        self.inputs = [prefix + inp for inp in inputs]
        self.audios_w = list(dataset[audio_w_column])
        self.audios_l = list(dataset[audio_l_column])
        self.durations = list(dataset[duration])
        self.indices = list(range(len(self.inputs)))

        self.mapper = {}
        for index, audio_w, audio_l, duration, text in zip(
            self.indices, self.audios_w, self.audios_l, self.durations, inputs
        ):
            self.mapper[index] = [audio_w, audio_l, duration, text]

        if num_examples != -1:
            self.inputs, self.audios_w, self.audios_l, self.durations = (
                self.inputs[:num_examples],
                self.audios_w[:num_examples],
                self.audios_l[:num_examples],
                self.durations[:num_examples],
            )
            self.indices = self.indices[:num_examples]

    def __len__(self):
        return len(self.inputs)

    def get_num_instances(self):
        return len(self.inputs)

    def __getitem__(self, index):
        s1, s2, s3, s4, s5 = (
            self.inputs[index],
            self.audios_w[index],
            self.audios_l[index],
            self.durations[index],
            self.indices[index],
        )
        return s1, s2, s3, s4, s5

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return [dat[i].tolist() for i in dat]


class Text2AudioDataset(Dataset):
    def __init__(
        self, dataset, prefix, text_column, audio_column, duration, num_examples=-1
    ):

        inputs = list(dataset[text_column])
        self.inputs = [prefix + inp for inp in inputs]
        self.audios = list(dataset[audio_column])
        self.durations = list(dataset[duration])
        self.indices = list(range(len(self.inputs)))

        self.mapper = {}
        for index, audio, duration, text in zip(
            self.indices, self.audios, self.durations, inputs
        ):
            self.mapper[index] = [audio, text, duration]

        if num_examples != -1:
            self.inputs, self.audios, self.durations = (
                self.inputs[:num_examples],
                self.audios[:num_examples],
                self.durations[:num_examples],
            )
            self.indices = self.indices[:num_examples]

    def __len__(self):
        return len(self.inputs)

    def get_num_instances(self):
        return len(self.inputs)

    def __getitem__(self, index):
        s1, s2, s3, s4 = (
            self.inputs[index],
            self.audios[index],
            self.durations[index],
            self.indices[index],
        )
        return s1, s2, s3, s4

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return [dat[i].tolist() for i in dat]
