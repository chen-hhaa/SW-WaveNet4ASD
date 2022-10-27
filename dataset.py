import random

import torch
from torch.utils.data import Dataset
import torchaudio
import joblib
import librosa
import re

#tsts
class transform(object):
    def __init__(self):
        self.TimeMasking = torchaudio.transforms.TimeMasking(time_mask_param=30)
        self.FrequencyMasking = torchaudio.transforms.FrequencyMasking(freq_mask_param=20)


    def __call__(self, x):
        if random.random() < 0.2:
            # if random.random() < 0.5:
            #     x = self.TimeMasking(x)
            # else:
            x = self.FrequencyMasking(x)
        return x


class WaveSpecIDDataset(Dataset):
    def __init__(self, root_folder, ID_factor, sr,
                 win_length=1024,
                 hop_length=512,
                 n_fft=1024,
                 n_mels=128,
                 power=2.0,
                 transform=transform):
        with open(root_folder, 'rb') as f:
            self.file_path_list = joblib.load(f)

        self.factor = ID_factor
        self.sr = sr
        self.win_len = win_length
        self.hop_len = hop_length
        self.mel_transform = torchaudio.transforms.MelSpectrogram(sample_rate=sr,
                                                                  win_length=win_length,
                                                                  hop_length=hop_length,
                                                                  n_fft=n_fft,
                                                                  n_mels=n_mels,
                                                                  power=power)
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB(stype='power')
        self.transform = transform
        # print(len(self.file_path_list))

    def __getitem__(self, item):
        file_path = self.file_path_list[item]
        machine = file_path.split('/')[-3]  # 机器类型
        id_str = re.findall('id_[0-9][0-9]', file_path)  # 机器编号
        if machine == 'ToyCar' or machine == 'ToyConveyor':
            id = int(id_str[0][-1]) - 1
        else:
            id = int(id_str[0][-1])
        label = int(self.factor[machine] * 7 + id)  # 除了 Toy类型机器，每个机器类型有 7个编号，乘7是为了label不重复

        # label = int(id//2)
        (x, _) = librosa.core.load(file_path, sr=self.sr, mono=True)  # 加载音频

        x = x[:self.sr * 10]  # (1, audio_length) # 截取前10s
        x_wav = torch.from_numpy(x)  # 将 wav numpy转成 wav tensor
        x_mel = self.amplitude_to_db(self.mel_transform(x_wav))  # 获取log-mel spectrogram, 从 wav 转成...  # [128, 313]
        # x_mel = self.transform(x_wav)
        return x_mel, x_wav.unsqueeze(0), label

    def __len__(self):
        return len(self.file_path_list)


if __name__ == '__main__':
    pass
