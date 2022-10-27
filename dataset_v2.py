import glob
import os.path

import numpy as np
import torch
from torch.utils.data import Dataset
import torchaudio
import joblib
import librosa
import re

ID_factor = {
       'fan': 0,
       'pump': 1,
       'slider': 2,
       'valve': 3,
       'ToyCar': 4,
       'ToyConveyor': 5,
}

#tsts
class Generator(object):
    def __init__(self, sr,
                 n_fft=1024,
                 n_mels=128,
                 win_length=1024,
                 hop_length=512,
                 power=2.0
                 ):
        self.mel_transform = torchaudio.transforms.MelSpectrogram(sample_rate=sr,
                                                                  win_length=win_length,
                                                                  hop_length=hop_length,
                                                                  n_fft=n_fft,
                                                                  n_mels=n_mels,
                                                                  power=power)
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB(stype='power')

    def __call__(self, x):
        # spec =  self.amplitude_to_db(self.mel_transform(x)).squeeze().transpose(-1,-2)
        # 返回log_melsepc
        return self.amplitude_to_db(self.mel_transform(x))


class Wav_Mel_ID_Dataset(Dataset):
    def __init__(self, root_floder='./data',
                 machine_type='slider',
                 machine_id='id_00',
                 sr=16000,
                 win_length=1024,
                 hop_length=512,
                 phase='train'):
        # with open(root_floder, 'rb') as f:
        #     self.file_path_list = (f)
        self.phase = phase
        if machine_type == 'all':
            if machine_id == 'all':
                self.wav_path_list = glob.glob(root_floder + f"/*/{self.phase}/*.wav")
            else:
                self.wav_path_list = glob.glob(root_floder + f"/*/{self.phase}/*{machine_id}*.wav")
        else:
            self.wav_path = os.path.join(root_floder, machine_type, self.phase)
            if machine_id == 'all':
                self.wav_path_list = glob.glob(self.wav_path + f"/*.wav")
            else:
                self.wav_path_list = glob.glob(self.wav_path + f"/*{machine_id}*.wav")
        # self.wav_path_list = self.wav_path_list[:len(self.wav_path_list)//2]

        self.sr = sr
        self.win_len = win_length
        self.hop_len = hop_length
        # print(len(self.file_path_list))

        self.transform = Generator(sr=self.sr)

    def __getitem__(self, item):
        file_path = self.wav_path_list[item]
        machine = file_path.split('/')[-3]  # 机器类型
        if self.phase == 'train':
            label = 0
        else:
            # label = 0 if file_path.split('/')[-2] == 'normal' else 1
            label = 0 if file_path.split('/')[-1].split('_')[0] == 'normal' else 1

        id_str = re.findall('id_[0-9][0-9]', file_path)  # 机器编号
        if machine == 'ToyCar' or machine == 'ToyConveyor':
            id = int(id_str[0][-1]) - 1
        else:
            id = int(id_str[0][-1])

        label_id = int(ID_factor[machine] * 7 + id)  # 除了 Toy类型机器，每个机器类型有 7个编号

        (x, _) = librosa.core.load(file_path, sr=self.sr, mono=True)  # 加载音频

        x = x[:self.sr * 10]  # (1, audio_length) # 截取前10s
        x_wav = torch.from_numpy(x)  # 将 wav numpy转成 wav tensor

        x_mel = self.transform(x_wav)  # 从 wav 转成...
        # x_mel = torch.cat([x_mel, x_mel[:,:,:320-x_mel.shape[2]]], dim=2)  # [1, 128, 320]

        return x_mel, x_wav.unsqueeze(0), label

    def __len__(self):
        return len(self.wav_path_list)

if __name__ == '__main__':
    pass
