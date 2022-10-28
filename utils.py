import matplotlib.pyplot as plt

import os
import shutil
import sys
import time
import yaml
import csv
import glob
import itertools
import re
import numpy as np
import librosa
import torch
import joblib
from itertools import chain
from dataset_v2 import Generator


def save_csv(save_file_path,
             save_data):
    with open(save_file_path, 'w', newline='') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerows(save_data)


# file process
def select_dirs(data_dir, data_type='dev_data'):
    base_path = os.path.join(data_dir, data_type)
    dir_path = os.path.abspath(f'{base_path}/*')
    dirs = glob.glob(dir_path)
    return dirs


def replay_visdom(writer, log_path):
    file_path = os.path.abspath(f'{log_path}/*')
    files = glob.glob(file_path)
    for file in files:
        writer.replay_log(file)


def create_file_list(target_dir,
                     dir_name='train',
                     ext='wav'):
    list_path = os.path.abspath(f'{target_dir}/{dir_name}/*.{ext}')
    files = sorted(glob.glob(list_path))
    return files


def create_wav_list(target_dir,
                    id_name,
                    dir_name='test',
                    prefix_normal='normal',
                    prefix_anomaly='anomaly',
                    ext='wav'):
    normal_files_path = f'{target_dir}/{dir_name}/{prefix_normal}_{id_name}*.{ext}'
    normal_files = sorted(glob.glob(normal_files_path))

    if dir_name == 'test':
        anomaly_files_path = f'{target_dir}/{dir_name}/{prefix_anomaly}_{id_name}*.{ext}'
        anomaly_files = sorted(glob.glob(anomaly_files_path))
        return normal_files, anomaly_files
    return normal_files


# get test machine id list
def get_machine_id_list(target_dir,
                        dir_name='test',
                        ext='wav'):
    dir_path = os.path.abspath(f'{target_dir}/{dir_name}/*.{ext}')
    files_path = sorted(glob.glob(dir_path))  # 得到了某个train目录下的所有wav文件列表
    #  set()函数创建一个无序不重复元素集，删除重复数据，set([iterable])
    machine_id_list = sorted(list(set(
        itertools.chain.from_iterable([re.findall('id_[0-9][0-9]', ext_id) for ext_id in files_path])
    )))
    return machine_id_list

def create_test_file_list(target_dir,
                          id_name,
                          dir_name='test',
                          prefix_normal='normal',
                          prefix_anomaly='anomaly',
                          ext='wav'):
    normal_files_path = f'{target_dir}/{dir_name}/{prefix_normal}_{id_name}*.{ext}'
    normal_files = sorted(glob.glob(normal_files_path))
    normal_labels = np.zeros(len(normal_files))  # 正常为 0

    anomaly_files_path = f'{target_dir}/{dir_name}/{prefix_anomaly}_{id_name}*.{ext}'
    anomaly_files = sorted(glob.glob(anomaly_files_path))
    anomaly_labels = np.ones(len(anomaly_files))  # 异常为 1

    files = np.concatenate((normal_files, anomaly_files), axis=0)  # 所有异常音频文件和正常音频文件列表
    labels = np.concatenate((normal_labels, anomaly_labels), axis=0)  # 所有异常音频文件和正常音频标签列表
    return files, labels


def path_to_dict(process_machines,
                 data_dir,  # 原始数据集目录
                 root_folder,
                 ID_factor,
                 dir_name='train',
                 data_type='', ):
    dirs = select_dirs(data_dir, data_type=data_type)
    path_list = []
    for index, target_dir in enumerate(sorted(dirs)):
        print('\n' + '=' * 20)
        print(f'[{index + 1}/{len(dirs)}] {target_dir}')
        time.sleep(1)
        machine_type = os.path.split(target_dir)[1]
        if machine_type not in process_machines:
            continue

        machine_id_list = get_machine_id_list(target_dir, dir_name=dir_name)
        for id_str in machine_id_list:
            files, _ = create_test_file_list(target_dir, id_str, dir_name=dir_name)
            path_list.append(files)

            print(f'{data_type} {machine_type} {id_str} were split to {len(files)} wav files!')
    #  chain.from_iterable()将嵌套列表合并成一个
    path_list = list(chain.from_iterable(path_list))
    os.makedirs(os.path.split(root_folder)[0], exist_ok=True)
    with open(root_folder, 'wb') as f:
        joblib.dump(path_list, f)

ID_factor = {
       'fan': 0,
       'pump': 1,
       'slider': 2,
       'valve': 3,
       'ToyCar': 4,
       'ToyConveyor': 5,
}


def transform(file_path, machine_type, id_str):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if machine_type == 'ToyCar' or machine_type == 'ToyConveyor':
        id = int(id_str[-1]) - 1
    else:
        id = int(id_str[-1])
    label = int(ID_factor[machine_type] * 7 + id)
    # label = int(id//2)
    label = torch.from_numpy(np.array(label)).long().to(device)
    (x, _) = librosa.core.load(file_path, sr=16000, mono=True)

    x_wav = x[None, None, :16000 * 10]  # (1, audio_length)
    x_wav = torch.from_numpy(x_wav)
    x_wav = x_wav.float().to(device)

    x_mel = x[:16000 * 10]  # (1, audio_length)
    x_mel = torch.from_numpy(x_mel)
    x_mel = Generator(16000)(x_mel)
    # x_mel = torch.cat([x_mel, x_mel[:, :, :320 - x_mel.shape[2]]], dim=2)  # [1, 128, 320]
    x_mel = x_mel.unsqueeze(0).to(device)
    return x_wav, x_mel, label


