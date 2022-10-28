import argparse
import os
import random

import torch
import yaml
from torch import optim
from torch.backends import cudnn
from torch.utils.data import DataLoader

from dataset import WaveSpecIDDataset
from dataset_v2 import Wav_Mel_ID_Dataset
from train_utils import generate_result, train, preprocess, eval, eval_v2, test
from wavenet import SpecWaveNet, WavegramWaveNet


if torch.cuda.is_available():
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda:0' if use_cuda else 'cpu')
    # cudnn.deterministic = True
    # cudnn.benchmark = True


config_path = './config.yaml'
with open(config_path) as f:
    param = yaml.safe_load(f)

parser = argparse.ArgumentParser(description='ASD')
parser.add_argument('--data-dir', default=param['data_dir'], type=str, help='data dir')
parser.add_argument('--machine_type', type=str, default='all')
parser.add_argument('--machine_id', type=str, default='all')
parser.add_argument('--result-dir', default=param['result_dir'], type=str, help='result saved dir')
parser.add_argument('--pre-data-dir', default=param['pre_data_dir'], type=str,
                    help='processing data saved dir')
parser.add_argument('--process-machines', default=param['process_machines'], type=list,
                    help='allowed processing machines')
parser.add_argument('--ID-factor', default=param['ID_factor'], help='times for different machine types and ids to label')

# extract log-mel spectrogram
parser.add_argument('--sr', default=param['sr'], type=int, help='sample rate of wav files')
parser.add_argument('--n-fft', default=param['n_fft'], type=int, help='STFT param: n_fft')
parser.add_argument('--n-mels', default=param['n_mels'], type=int, help='STFT param: n_mels')
parser.add_argument('--hop-length', default=param['hop_length'], type=int, help='STFT param: hop_length')
parser.add_argument('--win-length', default=param['win_length'], type=int, help='STFT param: win_length')
parser.add_argument('--power', default=param['power'], type=float, help='STFT param: power')

parser.add_argument('--num_classes', type=int, default=41)
parser.add_argument('--load_pretrained', type=bool, default=False)
parser.add_argument('--pretrained_ckpt_path', type=str, default='./ckpt/all_WaveNet_SW.pth')
parser.add_argument('--model_dir', type=str, default=param['model_dir'])
parser.add_argument('--epochs', type=int, default=200, help='maximum training epochs')
parser.add_argument('--early_stop', type=int, default=50, help='early stop epochs')

parser.add_argument('--batch_size', type=int, default=param['batch_size'])
parser.add_argument('--lr', type=float, default=param['lr'], help='learning rate of others in SGD')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum of SGD')
parser.add_argument('--seed', type=int, default=666, help='manual seed')

parser.add_argument('--version', default='WaveNet_SW(10.17)', type=str, help='trail version')
args = parser.parse_args()


def run(args):
    if args.seed is None:
        args.seed = random.randint(1, 10000)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed_all(args.seed)

    # load dataset
    print(f'Loading Dataset---type:{args.machine_type}   id:{args.machine_id}')
    root_folder = os.path.join(args.pre_data_dir, f'dataset_train_path_list.db')
    train_clf_dataset = WaveSpecIDDataset(root_folder=root_folder,
                                          sr=args.sr,
                                          ID_factor=param['ID_factor'],
                                          win_length=args.n_mels,
                                          hop_length=args.hop_length,
                                          n_fft=args.n_fft,
                                          n_mels=args.n_mels,
                                          power=args.power,
                                          )
    train_loader = torch.utils.data.DataLoader(train_clf_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=16, pin_memory=True, drop_last=True)

    model = SpecWaveNet()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # optimizer = optim.Adam(model.parameters(), lr=args.lr * 0.1)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0,
                                                     last_epoch=-1)
    model.to(device)
    best_auc = 0.0
    best_pauc = 0.0
    early_stop_counter = 0

    if args.load_pretrained:
        ckpt_pth = args.pretrained_ckpt_path
        model.load_state_dict(torch.load(ckpt_pth), strict=True)
        print('load pretrained model params!')
        # auc, p_auc = eval(model, test_loader)
        generate_result(args, model)

    for epoch in range(args.epochs):
        if early_stop_counter > args.early_stop:
            break

        train(model, epoch, train_loader, optimizer, scheduler)
        early_stop_counter += 1
        if epoch % 2 == 0:
            # auc, p_auc = eval(model, test_loader)
            auc, p_auc = eval_v2(model, args)
            if auc + p_auc > (best_auc + best_pauc):
                early_stop_counter = 0
                best_auc = auc
                best_pauc = p_auc
                torch.save(model.state_dict(), f'./{args.model_dir}/{args.machine_type}_{args.version}.pth')
            print(f'Sample Auroc:{auc:.3f} p_auc:{p_auc:.3f}   Best Auc:{best_auc:.4f}')
            # os.makedirs(ckp_folder, exist_ok=True)
            # torch.save(decoder.state_dict(), ckp_path)
    print('best_auc:', best_auc, '  p_auc:', best_pauc)

    return best_auc, best_pauc


if __name__ == '__main__':
    preprocess(args)
    auc, pauc = run(args)
