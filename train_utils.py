import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

from tqdm import tqdm
from dataset_v2 import Wav_Mel_ID_Dataset
from wavenet import WaveNet
from utils import show_result, get_machine_id_list, create_test_file_list, transform
import utils
import time
import yaml
import os

from torch.backends import cudnn

if torch.cuda.is_available():
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda:0' if use_cuda else 'cpu')
    # cudnn.deterministic = True
    # cudnn.benchmark = True


def train(model, epoch, train_loader, optimizer, scheduler):
    model.train()
    pbar = tqdm(train_loader, total=len(train_loader), ncols=100)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    loss_list = []
    for mel, wav, labels in pbar:
        wav = wav.float().to(device)
        input = mel.float().to(device)
        labels = labels.long().squeeze().to(device)
        predict_ids, _ = model(input, wav, labels)
        loss = criterion(predict_ids, labels)
        loss_list.append(loss.item())
        pbar.set_description(f'Epoch:{epoch} Training'
                             f'\tLclf:{loss.item():.5f}\t')

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if scheduler is not None and epoch >= 20:
            scheduler.step()
    print(f"Epoch: {epoch}\tLoss: {np.mean(loss_list)}")


def eval(model, test_loader):
    model.eval()
    pbar = tqdm(test_loader, total=len(test_loader), ncols=100)
    y_pred = []
    y_true = []
    for mel, labels, labels_id in pbar:
        with torch.no_grad():
            input = mel.to(device)
            labels_id = labels_id.long().squeeze().to(device)
            predict_ids, feature = model(input, labels_id)
        probs = - torch.log_softmax(predict_ids, dim=1).squeeze().cpu().numpy()
        for i, id in enumerate(labels_id.cpu().numpy()):
            y_pred.append(probs[i][id])
            y_true.append(labels[i].item())
        # y_pred.append(probs[labels_id])
        # y_true.append(labels.item())
        pbar.set_description('Validating')

    AUC = round(roc_auc_score(y_true, y_pred), 4)
    max_fpr = 0.1
    pAUC = round(roc_auc_score(y_true, y_pred, max_fpr=max_fpr), 4)

    return AUC, pAUC


def eval_v2(model, args):
    classifier = model
    classifier.eval()
    sum_auc, sum_pauc, num, total_time = 0, 0, 0, 0
    #
    # sum_auc_r, sum_pauc_r = 0, 0

    dirs = utils.select_dirs(args.data_path, data_type='')
    print('\n' + '=' * 20)
    for index, target_dir in enumerate(sorted(dirs)):
        start = time.perf_counter()
        machine_type = os.path.split(target_dir)[1]
        if machine_type not in args.process_machines:
            continue
        num += 1
        # get machine list
        machine_id_list = get_machine_id_list(target_dir, dir_name='test')
        performance = []
        performance_recon = []
        for id_str in machine_id_list:
            test_files, y_true = create_test_file_list(target_dir, id_str, dir_name='test')
            y_pred = [0. for _ in test_files]

            for file_idx, file_path in enumerate(test_files):
                x_wav, x_mel, label = transform(file_path, machine_type, id_str)
                with torch.no_grad():
                    net = classifier
                    predict_ids, feature = net(x_mel, x_wav, label)
                probs = - torch.log_softmax(predict_ids, dim=1).mean(dim=0).squeeze().cpu().numpy()
                y_pred[file_idx] = probs[label]

            # compute auc and pAuc
            max_fpr = 0.1
            auc = roc_auc_score(y_true, y_pred)
            p_auc = roc_auc_score(y_true, y_pred, max_fpr=max_fpr)
            performance.append([auc, p_auc])

        # calculate averages for AUCs and pAUCs
        averaged_performance = np.mean(np.array(performance, dtype=float), axis=0)
        mean_auc, mean_p_auc = averaged_performance[0], averaged_performance[1]
        # print(machine_type, 'AUC_clf:', mean_auc, 'pAUC_clf:', mean_p_auc)
        sum_auc += mean_auc
        sum_pauc += mean_p_auc

        time_nedded = time.perf_counter() - start
        total_time += time_nedded
        print(f'Test {machine_type} cost {time_nedded} secs')
    print(f'Total test time: {total_time} secs!')
    return sum_auc / num, sum_pauc / num


def test(model, type_list, id_list, args):
    model.eval()
    average_auc_li = []
    average_pauc_li = []
    for type in type_list:
        print(f'Staring test {type}......')
        Auc_list = []
        pAuc_list = []
        for id in id_list:
            test_dataset = Wav_Mel_ID_Dataset(root_floder=args.data_path,
                                              machine_type=type,
                                              machine_id=id,
                                              phase='test')
            test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
            AUC, pAUC = eval(model, test_loader)
            Auc_list.append(AUC)
            pAuc_list.append(pAUC)
            print(f'{type} {id}  AUC={AUC:.4f}  pAUC={pAUC:.4f}')
        average_auc_li.append(np.mean(Auc_list))
        average_pauc_li.append(np.mean(pAuc_list))
        print(f'{type} average  AUC={np.mean(Auc_list):.4f}  pAUC={np.mean(pAuc_list):.4f}')

    # print
    for i in range(len(average_auc_li)):
        print(f'{type_list[i]}-------AUC={average_auc_li[i]}   pAUC={average_pauc_li[i]}')
    print(f'average-------AUC={np.mean(average_auc_li)}   pAUC={np.mean(average_pauc_li)}')


# 生成csv文件
# This part of the code reference from https://github.com/liuyoude/STgram-MFN
def generate_result(args, model, save=True):
    classifier = model

    recore_dict = {}
    csv_lines = []
    if not save:
        csv_lines = []

    sum_auc, sum_pauc, num = 0, 0, 0
    dirs = utils.select_dirs(args.data_path, data_type='')
    result_dir = os.path.join(args.result_dir, args.version)
    os.makedirs(result_dir, exist_ok=True)
    print('\n' + '=' * 20)
    for index, target_dir in enumerate(sorted(dirs)):
        time.sleep(1)
        machine_type = os.path.split(target_dir)[1]
        if machine_type not in args.process_machines:
            continue
        num += 1
        # result csv
        csv_lines.append([machine_type])
        csv_lines.append(['id', 'AUC', 'pAUC'])
        performance = []
        # get machine list
        machine_id_list = get_machine_id_list(target_dir, dir_name='test')
        for id_str in machine_id_list:
            test_files, y_true = create_test_file_list(target_dir, id_str, dir_name='test')
            csv_path = os.path.join(result_dir, f'{machine_type}_anomaly_score_{id_str}.csv')
            anomaly_score_list = []
            y_pred = [0. for _ in test_files]
            for file_idx, file_path in enumerate(test_files):
                x_wav, x_mel, label = transform(file_path, machine_type, id_str)
                with torch.no_grad():
                    classifier.eval()
                    net = classifier
                    predict_ids, feature = net(x_mel, x_wav, label)
                probs = - torch.log_softmax(predict_ids, dim=1).mean(dim=0).squeeze().cpu().numpy()
                y_pred[file_idx] = probs[label]
                anomaly_score_list.append([os.path.basename(file_path), y_pred[file_idx]])
            if save:
                utils.save_csv(csv_path, anomaly_score_list)
            # compute auc and pAuc
            max_fpr = 0.1
            auc = roc_auc_score(y_true, y_pred)
            p_auc = roc_auc_score(y_true, y_pred, max_fpr=max_fpr)
            #
            csv_lines.append([id_str.split('_', 1)[1], auc, p_auc])
            performance.append([auc, p_auc])

        # calculate averages for AUCs and pAUCs
        averaged_performance = np.mean(np.array(performance, dtype=float), axis=0)
        mean_auc, mean_p_auc = averaged_performance[0], averaged_performance[1]
        print(machine_type, 'AUC:', mean_auc, 'pAUC:', mean_p_auc)
        recore_dict[machine_type] = mean_auc + mean_p_auc
        sum_auc += mean_auc
        sum_pauc += mean_p_auc
        csv_lines.append(['Average'] + list(averaged_performance))
    csv_lines.append(['Total Average', sum_auc / num, sum_pauc / num])
    print('Total average:', sum_auc / num, sum_pauc / num)
    result_path = os.path.join(result_dir, 'result.csv')
    if save:
        utils.save_csv(result_path, csv_lines)
    return recore_dict

#  该函数的作用是将所有训练音频文件的路径列表存入.db文件中，可以提高文件路径检索速度
def preprocess(args):
    root_folder = os.path.join(args.pre_data_dir, f'313frames_train_path_list.db')
    if not os.path.exists(root_folder):
        utils.path_to_dict(process_machines=args.process_machines,
                           data_dir=args.data_dir,
                           root_folder=root_folder,
                           ID_factor=args.ID_factor)
