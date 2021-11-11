import os, pickle, logging, numpy as np
from torch.utils.data import Dataset
import torch
import random

class Preprocess_Feeder(Dataset):
    def __init__(self, phase, path, connect_joint, debug, **kwargs):
        self.conn = connect_joint
        data_path_train = 'D:/flq/Animal-NAS/Animal-NAS/src/dataset/Animal-Skeleton/data_joint_train.npy'
        label_path_train = 'D:/flq/Animal-NAS/Animal-NAS/src/dataset/Animal-Skeleton/label_train.pkl'
        data_path_val = 'D:/flq/Animal-NAS/Animal-NAS/src/dataset/Animal-Skeleton/data_joint_val.npy'
        label_path_val = 'D:/flq/Animal-NAS/Animal-NAS/src/dataset/Animal-Skeleton/label_val.pkl'

        if os.path.exists(data_path_train) and os.path.exists(label_path_train):
           self.data_train = np.load(data_path_train, mmap_mode='r')
           self.data_val = np.load(data_path_val, mmap_mode='r')
           with open(label_path_train, 'rb') as f:
               self.sample_name_train, self.label_train = pickle.load(f, encoding='latin1')
           with open(label_path_val, 'rb') as f:
               self.sample_name_val, self.label_val = pickle.load(f, encoding='latin1')

        else:
            logging.info('')
            logging.error('Error: Do NOT exist data files: {} or {}!'.format(data_path_train, data_path_val))
            logging.info('Please generate data first!')
            raise ValueError()
        if debug:
            self.data = self.data[:300]
            self.label = self.label[:300]
            self.sample_name = self.sample_name[:300]
        self.phase = phase

    def __len__(self):
        # return len(self.label)
        if self.phase == 'train':
            return len(self.label_train)
        else:
            return len(self.label_val)

    def __getitem__(self, idx):
        if self.phase == 'train':
            data = np.array(self.data_train[idx])
            label = self.label_train[idx]
            name = self.sample_name_train[idx]
        else:
            data = np.array(self.data_val[idx])
            label = self.label_val[idx]
            name = self.sample_name_val[idx]

    # (C, T, V, M) -> (I, C, T, V, M)
        data = self.multi_input(data)

        return data, label, name


    def multi_input(self, data):
        C, T, V, M = data.shape
        data_new = np.zeros((3, C, T, V, M))
        data_new[0, :, :, :, :] = data
        for i in range(len(self.conn)):
            data_new[1, :, :, i, :] = data[:, :, i, :] - data[:, :, self.conn[i], :]
        for i in range(T - 1):
            data_new[2, :, i, :, :] = data[:, i + 1, :, :] - data[:, i, :, :]
        data_new[2, :, T - 1, :, :] = 0
        return data_new


    def get_k_fold_data(self, k, i, data, label, name):
        assert k > 1
        fold_size = data.shape[0] // k

        index = [i for i in range(data.shape[0])]
        random.shuffle(index)
        data = data[index[:], :]
        label = np.array(label)
        label = label[index]
        label = list(label)
        name = np.array(name)
        name = name[index]
        name = list(name)

        data_train, label_train, name_train = None, None, None
        for j in range(k):
            idx = slice(j * fold_size, (j + 1) * fold_size)
            data_part, label_part, name_part = data[idx, :], label[idx], name[idx]

            if j == i:
                data_test, label_test, name_test = data_part, label_part, name_part
            elif data_train is None:
                data_train, label_train, name_train = data_part, label_part, name_part
            else:
                data_train = np.vstack([data_train, data_part])
                label_train = label_train + label_part
                name_train = name_train + name_part

        return data_train, label_train, name_train, data_test, label_test, name_test