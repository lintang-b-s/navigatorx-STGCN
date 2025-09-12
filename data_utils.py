import numpy as np
import os 
import pandas as pd 
import math 
import data_utils
from torch.utils import data
import torch
from sklearn.preprocessing import StandardScaler


def load_data(dataset_name, dataset_file_name, len_train, len_val):
    vel = pd.read_csv(os.path.join(dataset_name, dataset_file_name), header=None)

    train = vel[:len_train]
    val = vel[len_train: len_train + len_val]
    test = vel[len_train + len_val:]
    return train, val, test


def data_transform(data, n_his, n_pred, device):
    
    n_vertex = data.shape[1]
    len_record = len(data)
    n_frame = n_his + n_pred
    seq_num = len_record - n_frame + 1
    
    x = np.zeros([seq_num, n_his, n_vertex, 1]) # bs, ts, n_vertex, c_in
    y = np.zeros([seq_num, n_pred, n_vertex])
    
    for i in range(seq_num):
        head = i
        tail_x = i + n_his
        tail_y = tail_x + n_pred
        x[i, :, :, :] = data[head: tail_x].reshape(n_his, n_vertex, 1) # seq_num, n_his, n_vertex, c_in
        y[i] = data[tail_x:tail_y] #  seq_num, n_pred, n_vertex 
    return torch.from_numpy(x).float().to(device), torch.from_numpy(y).float().to(device)

def generate_data(dataset_name,dataset_file_name, n_his, n_pred, batch_size, device):
    data_seq = pd.read_csv(os.path.join(dataset_name, dataset_file_name), header=None).shape[0]
    val_and_test_rate = 0.2

    len_val = int(math.floor(data_seq * val_and_test_rate))
    len_test = int(math.floor(data_seq * val_and_test_rate))
    len_train = int(data_seq - len_val - len_test)
    train, val, test =  data_utils.load_data(dataset_name, dataset_file_name, len_train, len_val)
    
    scaler = StandardScaler()
    train = scaler.fit_transform(train)
    val = scaler.transform(val)
    test = scaler.transform(test)
    

    train_data = TrafficDataset(seed=42, data=train, n_his=n_his, n_pred=n_pred, device=device)
    val_data = TrafficDataset(seed=42, data=val, n_his=n_his, n_pred=n_pred, device=device)
    test_data = TrafficDataset(seed=42, data=test, n_his=n_his, n_pred=n_pred, device=device)
    train_data_loader = data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_data_loader = data.DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_data_loader = data.DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return train_data_loader, val_data_loader, test_data_loader, scaler


def mape(y, y_pred):
    return np.mean(np.abs((y - y_pred) / (y+1e-5)))

def rmse(y, y_pred):    
    return np.sqrt(np.mean((y - y_pred)**2))

def mse(y, y_pred):
    return np.mean((y - y_pred)**2)

def mae(y, y_pred):
    return np.mean(np.abs(y - y_pred))


class TrafficDataset(data.Dataset):
    def __init__(self, seed, data, n_his, n_pred, device):
        super().__init__()
        self.np_rng = np.random.RandomState(seed=seed)
        self.generate_data(data, n_his, n_pred, device)

    def generate_data(self, data, n_his, n_pred, device):
        x, y = data_utils.data_transform(data, n_his, n_pred, device)
        self.data = x
        self.label = y   

    def __len__(self):
        # Number of data point we have
        return  self.data.shape[0]

    def __getitem__(self, idx):
        # Return the idx-th data point of the dataset
        data_point = self.data[idx]
        data_label = self.label[idx]
        return data_point, data_label
