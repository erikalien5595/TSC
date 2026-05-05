import os
import numpy as np
import pandas as pd
import os
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
import warnings

warnings.filterwarnings('ignore')


class Dataset_ETT_hour(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', cycle=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.cycle = cycle

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        train_data_std = pd.DataFrame(data[border1s[0]:border2s[0]], columns=df_data.columns)
        n = len(train_data_std)

        if self.cycle is None or int(self.cycle) <= 0:
            raise ValueError("`cycle` must be a positive integer to compute cycle_data")

        pos = (np.arange(n)+self.seq_len) % int(self.cycle)
        out = train_data_std.groupby(pos).mean()

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp['date'])
        df_stamp['hour'] = df_stamp['date'].dt.hour
        df_stamp['day'] = df_stamp['date'].dt.weekday
        self.hour_index = df_stamp['hour'].values
        self.day_index = df_stamp['day'].values

        # add cycle
        self.cycle_index = (np.arange(len(data)) % self.cycle)[border1:border2]
        self.cycle_data = out.values

        
        os.makedirs('./analyze_results/', exist_ok=True)
        try:
            basename = os.path.splitext(self.data_path)[0] if hasattr(self, 'data_path') else 'data'
            save_name = f'{basename}_TSC_fix_Q.npy'
            save_path = os.path.join('./analyze_results/', save_name)
            np.save(save_path, self.cycle_data)
        except Exception as e:
            print(f'Warning: failed to save cycle_data to .npy: {e}')

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        hour_index = torch.tensor(self.hour_index[s_end])
        day_index = torch.tensor(self.day_index[s_end])

        cycle_index = torch.tensor(self.cycle_index[s_end])

        return seq_x, seq_y, seq_x_mark, seq_y_mark, cycle_index, hour_index, day_index, -1, -1

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_ETT_minute(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTm1.csv',
                 target='OT', scale=True, timeenc=0, freq='t', cycle=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.cycle = cycle

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        train_data_std = pd.DataFrame(data[border1s[0]:border2s[0]], columns=df_data.columns)
        n = len(train_data_std)

        if self.cycle is None or int(self.cycle) <= 0:
            raise ValueError("`cycle` must be a positive integer to compute cycle_data")

        pos = (np.arange(n)+self.seq_len) % int(self.cycle)
        out = train_data_std.groupby(pos).mean()

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp['date'])
        df_stamp['hour'] = df_stamp['date'].dt.hour
        df_stamp['day'] = df_stamp['date'].dt.weekday
        self.hour_index = df_stamp['hour'].values
        self.day_index = df_stamp['day'].values

        # add cycle
        self.cycle_index = (np.arange(len(data)) % self.cycle)[border1:border2]
        self.cycle_data = out.values

        os.makedirs('./analyze_results/', exist_ok=True)
        
        try:
            basename = os.path.splitext(self.data_path)[0] if hasattr(self, 'data_path') else 'data'
            save_name = f'{basename}_TSC_fix_Q.npy'
            save_path = os.path.join('./analyze_results/', save_name)
            np.save(save_path, self.cycle_data)
        except Exception as e:
            print(f'Warning: failed to save cycle_data to .npy: {e}')

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        hour_index = torch.tensor(self.hour_index[s_end])
        day_index = torch.tensor(self.day_index[s_end])

        cycle_index = torch.tensor(self.cycle_index[s_end])

        return seq_x, seq_y, seq_x_mark, seq_y_mark, cycle_index, hour_index, day_index, -1, -1

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', cycle=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.cycle = cycle

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        if 'traffic' in self.data_path:
            self.scalerForEvaluation = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            if 'traffic' in self.data_path:
                train_data = df_data[border1s[0]:border2s[0]].copy()
                self.scalerForEvaluation.fit(train_data)  # 用于Evaluation
                df_data = np.log1p(df_data)
            train_data2 = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data2)  
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        
        train_data_std = pd.DataFrame(data[border1s[0]:border2s[0]], columns=df_data.columns)
        n = len(train_data_std)

        if self.cycle is None or int(self.cycle) <= 0:
            raise ValueError("`cycle` must be a positive integer to compute cycle_data")

        pos = (np.arange(n)+self.seq_len) % int(self.cycle)
        out = train_data_std.groupby(pos).mean()

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp['date'])
        df_stamp['hour'] = df_stamp['date'].dt.hour
        df_stamp['day'] = df_stamp['date'].dt.weekday
        df_stamp['month'] = df_stamp['date'].dt.month
        df_stamp['day_in_month'] = df_stamp['date'].dt.day
        self.hour_index = df_stamp['hour'].values
        self.day_index = df_stamp['day'].values
        self.month_index = df_stamp['month'].values - 1  # 0-11
        self.day_in_month_index = df_stamp['day_in_month'].values - 1  # 0-30

        # add cycle
        self.cycle_index = (np.arange(len(data)) % self.cycle)[border1:border2]
        self.cycle_data = out.values

        os.makedirs('./analyze_results/', exist_ok=True)
        
        try:
            basename = os.path.splitext(self.data_path)[0] if hasattr(self, 'data_path') else 'data'
            print(123)
            save_name = f'{basename}_TSC_fix_Q.npy'
            save_path = os.path.join('./analyze_results/', save_name)
            np.save(save_path, self.cycle_data)
        except Exception as e:
            print(f'Warning: failed to save cycle_data to .npy: {e}')

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        hour_index = torch.tensor(self.hour_index[s_end])
        day_index = torch.tensor(self.day_index[s_end])
        month_index = torch.tensor(self.month_index[s_end])
        day_in_month_index = torch.tensor(self.day_in_month_index[s_end])

        cycle_index = torch.tensor(self.cycle_index[s_end])
        return seq_x, seq_y, seq_x_mark, seq_y_mark, cycle_index, hour_index, day_index, month_index, day_in_month_index

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Solar(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', cycle=None):
        # size [seq_len, label_len, pred_len]
        # info
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.cycle = cycle

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        self.scalerForEvaluation = StandardScaler()
        df_raw = []
        with open(os.path.join(self.root_path, self.data_path), "r", encoding='utf-8') as f:
            for line in f.readlines():
                line = line.strip('\n').split(',')
                data_line = np.stack([float(i) for i in line])
                df_raw.append(data_line)
        df_raw = np.stack(df_raw, 0)
        df_raw = pd.DataFrame(df_raw)

        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_valid = int(len(df_raw) * 0.1)
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_valid, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        df_data = df_raw.values

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scalerForEvaluation.fit(train_data)  # 用于Evaluation
            # data = self.scaler.transform(df_data)
            df_data = np.log1p(df_data)
            train_data2 = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data2)  # 用于训练
            data = self.scaler.transform(df_data)
        else:
            data = df_data

        train_data_std = pd.DataFrame(data[border1s[0]:border2s[0]])
        n = len(train_data_std)

        if self.cycle is None or int(self.cycle) <= 0:
            raise ValueError("`cycle` must be a positive integer to compute cycle_data")

        pos = (np.arange(n)+self.seq_len) % int(self.cycle)
        out = train_data_std.groupby(pos).mean()

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

        # add cycle
        self.cycle_index = (np.arange(len(data)) % self.cycle)[border1:border2]
        self.cycle_data = out.values

        os.makedirs('./analyze_results/', exist_ok=True)
        
        try:
            basename = os.path.splitext(self.data_path)[0] if hasattr(self, 'data_path') else 'data'
            print(123)
            save_name = f'{basename}_TSC_fix_Q.npy'
            save_path = os.path.join('./analyze_results/', save_name)
            np.save(save_path, self.cycle_data)
        except Exception as e:
            print(f'Warning: failed to save cycle_data to .npy: {e}')


    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = torch.zeros((seq_x.shape[0], 1))
        seq_y_mark = torch.zeros((seq_x.shape[0], 1))

        cycle_index = torch.tensor(self.cycle_index[s_end])

        return seq_x, seq_y, seq_x_mark, seq_y_mark, cycle_index, -1, -1, -1, -1

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_PEMS(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', cycle=None):
        # size [seq_len, label_len, pred_len]
        # info
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.cycle = cycle

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        data_file = os.path.join(self.root_path, self.data_path)
        data = np.load(data_file, allow_pickle=True)
        data = data['data'][:, :, 0]

        num_train = int(len(data) * 0.6)
        num_test = int(len(data) * 0.2)
        num_valid = int(len(data) * 0.2)
        border1s = [0, num_train - self.seq_len, len(data) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_valid, len(data)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.scale:
            train_data = data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data)
            data = self.scaler.transform(data)

        train_data_std = pd.DataFrame(data[border1s[0]:border2s[0]])
        n = len(train_data_std)

        
        pos = (np.arange(n)+self.seq_len) % int(self.cycle)
        out = train_data_std.groupby(pos).mean()

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

        # add cycle
        self.cycle_index = (np.arange(len(data)) % self.cycle)[border1:border2]
        self.cycle_data = out.values

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = torch.zeros((seq_x.shape[0], 1))
        seq_y_mark = torch.zeros((seq_x.shape[0], 1))

        cycle_index = torch.tensor(self.cycle_index[s_end])
        return seq_x, seq_y, seq_x_mark, seq_y_mark, cycle_index, -1 , -1, -1, -1

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class DataSim(Dataset):
    """Simulated multivariate time series dataset.

    Requirements (per user):
      - sampling frequency: 1 step = 1 day (conceptually)
      - total steps: 365*30
      - signal: 7-day + 30-day periodic components + Gaussian white noise
      - channels: 2, independent

    Compatibility:
      - Same split logic as Dataset_Custom (70%/val/test with seq_len offset)
      - Same __getitem__ signature as Dataset_Custom
    """

    def __init__(
        self,
        root_path=None,
        flag='train',
        size=None,
        features='M',
        data_path=None,
        target=None,
        scale=True,
        timeenc=0,
        freq='d',
        cycle=None,
        total_steps=365 * 30,
        num_channels=1,
        noise_std=1,
        seed=42,
    ):
        if size is None:
            self.seq_len = 96
            self.label_len = 48
            self.pred_len = 96
        else:
            self.seq_len, self.label_len, self.pred_len = size

        assert flag in ['train', 'val', 'test']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.cycle = cycle

        self.total_steps = int(total_steps)
        self.num_channels = int(num_channels)
        self.noise_std = float(noise_std)
        self.seed = int(seed)

        # kept for interface compatibility
        self.root_path = root_path
        self.data_path = data_path

        self.__read_data__()

    def _make_sim_df(self) -> pd.DataFrame:
        T = self.total_steps
        t = np.arange(T, dtype=np.float64)

        p7 = 7
        p30 = 30


        data = np.zeros((T, self.num_channels), dtype=np.float64)
        for c in range(self.num_channels):
            np.random.seed(self.seed+c)
            a1 = np.random.uniform(2, 3)
            a2 = np.random.uniform(4, 5)
            ph1 = np.random.uniform(0, np.pi/2)
            ph2 = np.random.uniform(np.pi/2, 1 * np.pi)

            sig = a1 * np.sin(2 * np.pi * t / p7 + ph1) + a2 * np.sin(2 * np.pi * t / p30 + ph2)
            noise = np.random.normal(0.0, self.noise_std, size=T)
            data[:, c] = sig + noise

        start = pd.Timestamp('2000-01-01')
        dates = pd.date_range(start=start, periods=T, freq='D')
        df_raw = pd.DataFrame({'date': dates})
        for c in range(self.num_channels):
            df_raw[f'c{c}'] = data[:, c]
        return df_raw

    def __read_data__(self):
        self.scaler = StandardScaler()

        df_raw = self._make_sim_df()

        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        cols_data = [c for c in df_raw.columns if c != 'date']
        df_data = df_raw[cols_data]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        train_data_std = pd.DataFrame(data[border1s[0]:border2s[0]], columns=cols_data)
        n = len(train_data_std)

        if self.cycle is None or int(self.cycle) <= 0:
            raise ValueError("`cycle` must be a positive integer to compute cycle_data")

        pos = (np.arange(n) + self.seq_len) % int(self.cycle)
        out = train_data_std.groupby(pos).mean()

        df_stamp = df_raw[['date']][border1:border2].copy()
        df_stamp['date'] = pd.to_datetime(df_stamp['date'])
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], axis=1).values
        else:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

        df_stamp2 = df_raw[['date']][border1:border2].copy()
        df_stamp2['date'] = pd.to_datetime(df_stamp2['date'])
        df_stamp2['hour'] = df_stamp2['date'].dt.hour
        df_stamp2['day'] = df_stamp2['date'].dt.weekday
        df_stamp2['month'] = df_stamp2['date'].dt.month
        df_stamp2['day_in_month'] = df_stamp2['date'].dt.day

        self.hour_index = df_stamp2['hour'].values
        self.day_index = df_stamp2['day'].values
        self.month_index = df_stamp2['month'].values - 1
        self.day_in_month_index = df_stamp2['day_in_month'].values - 1

        self.cycle_index = (np.arange(len(data)) % int(self.cycle))[border1:border2]
        self.cycle_data = out.values

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        hour_index = torch.tensor(self.hour_index[s_end])
        day_index = torch.tensor(self.day_index[s_end])
        month_index = torch.tensor(self.month_index[s_end])
        day_in_month_index = torch.tensor(self.day_in_month_index[s_end])

        cycle_index = torch.tensor(self.cycle_index[s_end])
        return seq_x, seq_y, seq_x_mark, seq_y_mark, cycle_index, hour_index, day_index, month_index, day_in_month_index

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
