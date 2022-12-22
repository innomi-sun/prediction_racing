import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

class RacingDataset(Dataset):

    def __init__(self, csv_file_root, transform=None):

        data_original = pd.read_csv(csv_file_root, dtype=str)
        data_original['race_id'] = data_original['race_id'].str[:-2]
        data_original['baseinfo'] = data_original['baseinfo'].str.ljust(576, '0')

        # convert 1-3 to 1, and others to 0
        data_original['score'] = data_original['score'].apply(lambda x: 0 if int(x) > 3 else int(x))
        targets = data_original.groupby('race_id')['score'].apply(lambda x: np.pad(x.values, (0, 18 - x.values.shape[0]), 'constant', constant_values='0')).to_numpy()
        targets = np.stack(targets, axis=0).astype(np.float32)

        # group data by race
        data = data_original.groupby('race_id')['baseinfo'].apply(lambda x: ''.join(x).ljust(576 * 18, '0')).to_numpy()
        for idx, x in np.ndenumerate(data):
            data[idx] = np.asarray(list(x), dtype=np.uint8)

        # convert to  (H x W x C) in the range [0, 255]
        data = np.reshape(np.stack(data, axis=0), (-1, 24, 24, 18))
        data = (data / 9 * 255).astype(np.uint8)

        self.transform = transform
        self.targets = targets
        self.data = data

        # default set to 0
        self.odds = np.zeros_like(targets)
        if 'odds' in data_original.columns:
            data_original['odds'] = data_original['odds'].astype('float64')
            odds = data_original.groupby('race_id')['odds'].apply(lambda x: np.pad(x.values, (0, 18 - x.values.shape[0]), 'constant', constant_values='0')).to_numpy()
            odds = np.stack(odds, axis=0)
            self.odds = odds

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        
        if self.transform:
            data = self.transform(self.data[idx])
        else:
            data = self.data[idx]
        
        return data, self.targets[idx], (None if self.odds is None else self.odds[idx])