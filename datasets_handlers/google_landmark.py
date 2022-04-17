# No data is available in public for testing => use train_test_split to gen it


import pandas as pd
from torch.utils.data import Dataset


class google_landmark_dataset(Dataset):
    def __init__(self):
        self.df = pd.read_csv('google-landmark/train.csv')
    
    def __getitem__(self, idx):
        name = self.df.iloc[idx]['id']
        path = f'google-landmark/train/{name[0]}/{name[1]}/{name[2]}/{name}.jpg'
        label = int(self.df.iloc[idx]['landmark_id'])
        return path, label

    def __len__(self):
        return self.df.shape[0]