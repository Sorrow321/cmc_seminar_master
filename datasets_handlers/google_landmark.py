# No data is available in public for testing => use train_test_split to gen it


import pandas as pd
import json
from torch.utils.data import Dataset


class google_landmark_dataset(Dataset):
    def __init__(self):
        self.df = pd.read_csv('data/google-landmark/train.csv')
        with open('data/google-landmark/glm_mapping.json', 'rt') as file:
            self.mapping = json.load(file)

    def __getitem__(self, idx):
        name = self.df.iloc[idx]['id']
        path = f'google-landmark/train/{name[0]}/{name[1]}/{name[2]}/{name}.jpg'
        label = self.mapping[str(self.df.iloc[idx]['landmark_id'])]
        return path, label

    def __len__(self):
        return self.df.shape[0]