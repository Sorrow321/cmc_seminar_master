import pandas as pd
import json
from torch.utils.data import Dataset


class cub200_dataset(Dataset):
    def __init__(self, split):
        if split not in ['train', 'test']:
            raise Exception('wrong split: should be train or test')
        df = pd.read_csv(f'data/cub200/lists/{split}.txt', header=None, names=['path'])
        with open('data/cub200/class_to_idx.json', 'rt') as file:
            self.class_to_idx = json.load(file)
        self.paths = list(df['path'])
        self.labels = list(df.path.map(lambda x: str(x).split('/')[0].split('.')[1]))

    def __getitem__(self, idx):
        path = 'cub200/images/' + self.paths[idx]
        label = self.class_to_idx[self.labels[idx]]
        return path, label

    def __len__(self):
        return len(self.paths)