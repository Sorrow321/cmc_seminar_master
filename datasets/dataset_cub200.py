import cv2
import pandas as pd
from torch.utils.data import Dataset


class cub200_dataset(Dataset):
    def __init__(self, split):
        if split not in ['train', 'test']:
            raise Exception('wrong split: should be train or test')
        df = pd.read_csv(f'cub200/lists/{split}.txt', header=None, names=['path'])
        self.paths = df['path']
        self.labels = df.path.map(lambda x: str(x).split('/')[0].split('.')[1])

    def __getitem__(self, idx):
        image = cv2.imread('cub200/images/' + self.paths[idx])
        label = self.labels[idx]
        return image, label

    def __len__(self):
        return len(self.paths)