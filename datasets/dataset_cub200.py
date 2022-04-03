import cv2
import pandas as pd
from torch.utils.data import Dataset


class cub200_train_dataset(Dataset):
    def __init__(self):
        df = pd.read_csv('cub200/lists/train.txt', header=None, names=['path'])
        self.paths = df['path']
        self.labels = df.path.map(lambda x: str(x).split('/')[0].split('.')[1])

    def __getitem__(self, idx):
        image = cv2.imread('cub200/images/' + self.paths[idx])
        label = self.labels[idx]
        return image, label

    def __len__(self):
        return len(self.paths)


class cub200_test_dataset(Dataset):
    def __init__(self):
        df = pd.read_csv('cub200/lists/test.txt', header=None, names=['path'])
        self.paths = df['path']
        self.labels = df.path.map(lambda x: str(x).split('/')[0].split('.')[1])

    def __getitem__(self, idx):
        image = cv2.imread('cub200/images/' + self.paths[idx])
        label = self.labels[idx]
        return image, label

    def __len__(self):
        return len(self.paths)