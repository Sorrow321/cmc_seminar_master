import pandas as pd
import json
from torch.utils.data import Dataset


class cub200_2011_dataset(Dataset):
    def __init__(self, split):
        if split not in ['train', 'test']:
            raise Exception('wrong split: should be train or test')
        images = pd.read_csv('data/CUB_200_2011/images.txt', header=None, names=['idx', 'path'], delim_whitespace=True)
        labels = pd.read_csv('data/CUB_200_2011/image_class_labels.txt', header=None, names=['idx', 'label'], delim_whitespace=True)
        splits = pd.read_csv('data/CUB_200_2011/train_test_split.txt', header=None, names=['idx', 'split'], delim_whitespace=True)
        df = pd.merge(pd.merge(images, labels), splits)

        if split == 'train':
            self.df = df[df.split == 1]
        else:
            self.df = df[df.split == 0]

    def __getitem__(self, idx):
        path = 'CUB_200_2011/images/' + self.df.iloc[idx]['path']
        label = self.df.iloc[idx]['label']
        return path, label

    def __len__(self):
        return self.df.shape[0]