import pandas as pd
from torch.utils.data import Dataset


class stanford_products_dataset(Dataset):
    def __init__(self, split):
        if split not in ['train', 'test']:
            raise Exception('wrong split: should be train or test')
        if split == 'train':
            path = 'Stanford_Online_Products/Ebay_train.txt'
        else:
            path = 'Stanford_Online_Products/Ebay_test.txt'
        self.df = pd.read_csv(path, sep=' ')[['path', 'class_id']]

    def __getitem__(self, idx):
        path = 'Stanford_Online_Products/' + self.df.iloc[idx]['path']
        label = self.df.iloc[idx]['class_id']
        return path, label

    def __len__(self):
        return self.df.shape[0]