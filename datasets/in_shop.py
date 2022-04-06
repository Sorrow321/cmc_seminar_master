import cv2
from h11 import Data
import pandas as pd
from torch.utils.data import Dataset


class in_shop_dataset(Dataset):
    def __init__(self, split):
        if split not in ['train', 'gallery', 'query']:
            raise Exception('wrong split: should be train or gallery or query')
        df = pd.read_csv(
            'In-shop Clothes Retrieval Benchmark/Eval/list_eval_partition.txt',
            skiprows=1,
            delim_whitespace=True
        )
        self.df = df[df.evaluation_status == split][['image_name', 'item_id']]

    def __getitem__(self, idx):
        path = 'In-shop Clothes Retrieval Benchmark/Img/' + self.df.iloc[idx]['image_name']
        image = cv2.imread(path)
        label = self.df.iloc[idx]['item_id']
        return image, label

    def __len__(self):
        return self.df.shape[0]