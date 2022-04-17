import pandas as pd
import json
from torch.utils.data import Dataset


class in_shop_dataset(Dataset):
    def __init__(self, split):
        if split not in ['train', 'gallery', 'query']:
            raise Exception('wrong split: should be train or gallery or query')
        df = pd.read_csv(
            'data/In-shop Clothes Retrieval Benchmark/Eval/list_eval_partition.txt',
            skiprows=1,
            delim_whitespace=True
        )
        self.df = df[df.evaluation_status == split][['image_name', 'item_id']]
        with open('data/In-shop Clothes Retrieval Benchmark/class_to_idx.json', 'rt') as file:
            self.class_to_idx = json.load(file)

    def __getitem__(self, idx):
        path = 'In-shop Clothes Retrieval Benchmark/Img/' + self.df.iloc[idx]['image_name']
        label = self.class_to_idx[self.df.iloc[idx]['item_id']]
        return path, label

    def __len__(self):
        return self.df.shape[0]