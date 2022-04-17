import cv2
from torch.utils.data import Dataset


class BaseImageDataset(Dataset):
    def __init__(self, dataset, path_prefix='', augmentations=None):
        self.dataset = dataset
        self.augmentations = augmentations
        self.path_prefix = path_prefix
    
    def __getitem__(self, idx):
        path, label = self.dataset[idx]
        path = self.path_prefix + path
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.augmentations is not None:
            img = self.augmentations(image=img)['image']
        return img, label


    def __len__(self):
        return len(self.dataset)