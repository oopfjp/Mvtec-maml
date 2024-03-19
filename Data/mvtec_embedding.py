import os
import random
import torch
from torch.utils.data import Dataset, DataLoader
from utils import files_to_tensor, cls_files
from category_split import train_cls

class MvTec(Dataset):
    def __init__(self, root = 'D:\\datasets\\meta-learning\\mvtec'):
        self.root = root
        self.remove_cls = 'bottle'
        self.category = [name for name in os.listdir(self.root)]
        self.category.remove(self.remove_cls)
        self.cls_to_files = cls_files(self.category, self.root)
        self.target_to_files = {}
        self.list_to_files = []

        for cls in train_cls:
            self.target_to_files[cls] = self.cls_to_files[cls]

        for idx, (_, files) in enumerate(self.target_to_files.items()):
            for file in files:
                self.list_to_files.append([idx, file])
        random.shuffle(self.list_to_files)

    def __getitem__(self, idx):
        label, data = self.list_to_files[idx]
        return torch.tensor([label],device='cuda'), files_to_tensor([data]).to('cuda')


    def __len__(self):
        return len(self.list_to_files)


if __name__ == "__main__":
    Dataset = MvTec()
    loader = DataLoader(Dataset, batch_size=64)
    a=1
    # val_Dataset = MvTec(args, mode='val')
    # # label, file = Dataset[0]
    # train_dataLoader = DataLoader(train_Dataset, batch_size=64)
    # for idx, (target, input) in enumerate(train_dataLoader):
    #     print(idx)
    #     a = 1
    # a = 1



