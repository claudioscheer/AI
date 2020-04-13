import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms
import pandas as pd


class BostonDataset(Dataset):
    def __init__(self, csv_path):
        self.data = pd.read_csv(csv_path, header=None)
        self.house_size = self.data.iloc[:, 0]  # house size in square feet
        self.bedrooms = self.data.iloc[:, 1]  # bedrooms
        self.price = self.data.iloc[:, 2]  # price
        self.data_len = len(self.data.index)

    def __getitem__(self, index):
        x = [self.house_size[index], self.bedrooms[index]]
        x = torch.Tensor(x).cuda()
        y = [self.price[index]]
        y = torch.Tensor(y).cuda()
        return x, y

    def __len__(self):
        return self.data_len
