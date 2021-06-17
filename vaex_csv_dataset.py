import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, Subset, DataLoader
from sklearn import datasets
import vaex
import pandas as pd 
import numpy as np


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, Subset, DataLoader
from sklearn import datasets
import vaex
import pandas as pd 
import numpy as np



class VaexDataset(torch.utils.data.Dataset):
    def __init__(self, path, transform=None):
        self.path = path
        self.df = vaex.open(self.path)
        self.len = len(self.df)
        self.transform = transform

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        x = self.df.take([idx])
        x_np = x.values
        target = torch.as_tensor(x_np)[0][-1]  # last col
        features = torch.as_tensor(x_np)[0][0:-1]  # all but last

        # pull a sample in a dict
        sample = {'features': features,
                  'target': target,
                  'idx': torch.as_tensor(idx)}

        if self.transform:
            sample = self.transform(sample)
        return sample

if __name__ == "__main__": 
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    X, y = datasets.make_classification(n_samples=1000,
                                            n_features=2,
                                            n_informative=2,
                                            n_redundant=0,
                                            n_classes=2,
                                            random_state=15)
    # place data into df
    df = pd.DataFrame({'x1': X[:, 0], 'x2': X[:, 1], 'y': y})
    df.to_csv('./data/classification_demo.csv', index=False)
    vaex.open('./data/classification_demo.csv',convert=True)
    csv_dataset = VaexDataset('./data/classification_demo.csv.hdf5', transform=None)
    loader = DataLoader(csv_dataset, batch_size=300, pin_memory=True) # collate_fn=collate_wrapper,
    for batch_ndx, sample_ in enumerate(loader):
        print(sample_['idx'])