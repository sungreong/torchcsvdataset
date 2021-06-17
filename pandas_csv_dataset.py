import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, Subset, DataLoader
from sklearn import datasets
import pandas as pd
import numpy as np



class CSVDataset(torch.utils.data.Dataset):
    def __init__(self, path, chunksize, nb_samples, transform=None):
        self.path = path
        self.chunksize = chunksize
        self.len = nb_samples // self.chunksize
        self.transform = transform

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        x = next(
            pd.read_csv(
                self.path,
                skiprows=idx * self.chunksize,  #+1 to skip the header
                chunksize=self.chunksize))

        target = torch.as_tensor(x.values)[0][-1]  # last col
        features = torch.as_tensor(x.values)[0][0:-1]  # all but last

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
    csv_dataset = CSVDataset('./data/classification_demo.csv', chunksize=1, nb_samples=1000, transform=None)

    loader = DataLoader(csv_dataset, batch_size=100, pin_memory=True) # collate_fn=collate_wrapper,
    for batch_ndx, sample_ in enumerate(loader):
        print(sample_['idx'])