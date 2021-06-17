import torch
from torch.utils.data import Dataset, Subset, DataLoader
from sklearn import datasets
import vaex
import pandas as pd 
import numpy as np , sys
sys.path.append("./")
from src import VaexDataset

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
    print(len(csv_dataset))
    loader = DataLoader(csv_dataset, batch_size=300, pin_memory=True) # collate_fn=collate_wrapper,
    for batch_ndx, sample_ in enumerate(loader):
        print(sample_['idx'])