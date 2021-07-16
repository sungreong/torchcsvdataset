
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, Subset, DataLoader
from sklearn import datasets
import pandas as pd
import numpy as np ,sys
sys.path.append("./")
from src.pandas_csv_dataset import CSVDataset
import argparse
def get_params():
    ''' Get parameters from command line '''
    parser = argparse.ArgumentParser()

    # Name of trial
    parser.add_argument("--n_samples" , type = int , default=100_000)
    parser.add_argument("--batch_size" , type = int , default=1_000)
    params = vars(parser.parse_known_args()[0])

    return params

if __name__ == "__main__": 
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    params = get_params()
    csv_dataset = CSVDataset('./data/classification_demo.csv', chunksize=1, nb_samples=params["n_samples"], transform=None)
    loader = DataLoader(csv_dataset, batch_size=params["batch_size"], pin_memory=True) # collate_fn=collate_wrapper,
    for batch_ndx, sample_ in enumerate(loader):
        print(sample_['idx'])