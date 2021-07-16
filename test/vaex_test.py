import torch
from torch.utils.data import Dataset, Subset, DataLoader
from sklearn import datasets
import vaex
import pandas as pd 
import numpy as np , sys
sys.path.append("./")
from src.vaex_csv_dataset import VaexDataset
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
    params = get_params()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # X, y = datasets.make_classification(n_samples=1000,
    #                                         n_features=2,
    #                                         n_informative=2,
    #                                         n_redundant=0,
    #                                         n_classes=2,
    #                                         random_state=15)
    # place data into df
    #df = pd.DataFrame({'x1': X[:, 0], 'x2': X[:, 1], 'y': y})
    #df.to_csv('./data/classification_demo.csv', index=False)
    #vaex.open('./data/classification_demo.csv',convert=True)
    csv_dataset = VaexDataset('./data/classification_demo.csv.hdf5', transform=None)
    print(len(csv_dataset))
    loader = DataLoader(csv_dataset, batch_size=params["batch_size"], pin_memory=True) # collate_fn=collate_wrapper,
    for batch_ndx, sample_ in enumerate(loader):
        print(sample_['idx'])