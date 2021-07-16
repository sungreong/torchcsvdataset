from sklearn import datasets
import vaex
import pandas as pd 
import argparse 


def get_params():
    ''' Get parameters from command line '''
    parser = argparse.ArgumentParser()

    # Name of trial
    parser.add_argument("--n_samples" , type = int , default=100_000)
    parser.add_argument("--n_features" , type = int , default=2)
    params = vars(parser.parse_known_args()[0])

    return params

if __name__ == "__main__": 
    params = get_params()
    X, y = datasets.make_classification(n_samples=params["n_samples"],n_features=params["n_features"],
                                            n_informative=2,
                                            n_redundant=0,
                                            n_classes=2,
                                            random_state=15)
    # place data into df
    df = pd.DataFrame(X)
    df["y"] = y
    df.to_csv('./data/classification_demo.csv', index=False)
    vaex.open('./data/classification_demo.csv',convert=True)