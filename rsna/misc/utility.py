import pandas as pd

def loadpickle(filename):
    """
    Load pickle file in datframe
    returns dataframe
    """
    if filename and not filename.isspace():
        return pd.read_pickle(filename)
    else:
        #logger empty file name
        print("Empty filename")