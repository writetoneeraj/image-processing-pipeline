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

label_to_num = {
                    'any': 0,
                    'epidural': 1,
                    'subdural': 2,
                    'subarachnoid': 3,
                    'intraventricular': 4,
                    'intraparenchymal': 5
}

num_to_label = {v:k for k,v in num_to_labels.items()}