import pandas as pd
import sys
import argparse
import pydicom
from tqdm import tqdm
import numpy as np
import logging
from src.common import dicomdata

# Get arguments for input of training data and output for processed data.
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input')
    parser.add_argument('--output')
    parser.add_argument('--img_dir')
    return parser.parse_args()
    
def loaddataset(input):
    """
    Read csv file and return dataframe
    """
    return pd.read_csv(input)

def removeduplicates(df, _keep='first', _inplace=True):
    """
    Remove duplicates keeping first row.
    In case inplace is True, drop_duplicates will return None,
    as it drop duplicates from the original dataframe itself
    instead of returning a copy.
    """
    df.drop_duplicates(keep=_keep, inplace=_inplace)
    return df

def splitcolumn(df):
    """
    Method read column value and split in multiple columns. This is very much specific to data.
    """
    df['Hemorrhage'] = df['ID'].apply(lambda x : str(x).rsplit('_',1)[1])
    df['ID'] = df['ID'].apply(lambda x : str(x).rsplit('_',1)[0])
    return df

def pivot_dataframe(args, df, column_name, value, indexcolumn):
    """
    This method is used to convert row wise data to column wise.
    column name for which pivoting is required.
    value to keep for pivoted columns
    indexcolumn on which dataframe has to be indexed.
    """
    df = pd.pivot_table(df, columns=column_name, values=value, index=indexcolumn).reset_index()
    # Todo
    # Replace with configuration
    df['filepath'] = args.img_dir + df['ID'] + '.dcm'
    return df

def remove_corrupted_images(ids, df):
    copy_df = df.copy()
    for id in ids:
        copy_df = copy_df.drop(copy_df[copy_df['ID'] == id].index, axis=0)
    return copy_df

def getdicomdata(train):
    return dicomdata.dicomDataframe(train)
    
def process_data():
    args = get_args()
    df = loaddataset(args.input)
    print(f'Shape of dataset before removing duplicates {df.shape}')
    print(f"Before removing duplicates : {df}")
    df = removeduplicates(df, 'first')
    print(f'Shape of dataset after removing duplicates {df.shape}')
    print(f'df columns : {df.head()}')
    #df = splitcolumn(df)
    #df = pivot_dataframe(args,df,['Hemorrhage'],'Label',['ID'])
    dicom_metadata, corrupted_ids = getdicomdata(df)
    dicom_metadata = pd.DataFrame(dicom_metadata)
    df_uncorrupted = remove_corrupted_images(corrupted_ids, df)
    print("After Processing Dicom metadata: ", dicom_metadata.head())
    df_dicom = df_uncorrupted.merge(dicom_metadata, right_on="ID", left_on="ID")
    print(df_dicom.head())
    print(df_dicom.columns)
    df_dicom.to_pickle(args.output)

if __name__ == '__main__':
    process_data()