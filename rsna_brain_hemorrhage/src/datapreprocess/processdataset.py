from src.common import utility, dicomdata
import argparse
from matplotlib import pyplot as plt
import pandas as pd

def arg_parser():
    """Read arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--input')
    parser.add_argument('--output')
    parser.add_argument('--brain-diff', type=float)
    return parser.parse_args() 

def show_distribution(df):
    """Get value counts of all labels and plot for 0 and 1"""
    value_counts_df = df[['any','epidural','intraparenchymal','intraventricular','subarachnoid','subdural']].apply(lambda x : x.value_counts(), axis=0)
    value_counts_df.T.plot.bar()

def parse_position(df):
    """Split all values of position into seperate columns
    Should be moved under utility.
    """
    expanded = df.ImagePositionPatient.apply(lambda x: pd.Series(x))
    expanded.columns = ['Position1', 'Position2', 'Position3']
    return pd.concat([df, expanded], axis=1)


def parse_orientation(df):
    """Split all values of image orientation into seperate columns
    Should be moved under utility.
    """
    expanded = df.ImageOrientationPatient.apply(lambda x: pd.Series(x))
    expanded.columns = ['Orient1', 'Orient2', 'Orient3', 'Orient4', 'Orient5', 'Orient6']
    return pd.concat([df, expanded], axis=1)

def windowing(df, args):
    brain_diff = args.brain_diff
    if brain_diff:
        df = df[df.brain_diff > brain_diff]
    df['WindowCenter'] = df.WindowCenter.apply(lambda x: dicomdata.get_dicom_value(x))
    df['WindowWidth'] = df.WindowWidth.apply(lambda x: dicomdata.get_dicom_value(x))
    df['PositionOrd'] = df.groupby('SeriesInstanceUID')[['Position3']].rank() / df.groupby('SeriesInstanceUID')[['Position3']].transform('count')
    return df

def processdata():
    """Read dicom metatdata and labels pickle processed in createdataset.py file
    Split position and image orientation values in individual columns
    Check window center and width get one value only.
    Add new field for Position Order.
    Save output in output pickle format.
    Can be merged with createdataset.py. Rework on next iteration.
    """
    args = arg_parser()
    df = utility.loadpickle(args.input)
    df = parse_orientation(df)
    df = parse_position(df)
    df = windowing(df, args)
    show_distribution(df)
    df = df.drop(columns='Unnamed: 0', axis=1)
    df.to_pickle(args.output)

if __name__ == '__main__':
    processdata()