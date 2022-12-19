from sklearn.model_selection._split import _BaseKFold, _RepeatedSplits, BaseShuffleSplit, _validate_shuffle_split
import argparse
from src.common import utility
import numpy as np

def groupk_split():
    from sklearn.model_selection import GroupKFold
    
def kfold_split():
    from sklearn.model_selection import KFold

def stratifiedkfold_split():
    from sklearn.model_selection import StratifiedKFold

def _get_fold(x, folds):
    return x['index'].values%np.int(folds)

def group_fold(df, folds, group):
    df = df.drop_duplicates().reset_index(drop=True)
    df['fold'] = df.groupby(group).ngroup()
    df['fold'] = df['fold'] % np.int(folds)
    return df

def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input')
    parser.add_argument('--output')
    parser.add_argument('--folds')
    parser.add_argument('--group')
    parser.add_argument('--method')
    parser.add_argument('--seed')
    return parser.parse_args()

if __name__ == '__main__':
    args=args_parser()
    current_module = __import__(__name__)
    method = getattr(current_module ,args.method)
    train_folds = method(utility.loadpickle(args.input), args.folds, args.group)
    train_folds.to_pickle(args.output)