# from sklearn.model_selection._split import _BaseKFold, _RepeatedSplits, BaseShuffleSplit, _validate_shuffle_split
import argparse
from misc import utility

def groupk_split():
    from sklearn.model_selection import GroupKFold
    raise NotImplementedError("Not yet implemented")
    
def kfold_split():
    from sklearn.model_selection import KFold
    raise NotImplementedError("Not yet implemented")

def stratifiedkfold_split():
    from sklearn.model_selection import StratifiedKFold
    raise NotImplementedError("Not yet implemented")

def group_fold(df, folds, group):
    #folddf = df.groupby(group).reset_index(drop=True).drop_duplicates().reset_index()
    folddf = df['PatientID'].reset_index(drop=True).drop_duplicates().reset_index()
    folddf['fold'] = (folddf['index'].values)%5
    folddf = folddf.drop('index', 1)
    #folddf['fold'] = int((folddf['index'].values)%folds)
    # folddf = folddf.drop('index', 1)
    df = df.merge(folddf, left_on='PatientID', right_on='PatientID')
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
    # method = getattr(make_folds, args.method)
    # print("method : ", method)
    train_folds = group_fold(utility.loadpickle(args.input), args.folds, args.group)
    print(train_folds)
    train_folds.to_pickle(args.output)