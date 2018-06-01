import os
import sys
CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.append("%s/../../"%(CURRENT_DIR))
from lib import *
# from .split import *
#

print(sys.path)



def data_split_pred(df, pred_list):
    new_df = df[df['uid'].isin(pred_list)]
    left_df = df.drop(new_df.index)
    train_df, test_df = split(left_df, 0.5)
    # print('df:', len(df))
    # print('new_df:', len(new_df))
    # print('train_df:', len(train_df))
    # print('test_df:', len(test_df))
    return train_df, test_df, new_df


def data_split_train(df, train_list):
    left_df = df[df['uid'].isin(train_list)]
    new_df = df.drop(left_df.index)
    train_df, test_df = split(left_df, 0.5)
    print('df:', len(df))
    print('new_df:', len(new_df))
    print('train_df:', len(train_df))
    print('test_df:', len(test_df))
    return train_df, test_df, new_df

def data_new(df, uid, is_overdue, applist_list):
    more_df = pd.DataFrame([[uid, is_overdue, applist_list]], columns=df.columns)
    new_df = df.append(more_df)
    return new_df