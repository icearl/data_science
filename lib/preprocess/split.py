import os
import sys
CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.append("%s/../.."%(CURRENT_DIR))
from lib import *
def split(df, ritio_of_train):
    """

    :param df:
    :param ritio_of_train:
    :return:
    """
    df = df.sample(frac=1.0)  # 全部打乱
    cut_idx = int(round(ritio_of_train * df.shape[0]))
    df_train, df_test = df.iloc[:cut_idx], df.iloc[cut_idx:]
    # df_train.to_csv('df_train.csv', encoding="utf-8")
    # df_test.to_csv('df_test.csv', encoding="utf-8")
    return df_train, df_test