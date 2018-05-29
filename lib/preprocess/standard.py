from lib import *

def min_max_scalar(column):
    """
    http://www.cnblogs.com/irran/p/data_preprocess.html
    :param column:
    :return:
    """
    max_min = np.max(column) - np.min(column)
    if max_min == 0:
        column = 0
    else:
        column = (column - np.min(column)) / (np.max(column) - np.min(column)) * 10
    return column


def all_standary_feature(all_feature_df):
    """
    所有特征规范到0-10区间。
    :param all_feature_df:
    :return:
    """
    all_standary_feature_df = all_feature_df.apply(min_max_scalar)
    # print('all_standary_feature_df', all_standary_feature_df)
    return all_standary_feature_df