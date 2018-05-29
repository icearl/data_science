
CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.append("%s/.."%(CURRENT_DIR))
import time
import os
import sys
from lib import *


def applist_feature():
    start_time = time.time()
    # 获取数据
    header_list = ['uid', 'is_overdue', 'installed_apk']
    app_cate_list = ['bad']
    # df = txt_to_df('../data/libing_join_applist_data.txt', header_list)
    df = txt_to_df('../data/libing_join_applist_data_test.txt', header_list)
    df = drop_nan(df, 'installed_apk')
    # 拆分数据
    train_df, test_df = split(df, 0.5)
    # 获取 tfidf 分数，作为特征
    # tfidf_score_feature_df = tfidf_score_feature_by_multi_cates(train_df, test_df, 'installed_apk', 'is_overdue', app_cate_list)
    # 获取统计特征，作为特征：一个用户 1. 含有某个类别app的个数 2. 该类别个数 / 该用户app总数
    # classic_feature_df = multi_classic_feature(app_cate_list, train_df, test_df, 'installed_apk', 'is_overdue')
    # 原始 tfidf
    origin_tfidf_df = tfidf_origin_score_feature(train_df, test_df, 'installed_apk', 'is_overdue')
    # 拼接所有特征
    # all_feature_df = all_feature(tfidf_score_feature_df, classic_feature_df)
    all_feature_df = all_feature(tfidf_score_feature_df, origin_tfidf_df)
    # 1. 测试所有原始特征
    pred_cross_val(test_df, 'is_overdue', all_feature_df, '不标准化，不组合：all_feature_df')
    # # 特征标准化
    all_standary_feature_df = all_standary_feature(all_feature_df)
    # # 2. 测试所有标准化特征
    # pred_cross_val(test_df, 'is_overdue', all_standary_feature_df, '标准化，不组合：all_standary_feature_df')
    # # 组合所有标准化特征
    # all_cross_standary_feature = get_cross_feature(all_standary_feature_df)
    # # 组合特征
    # all_cross_feature = get_cross_feature(all_feature_df)
    # # 3. 先标准，再组合：测试组合后的所有标准化特征（包括组合之前的特征）
    # pred_cross_val(test_df, 'is_overdue', all_cross_standary_feature, '先标准化，再组合：all_cross_standary_feature')
    # # 4. 不标准化，直接组合：测试组合后的所有特征（包括组合之前的特征）
    # pred_cross_val(test_df, 'is_overdue', all_cross_feature, '不标准化，直接组合：all_cross_feature')
    end_time = time.time()
    spend_time = end_time - start_time
    print('spend_time:', spend_time)


if __name__ == '__main__':
    applist_feature()
