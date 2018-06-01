import time
import os
import sys
CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.append("%s/.."%(CURRENT_DIR))
from lib import *


def applist_test():
    start_time = time.time()
    # 获取数据
    header_list = ['uid', 'is_overdue', 'installed_apk']
    # app_cate_list = ['white', 'bad', 'dept', 'tools']
    app_cate_list = ['white', 'bad', 'dept', 'dept_on_credit', 'dept_not_on_credit', 'tools']
    # df = txt_to_df('../data/libing_join_applist_data.txt', header_list)
    df = txt_to_df('../data/libing_join_applist_data_test.txt', header_list)
    df = drop_nan(df, 'installed_apk')
    # 拆分数据
    train_df, test_df = split(df, 0.5)
    # print('test_df', test_df)
    test_df, new_df = split(test_df, 0.5)
    print('test_df', len(test_df))
    print('new_df', len(new_df))
    # 获取 tfidf 分数，作为特征
    test_tfidf_score_feature_df, new_tfidf_score_feature_df = \
        tfidf_score_feature_by_multi_cates(train_df, test_df, new_df, 'installed_apk', 'is_overdue', app_cate_list)
    # 获取统计特征，作为特征：一个用户 1. 含有某个类别app的个数 2. 该类别个数 / 该用户app总数
    test_classic_feature_df, new_classic_feature_df = multi_classic_feature(app_cate_list, train_df, test_df, new_df, 'installed_apk', 'is_overdue')
    # 原始 tfidf
    test_res_df, new_res_df = \
        tfidf_origin_score_feature(train_df, test_df, new_df, 'installed_apk', 'is_overdue')
    # 拼接所有特征
    # print('new_tfidf_score_feature_df', len(new_tfidf_score_feature_df))
    # print('new_classic_feature_df', len(new_classic_feature_df))
    test_all_feature_df = all_feature(test_tfidf_score_feature_df, test_classic_feature_df)
    test_all_feature_df = all_feature(test_all_feature_df, test_res_df)
    new_all_feature_df = all_feature(new_tfidf_score_feature_df, new_classic_feature_df)
    new_all_feature_df = all_feature(new_all_feature_df, new_res_df)
    # print('test_all_feature_df', len(test_all_feature_df))
    # print('new_all_feature_df', len(new_all_feature_df))
    # 1. 测试所有原始特征
    # pred_cross_val(new_df, 'is_overdue', test_all_feature_df, '不标准化，不组合：all_feature_df')
    pred(test_df, test_all_feature_df, new_df, new_all_feature_df, 'is_overdue', '不标准化，不组合：all_feature_df')
    # pred(test_df, test_res_df, new_df, new_res_df, 'is_overdue', '不标准化，不组合：all_feature_df')
    # # 特征标准化
    # all_standary_feature_df = all_standary_feature(all_feature_df)
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


def applist_pred():
    start_time = time.time()
    # 获取数据
    header_list = ['uid', 'is_overdue', 'installed_apk']
    # app_cate_list = ['white', 'bad', 'dept', 'dept_on_credit', 'dept_not_on_credit', 'tools',
    #                  'camera', 'card_game', 'chat', 'edu', 'game', 'health', 'joy', 'life',
    #                  'news', 'read', 'shop', 'sport', 'tools', 'top_bad_list', 'traffic',
    #                  'video', 'work']
    # app_cate_list = ['white', 'bad', 'dept', 'dept_on_credit', 'dept_not_on_credit', 'tools']
    app_cate_list = ['dept', 'tools']
    # df = txt_to_df('../data/libing_join_applist_data.txt', header_list)
    df = txt_to_df('../data/libing_join_applist_data_test.txt', header_list)
    df = drop_nan(df, 'installed_apk')
    # 拆分数据
    lists_3k = pickle.load(open('3k_uids_for_test.pkl', 'rb'))
    lists_6k = pickle.load(open('6k_uids_for_train.pkl', 'rb'))
    lists_9k = lists_3k + lists_6k
    print('lists_9k len', len(lists_9k))
    list_qian = lists_9k[:4500]
    list_hou = lists_9k[4500:]
    train_df, test_df, new_df = data_split_pred(df, list_qian)

    print('test_df', len(test_df))
    print('new_df', len(new_df))
    # 获取 tfidf 分数，作为特征
    test_tfidf_score_feature_df, new_tfidf_score_feature_df = \
        tfidf_score_feature_by_multi_cates(train_df, test_df, new_df, 'installed_apk', 'is_overdue', app_cate_list)
    # 获取统计特征，作为特征：一个用户 1. 含有某个类别app的个数 2. 该类别个数 / 该用户app总数
    test_classic_feature_df, new_classic_feature_df = multi_classic_feature(app_cate_list, train_df, test_df, new_df, 'installed_apk', 'is_overdue')
    # 原始 tfidf
    test_res_df, new_res_df = \
        tfidf_origin_score_feature(train_df, test_df, new_df, 'installed_apk', 'is_overdue')
    # 拼接所有特征
    print('new_tfidf_score_feature_df', len(new_tfidf_score_feature_df))
    print('new_classic_feature_df', len(new_classic_feature_df))
    test_all_feature_df = all_feature(test_tfidf_score_feature_df, test_classic_feature_df)
    test_all_feature_df = all_feature(test_all_feature_df, test_res_df)
    new_all_feature_df = all_feature(new_tfidf_score_feature_df, new_classic_feature_df)
    new_all_feature_df = all_feature(new_all_feature_df, new_res_df)
    print('test_all_feature_df', len(test_all_feature_df))
    print('new_all_feature_df len', len(new_all_feature_df))
    new_all_feature_df.to_csv('list_qian.csv', encoding="utf-8")

    # 1. 测试所有原始特征
    # pred_cross_val(new_df, 'is_overdue', test_all_feature_df, '不标准化，不组合：all_feature_df')
    pred(test_df, test_all_feature_df, new_df, new_all_feature_df, 'is_overdue', '不标准化，不组合：all_feature_df')
    # # 特征标准化
    # all_standary_feature_df = all_standary_feature(all_feature_df)
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
    applist_test()
    # applist_pred()

