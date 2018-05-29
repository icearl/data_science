from lib import *

def tfidf_origin_score_feature(train_df, test_df, feature_name, label_name):
    """
    不提取特征，所有的特征
    :param app_cate: app 类别名
    :return:df格式，根据 kind 增加了 tfidf 分数
    """
    print('origin start')
    x_train = train_df[feature_name]
    y_train = train_df[label_name]
    x_test = test_df[feature_name]
    y_test = test_df[label_name]

    split_word = ' '
    count_vectorizer = CountVectorizer(min_df=1, lowercase=False, vocabulary=None,
                                       token_pattern=r'(?u)(?<={}).*?(?={})|(?u)^.*?(?={})|(?u)(?<={}).*?$'.format(
                                           split_word, split_word, split_word, split_word))
    counts_csr_train = count_vectorizer.fit_transform(x_train)

    tfidf_transformer = TfidfTransformer(smooth_idf=False)
    tf_idf_csr_train = tfidf_transformer.fit_transform(counts_csr_train)

    x_train = tf_idf_csr_train
    # print('x_train')
    # 测试集
    counts_csr_test = count_vectorizer.transform(x_test)
    # # print('counts_csr_test', counts_csr_test)
    # app_list = count_vectorizer.get_feature_names()
    # csr_df = counts_csr_test.todense()
    # applist_df = pd.DataFrame(csr_df, columns=app_list)
    # print('is nan', np.isnan(applist_df).any().any())
    # applist_df.to_csv('applist_df_11.csv', encoding="utf-8")

    tf_idf_csr_test = tfidf_transformer.transform(counts_csr_test)
    x_test = tf_idf_csr_test
    # print('x_test')

    feature_name_list = count_vectorizer.get_feature_names()
    # 模型训练
    # auc_score, pred_proba = sk_xgb_cross_train(x_train, x_test, y_train, y_test, feature_name_list)
    # print('f_name')
    auc_score, pred_proba = lr_cross_val(x_train, x_test, y_train, y_test)
    # auc_score, pred_proba_1 = xgb_cross_train(x_train, x_test, y_train, y_test)

    print('origin auc_score', auc_score)
    pred_proba_list = pred_proba[:, 1]
    # pred_proba_list = pred_proba_1
    # print(pred_list)
    res_df = pd.DataFrame()
    res_df['origin_tfidf'] = pred_proba_list
    print('origin end')
    return res_df


def tfidf_score_feature_by_single_cate(train_df, test_df, feature_name, label_name, app_cate):
    """

    :param app_cate: app 类别名
    :return:df格式，根据 kind 增加了 tfidf 分数
    """
    if app_cate != 'None':
        app_cate_name_list = single_kind_by_config_file(app_cate)

    x_train = train_df[feature_name]
    y_train = train_df[label_name]
    x_test = test_df[feature_name]
    y_test = test_df[label_name]

    split_word = ' '
    if app_cate == 'None':
        count_vectorizer = CountVectorizer(min_df=1, lowercase=False, vocabulary=None,
                                           token_pattern=r'(?u)(?<={}).*?(?={})|(?u)^.*?(?={})|(?u)(?<={}).*?$'.format(
                                               split_word, split_word, split_word, split_word))
        counts_csr_train = count_vectorizer.fit_transform(x_train)
    else:
        count_vectorizer = CountVectorizer(min_df=1, lowercase=False, vocabulary=app_cate_name_list,
                                           token_pattern=r'(?u)(?<={}).*?(?={})|(?u)^.*?(?={})|(?u)(?<={}).*?$'.format(
                                               split_word, split_word, split_word, split_word))
        counts_csr_train = count_vectorizer.transform(x_train)

    # tfidf_transformer = TfidfTransformer(smooth_idf=False)
    # tf_idf_csr_train = tfidf_transformer.fit_transform(counts_csr_train)

    x_train = counts_csr_train

    # 测试集
    counts_csr_test = count_vectorizer.transform(x_test)
    # # print('counts_csr_test', counts_csr_test)
    # app_list = count_vectorizer.get_feature_names()
    # csr_df = counts_csr_test.todense()
    # applist_df = pd.DataFrame(csr_df, columns=app_list)
    # print('is nan', np.isnan(applist_df).any().any())
    # applist_df.to_csv('applist_df_11.csv', encoding="utf-8")

    # tf_idf_csr_test = tfidf_transformer.transform(counts_csr_test)
    x_test = counts_csr_test

    feature_name_list = count_vectorizer.get_feature_names()
    # 模型训练
    # auc_score, pred_proba = sk_xgb_cross_train(x_train, x_test, y_train, y_test, feature_name_list)
    auc_score, pred_proba = lr_cross_val(x_train, x_test, y_train, y_test)
    # auc_score, pred_proba_1 = xgb_cross_train(x_train, x_test, y_train, y_test)

    print(app_cate, 'auc_score', auc_score)
    pred_proba_list = pred_proba[:, 1]
    # pred_proba_list = pred_proba_1
    # print(pred_list)
    return pred_proba_list


def tfidf_score_feature_by_multi_cates(train_df, test_df, feature_name, label_name, app_cate_list):
    """

    :param train_df:
    :param test_df:
    :param feature_name:
    :param label_name:
    :param app_cate_list: app 类别列表
    :return:tf-idf scores of all keywords.
    """
    print('tfidf_score_feature_by_multi_cates.......start')
    # res_df = pd.DataFrame(test_df['uid'])
    res_df = pd.DataFrame()
    # print('res_df', res_df, '111', type(res_df))
    for app_cate in app_cate_list:
        temp_pred_list = tfidf_score_feature_by_single_cate(train_df, test_df, feature_name, label_name, app_cate)
        # print(res_df['uid'])
        res_df[app_cate + '_tfidf'] = temp_pred_list
    # print('tfidf_score_df', res_df)
    print('tfidf_score_feature_by_multi_cates.......end')
    return res_df


def single_classic_feature(app_cate, train_df, test_df, feature_name, label_name):
    app_cate_name_list = single_kind_by_config_file(app_cate)
    train_x = train_df[feature_name]
    train_y = train_df[label_name]
    test_x = test_df[feature_name]
    test_y = test_df[label_name]

    split_word = ' '
    count_vectorizer = CountVectorizer(min_df=1, lowercase=False, vocabulary=None,
                                       token_pattern=r'(?u)(?<={}).*?(?={})|(?u)^.*?(?={})|(?u)(?<={}).*?$'.format(
                                           split_word, split_word, split_word, split_word))
    counts_csr_train = count_vectorizer.fit_transform(train_x)
    counts_csr_test = count_vectorizer.transform(test_x)
    app_list = count_vectorizer.get_feature_names()
    csr_df = counts_csr_test.todense()
    applist_df = pd.DataFrame(csr_df, columns=app_list)
    cate_stat_df = stat_cate(applist_df, app_cate_name_list, app_list)
    return cate_stat_df


def multi_classic_feature(app_cate_list, train_df, test_df, feature_name, label_name):
    print('multi_classic_feature.......start')
    res_df = pd.DataFrame()
    # print('res_df', res_df, '111', type(res_df))
    for app_cate in app_cate_list:
        temp_classic_feature_df = single_classic_feature(app_cate, train_df, test_df, feature_name, label_name)
        # print(res_df['uid'])
        # print('res_df', res_df)
        # print('temp_classic_feature_df', temp_classic_feature_df)
        res_df[app_cate + '_cate_app_num'] = temp_classic_feature_df['cate_app_num']
        res_df[app_cate + '_cate/all'] = temp_classic_feature_df['cate/all']
        # res_df.loc[:, app_cate + '_cate/all'] = temp_classic_feature_df
    # print('multi_classic_feature_df', res_df)
    print('multi_classic_feature.......done')
    return res_df


def all_app_num(row_list):
    s = pd.Series(row_list)
    return sum(s)


def cate_app_num(row_list, **kw):
    bool_list = kw['bool_list']
    s1 = pd.Series(row_list)
    s2 = pd.Series(bool_list)
    s = (s1 + s2) == 2
    return sum(s)


def get_bool_list(app_cate_name_list, app_list):
    # 把 applist 转成 0/1 的 list
    bool_list = []
    cnt = 0
    count = 0
    for app_name in app_cate_name_list:
        cnt += 1
        if app_name in app_list:
            bool_list.append(1)
            count += 1
        else:
            bool_list.append(0)
    return bool_list


def stat_cate(applist_df, app_cate_name_list, app_list):
    bool_list = get_bool_list(app_cate_name_list, app_list)
    stat_df = pd.DataFrame()
    stat_df['all_app_num'] = applist_df.apply(lambda row: all_app_num(list(row)), axis=1)
    stat_df['cate_app_num'] = applist_df.apply(lambda row: cate_app_num(list(row), bool_list=bool_list), axis=1)
    stat_df['cate/all'] = stat_df['cate_app_num'] / stat_df['all_app_num']
    # print('stat  all_app_num', stat_df['all_app_num'])
    # print('cate_app_num', stat_df['cate_app_num'])
    # print('cate/all', stat_df['cate/all'])
    return stat_df


def applist_df(df, feature_name, label_name):
    """

    :param df: 原始df
    :param feature_name:applist 的字段名
    :param label_name:标签名
    :return:df 格式，每列为某个 app 是否安装的 0 / 1 值
    """
    data_x = df[feature_name]
    data_y = df[label_name]
    # x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.2)
    split_word = ' '
    count_vectorizer = CountVectorizer(min_df=1, lowercase=False, max_features=None,
                                       token_pattern='(?u)(?<={}).*?(?={})|(?u)^.*?(?={})|(?u)(?<={}).*?$'.format(
                                           split_word, split_word, split_word, split_word))
    counts_csr_train = count_vectorizer.fit_transform(data_x)
    tfidf_transformer = TfidfTransformer(smooth_idf=False)
    tf_idf_csr_train = tfidf_transformer.fit_transform(counts_csr_train)

    data_x = tf_idf_csr_train

    # counts_csr_test = count_vectorizer.transform(x_test)
    # tf_idf_csr_test = tfidf_transformer.transform(counts_csr_test)
    # x_test = tf_idf_csr_test

    app_list = count_vectorizer.get_feature_names()
    csr_df = counts_csr_train.todense()
    applist_df = pd.DataFrame(csr_df, columns=app_list)

    return app_list, applist_df