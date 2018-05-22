def score_cross_val(train_df, feature_name, label_name, app_list):
    data_x = train_df[feature_name]
    data_y = train_df[label_name]
    x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.2)
    # 训练集
    split_word = ' '
    count_vectorizer = CountVectorizer(min_df=1, lowercase=False, vocabulary=app_list,
                                       token_pattern='(?u)(?<={}).*?(?={})|(?u)^.*?(?={})|(?u)(?<={}).*?$'.format(
                                           split_word, split_word, split_word, split_word))
    counts_csr_train = count_vectorizer.fit_transform(x_train)
    tfidf_transformer = TfidfTransformer(smooth_idf=False)
    tf_idf_csr_train = tfidf_transformer.fit_transform(counts_csr_train)

    x_train = tf_idf_csr_train
    # 测试集
    counts_csr_test = count_vectorizer.transform(x_test)
    tf_idf_csr_test = tfidf_transformer.transform(counts_csr_test)
    x_test = tf_idf_csr_test
    # 模型训练
    lr = LogisticRegression(C=1)
    lr.fit(x_train, y_train)
    pred_proba = lr.predict_proba(x_test)
    fpr, tpr, thresholds = roc_curve(y_test, pred_proba[:, 1])
    auc_score = auc(fpr, tpr)
    print('auc_score', auc_score)

    return None