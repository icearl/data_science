def score_train(train_df, feature_name, label_name, app_list):
    data_x = train_df[feature_name]
    data_y = train_df[label_name]
    split_word = ' '
    count_vectorizer = CountVectorizer(min_df=1, lowercase=False, vocabulary=app_list,
                                       token_pattern='(?u)(?<={}).*?(?={})|(?u)^.*?(?={})|(?u)(?<={}).*?$'.format(
                                           split_word, split_word, split_word, split_word))
    counts_csr_train = count_vectorizer.fit_transform(data_x)
    tfidf_transformer = TfidfTransformer(smooth_idf=False)
    tf_idf_csr_train = tfidf_transformer.fit_transform(counts_csr_train)

    data_x = tf_idf_csr_train

    # 模型训练
    model = LogisticRegression(C=1)
    model.fit(data_x, data_y)

    return count_vectorizer, tfidf_transformer, model


