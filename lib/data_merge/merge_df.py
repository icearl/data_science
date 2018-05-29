def all_feature(tfidf_score_feature_df, classic_feature_df):
    print('all_feature.......start')
    all_feature_df = tfidf_score_feature_df.join(classic_feature_df)
    # print('all_feature_df', all_feature_df)
    print('all_feature.......end')
    return all_feature_df