def get_cross_feature(all_feature):
    """
    加减乘除加上平滑系数
    :param feature:
    :return:
    """
    res_df = all_feature
    cols_str = list(all_feature.columns)
    combins_list = [c for c in combinations(cols_str, 2)]
    for combin_tuple in combins_list:
        first_str = combin_tuple[0]
        second_str = combin_tuple[1]
        res_df[first_str + '+' + second_str] = all_feature[first_str] + all_feature[second_str]
        res_df[first_str + '-' + second_str] = all_feature[first_str] - all_feature[second_str]
        res_df[second_str + '-' + first_str] = all_feature[second_str] - all_feature[first_str]
        res_df[first_str + '*' + second_str] = all_feature[first_str] * all_feature[second_str]
        res_df[first_str + '/' + second_str] = (all_feature[first_str] + 0.1) / (all_feature[second_str] + 0.1)
        res_df[second_str + '/' + first_str] = (all_feature[second_str] + 0.1) / (all_feature[first_str] + 0.1)
    # print('get_cross_feature', res_df)
    return res_df