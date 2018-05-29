def drop_null_str(string):
    """
    去掉 list（内部为str）里的空str（''）
    :param string:
    :return:
    """
    str_list = string.split(',')
    new_list = []
#     print(str_list)
    for i in str_list:
        # if i != '' and i != 'android':
        if i != '':
            new_list.append(i)
    return ' '.join(new_list)


def drop_nan(df, app_name):
    """
    去空
    :param df:
    :param app_name:applist 的字段名
    :return:
    """
    df = df.dropna(axis=0, how='any')
    df[app_name] = df[app_name].apply(drop_null_str)
    return df
