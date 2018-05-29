def single_kind_by_config_file(app_cate):
    """

    :param app_cate: app类别名
    :return: list of kind.内部元素为 appname_str
    """
    with open('../conf/' + app_cate + '.py') as file:
        lines = file.readlines()
        res_list = []
        for string in lines[:-1]:
            string_new = string[:-1]
            if string[0] != '#':
                res_list.append(string_new)
        if lines[-1][0] != '#':
            res_list.append(lines[-1])
    # print(app_cate, res_list)
    return res_list