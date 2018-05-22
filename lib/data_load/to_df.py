import pandas as pd


def txt_to_df(file_name, header_list):
    """

    :param file_name:str
    :param header_list: list,元素为 str
    :return: df
    """
    df = pd.read_csv(file_name, sep='\t', encoding='utf-8', names=header_list)
    return df


def csv_to_df(file_name, header_list):
    """

    :param file_name:str
    :param header_list: list,元素为 str
    :return: df
    """
    df = pd.read_csv(file_name)
    return df