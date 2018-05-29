def get_bad_list():
    with open('bad_list.txt') as file:
        lines = file.readlines()
        res_list = []
        for string in lines[:-1]:
            string_new = string[:-1]
            res_list.append(string_new)
        res_list.append(lines[-1])
        return res_list
print(get_bad_list())