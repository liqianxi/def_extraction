import os

def load_data(path):
    with open(path,'r') as obj:
       raw_str_list = [i.replace('"','').split('\t') for i in obj.read().splitlines()]
       return [(int(i[1]), i[0]) for i in raw_str_list]

def dataset_convert(path):
    data_set = []

    for filename in os.listdir(path):
        f = os.path.join(path, filename)
        # checking if it is a file
        if os.path.isfile(f):
            data_set += load_data(f)

    return data_set