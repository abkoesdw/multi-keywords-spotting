import numpy as np
import h5py

# labels = ['econom', 'financ', 'movie', 'music', 'news', 'resume', 'scien',
#           'sport', 'stop', 'world', 'us']


def find_start_end(line_temp, line):
    start_file = 0
    end_file = 0
    filename = []
    if len(line_temp) == 1 and line != ".":
        filename_temp = line.split("/")[-1]
        filename = filename_temp.split(".")[0]
        start_file = 1
        end_file = 0
        print(filename)
    elif len(line_temp) == 1 and line == ".":
        end_file = 1
        start_file = 0

    return filename, start_file, end_file


def find_kw(line_temp):
    labels = ['econom', 'financ', 'movie', 'music', 'news', 'resume', 'scien',
              'sport', 'stop', 'world', 'us']
    keyword = []
    kw_found = 0
    for label2 in labels:
        if label2 in line_temp[-1][0:len(label2)]:
            if label2 == "us":
                if line_temp[-1][:-1] == "us":
                    keyword = "us"
                    print(keyword)
                    kw_found = 1
                    break
            else:
                keyword = line_temp[-1][:-1]
                print(keyword)
                kw_found = 1
                break

    return keyword, kw_found


def generator_train():
    path_feature = "./data/multi-feature/"
    while 1:
        # following loads data from HDF5 file numbered with fileIndex
        x = h5py.File(path_feature + "train/train.hdf5", "r")
        keys = x.keys()
        for key in keys:
            print(x[key])



def generator_test():
    x = 1
    return x




