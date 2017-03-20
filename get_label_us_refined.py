import numpy as np
import glob
import os

label = 'us/'
path_label = "./data/" + label + "label/"

label_filename = label[:-1] + '_refined'
prev_fname = '----'
prev_start = 0
prev_end = 0
prev_total = 0
prev_kw = 0
flag_double = 0
print(path_label + label_filename)
with open(path_label + label_filename, 'wb') as f2:
    with open(path_label + label[:-1]) as f:
        k = 0
        for line in f:
            line_split = line.split(',')
            filename = line_split[0]
            keyword = line_split[1]
            start = line_split[2]
            end = line_split[3]
            total = line_split[4][:-1]

            if filename == prev_fname:
                prev_label_info = np.array([filename, prev_kw, prev_start, prev_end, keyword, start, end, total]).reshape(1, 8)

            if filename != prev_fname and k == 0:
                prev_label_info = np.array([filename, keyword, start, end, total]).reshape(1, 5)

            if filename != prev_fname and k != 0:
                np.savetxt(f2, prev_label_info, delimiter=',', fmt='%s')
                prev_label_info = np.array([filename, keyword, start, end, total]).reshape(1, 5)
                flag_double = 0

            prev_fname = filename
            prev_kw = keyword
            prev_start = start
            prev_end = end
            prev_total = total

            k += 1
        np.savetxt(f2, prev_label_info, delimiter=',', fmt='%s')