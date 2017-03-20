import numpy as np
import glob
import os

label = 'world/'
path_label = "./data/" + label + "label/"

label_filename = label[:-1] + '_refined'
prev_fname = '----'
prev_start = 0
prev_end = 0
prev_total = 0
flag_double = 0
with open(path_label + label_filename, 'wb') as f2:
    print(path_label + label_filename)
    with open(path_label + label[:-1]) as f:
        k = 0
        for line in f:
            line_split = line.split(',')
            filename = line_split[0]
            start = line_split[1]
            end = line_split[2]
            total = line_split[3][:-1]
            # print(line_split[-1][:-2])

            if filename == prev_fname:
                prev_label_info = np.array([filename, prev_start, prev_end, start, end, total]).reshape(1, 6)

            if filename != prev_fname and k == 0:
                prev_label_info = np.array([filename, start, end, total]).reshape(1, 4)

            if filename != prev_fname and k != 0:
                np.savetxt(f2, prev_label_info, delimiter=',', fmt='%s')
                prev_label_info = np.array([filename, start, end, total]).reshape(1, 4)
                flag_double = 0

            prev_fname = filename
            prev_start = start
            prev_end = end
            prev_total = total

            k += 1
        np.savetxt(f2, prev_label_info, delimiter=',', fmt='%s')