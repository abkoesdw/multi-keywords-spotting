import numpy as np
import glob
import os

labels_list_wo_us = ['econom', 'financ', 'movie', 'music', 'news', 'resume', 'scien',
                     'sport', 'stop', 'world']
labels_list_us = ['econom', 'financ', 'movie', 'music', 'news', 'resume', 'scien',
                  'sport', 'stop', 'world', 'us']
path_label = "./data/multi-label/"
label_filename_temp = 'multi-label_refined_temp'
prev_fname = '----'
prev_start = 0
prev_end = 0
prev_total = 0
prev_kwrds = 0
flag_double = 0

with open(path_label + label_filename_temp, 'wb') as f2:
    print(path_label + label_filename_temp)
    with open(path_label + 'multi-label') as f:
        k = 0
        count_eq = 0
        for line in f:
            line_split = line.split(',')
            filename_temp = line_split[0]
            filename = filename_temp.split('_')[:-1]
            s = '_'
            filename = s.join(filename)
            kwrds = line_split[1]
            start = line_split[2]
            end = line_split[3]
            total = line_split[4][:-1]
            # print(line_split[-1][:-2])

            if filename == prev_fname:
                prev_label_info = np.array([filename, prev_kwrds, prev_start, prev_end,
                                            kwrds, start, end, total]).reshape(1, 8)
                count_eq += 1

            if filename != prev_fname and k == 0:
                prev_label_info = np.array([filename, kwrds, start, end, total]).reshape(1, 5)

            if filename != prev_fname and k != 0:
                np.savetxt(f2, prev_label_info, delimiter=',', fmt='%s')
                prev_label_info = np.array([filename, kwrds, start, end, total]).reshape(1, 5)
                flag_double = 0

            prev_fname = filename
            prev_kwrds = kwrds
            prev_start = start
            prev_end = end
            prev_total = total

            k += 1
        np.savetxt(f2, prev_label_info, delimiter=',', fmt='%s')

label_filename = 'multi-label_refined'
filename_list = []
with open(path_label + label_filename, 'wb') as f3:
    with open(path_label + label_filename_temp, 'r') as f4:
        for line in f4:
            line_ = line.split(',')
            if line_[0] in filename_list:
                # print(line_[0])
                continue
            else:
                line_[-1] = line_[-1][:-1]
                line_temp = np.asarray(line_).astype(np.str).reshape(1, len(line_))
                np.savetxt(f3, line_temp, delimiter=',', fmt='%s')
                filename_list = np.append(filename_list, line_[0])

count = dict()
count['us'] = 0
flagx = 0
for label in labels_list_wo_us:
    count[label] = 0

with open(path_label + label_filename, 'r') as f3:
    for line in f3:
        # print(line)
        line_ = line.split(',')
        if len(line_) > 5:
            # print(line_)
            # print(int(len(line_)/3) + 1)
            flagx = 1
        else:
            flagx = 0

        for k in range(1, len(line_), 3):
            for label in labels_list_wo_us:
                if label in line_[k]:
                    # if flagx:
                        # print(line_[k])
                    count[label] += 1
        if 'us' in line_:
            count['us'] += 1

for label in labels_list_us:
    print(label, ':', count[label])

