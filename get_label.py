import echo_canc_lib as ec
import os
import numpy as np
import glob

path_txt = "/home/abkoesdw/Documents/cognitech/keyword-project/mfctxt-files-arief/"
path_label = "/home/abkoesdw/Documents/cognitech/keyword-project/"
path_mfcc = "/home/abkoesdw/Documents/cognitech/keyword-project/mfcc_new/"

if not os.path.exists(path_mfcc):
    os.makedirs(path_mfcc)

files = glob.glob(path_mfcc + "*.npz")
for file_ in files:
    os.remove(file_)

with open(path_label + 'trigger.mlf') as f:
    num_files = len(f.readlines())

label_data = open(path_label + 'trigger.mlf', 'r')

k = 1
flag_system = 0
flag_frame = 0
file_id = 0
with open(path_label + 'label_file', 'wb') as label_file:
    for line_ in label_data:
        line_temp = line_
        line_ = line_.split()
        line_ = np.asarray(line_)
        # print(line_)
        if line_temp[1:5] == 'data':
            file_id += 1
            filename_ = line_temp
            filename_ = filename_.split('/')[-1:][0]
            filename_ = filename_[:-6]
            # filename_ = np.asarray(filename_).astype(np.str)
            #
            filename_ = filename_ + '_' + str(file_id)
            print("file #", str(k), ", name: ", filename_)

        if "system" in line_:
            line__ = line_
            start_frame = np.int(line__[0])
            flag_system = 1
            line_prev = line__
            continue

        if flag_system:
            line_now = line_
            if len(line_now) == 4:
                line_prev2 = line_prev
                line_prev = line_now
            else:
                line_now = line_prev2
                print(np.int(round(np.float(line_now[1]))))
                end_frame = np.int(round(np.float(line_now[1])))
                flag_system = 0
                print(end_frame)

        if len(line_) == 1 and 'data' not in line_temp:
            file_end = file_prev
            print("file end", file_end[1])
            frame_length = file_end[1].astype(np.float)
            frame_length = int(round(frame_length))
            flag_frame = 1
        else:
            file_prev = line_

        if flag_frame:
            label_info = np.array([filename_, start_frame, end_frame, frame_length])
            label_info = label_info.reshape(1, 4)
            np.savetxt(label_file, label_info, delimiter=',', fmt='%s')
            flag_frame = 0
print(file_id)