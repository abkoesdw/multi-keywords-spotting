import numpy as np
import glob
import os

label = 'econom/'
path_label = "./data/" + label + "label/"

label_filename = label[:-1] + '_refined'
prev_fname = '----'
prev_start = 0
prev_end = 0
prev_total = 0
flag_double = 0
with open(path_label + label_filename, 'wb') as f2:
    with open(path_label + label[:-1]) as f:
        k = 0
        for line in f:
            line_split = line.split(',')
            filename = line_split[0]
            start = line_split[1]
            end = line_split[2]
            total = line_split[3]

            if filename == prev_fname:
                prev_label_info = np.array([filename, prev_start, prev_end, start, end, total]).reshape(1, 6)
                flag_double = 1

            if filename != prev_fname and k != 0:
                np.savetxt(f2, prev_label_info, delimiter=',', fmt='%s')
                prev_label_info = np.array([filename, start, end, total]).reshape(1, 4)

            prev_fname = filename
            prev_start = start
            prev_end = end
            prev_total = total

            k += 1
            if k == len(f):
                print(len(f))

# file_id = 0
# with open(path_label + label_class, 'wb') as label_filename:
#     for label_file in label_files:
#         label_data = open(label_file, 'r')
#         flag_kw = 0
#         flag_frame = 0
#         label_info_temp = np.empty((0, 3))
#
#         label_name = path_txt.split('/')[-2]
#         for line_ in label_data:
#             line_temp = line_
#             line_ = line_.split()
#             line_ = np.asarray(line_)
#
#             if line_temp[1:5] == 'data':
#                 if file_id >= 2000:
#                     break
#                 file_id += 1
#                 filename_ = line_temp
#                 filename_ = filename_.split('/')[-1:][0]
#                 filename_ = filename_[:-6]
#                 filename_ = filename_ + '_' + str(file_id)
#                 print("filename: ", filename_)
#
#             if len(line_) == 5:
#                 word = line_[-1]
#                 if label_name in word:
#                     line__ = line_
#                     start_frame = np.int(line__[0])
#                     print("keyword: ", word)
#                     print("start frame: ", start_frame)
#                     flag_kw = 1
#                     line_prev = line__
#                     continue
#
#             if flag_kw:
#                 line_now = line_
#                 if len(line_now) == 4:
#                     line_prev2 = line_prev
#                     line_prev = line_now
#                 else:
#                     line_now = line_prev2
#                     # print(np.int(round(np.float(line_now[1]))))
#                     end_frame = np.int(round(np.float(line_now[1])))
#                     print("end frame: ", end_frame)
#                     print(np.array([filename_, start_frame, end_frame]).reshape(1, 3).shape)
#                     label_info_temp = np.concatenate((label_info_temp,
#                                                       np.array([filename_, start_frame,
#                                                                 end_frame]).reshape(1, 3)), axis=0)
#                     flag_kw = 0
#
#             if len(line_) == 1 and 'data' not in line_temp:
#                 file_end = file_prev
#                 print("total frame: ", file_end[1])
#                 frame_length = file_end[1].astype(np.float)
#                 frame_length = int(round(frame_length))
#                 flag_frame = 1
#
#             else:
#                 file_prev = line_
#
#             if flag_frame:
#                 num_kw, _ = np.shape(label_info_temp)
#                 label_info = np.concatenate((label_info_temp, np.tile(frame_length, (num_kw, 1))), axis=1)
#                 np.savetxt(label_filename, label_info, delimiter=',', fmt='%s')
#                 flag_frame = 0
#                 label_info_temp = np.empty((0, 3))
