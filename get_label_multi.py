import numpy as np
import glob
import os

labels = ['econom', 'financ', 'movie', 'music', 'news', 'resume', 'scien',
          'sport', 'stop', 'world', 'us']
labels_list = ['econom', 'financ', 'movie', 'music', 'news', 'resume', 'scien',
               'sport', 'stop', 'world']

path_label = "./data/multi-label/"
if not os.path.exists(path_label):
        os.makedirs(path_label)

with open(path_label + 'multi-label', 'wb') as label_filename:
    file_id = 0
    keyword = 'X'
    prev_word = 'X'
    word_now = 'X'
    for label in labels:
        print(label)
        path_txt = "./data/" + label + '/'
        label_files = glob.glob(path_txt + "*.mfl")
        label_class = path_txt.split('/')[-2]
        num_files = 0
        for label_file in label_files:
            label_data = open(label_file, 'r')
            flag_kw = 0
            flag_kw_us = 0
            flag_frame = 0
            label_info_temp = np.empty((0, 4))

            label_name = path_txt.split('/')[-2]
            for line_ in label_data:
                line_temp = line_
                line_ = line_.split()
                line_ = np.asarray(line_)

                if line_temp[1:5] == 'data':
                    # file_id += 1
                    filename_ = line_temp
                    # filename_ = filename_.split('/')[-1:][0]
                    filename_temp = filename_
                    # filename_ = filename_ + '_' + str(file_id)

                if len(line_) == 5 and not flag_kw:
                    word = line_[-1]

                    for label2 in labels_list:
                        if label2 in word:
                            if word[0] == label2[0]:
                                line__ = line_
                                start_frame = np.int(line__[0])
                                keyword = word
                                flag_kw = 1
                                line_prev = line__
                                continue

                    if word == 'u':
                        line__ = line_
                        prev_word = word
                        start_frame = np.int(line__[0])
                        continue

                    if prev_word == 'u':
                        if word == 's':
                            flag_kw_us = 1
                            flag_kw = 1
                            continue
                        else:
                            prev_word = 'X'
                            flag_kw_us = 0
                            flag_kw = 0
                            continue
                    if flag_kw_us or flag_kw:
                        continue

                if flag_kw:
                    line_now = line_
                    if len(line_now) == 4:
                        line_prev2 = line_prev
                        line_prev = line_now
                    elif len(line_now) == 5:
                        line_now = line_prev2
                        end_frame = np.int(round(np.float(line_now[1])))
                        if flag_kw_us:
                            keyword = 'us'
                            prev_word = 'X'
                            file_id += 1
                            filename_ = filename_temp[0] + '_' + str(file_id)
                            print(filename_)
                            label_info_temp = np.concatenate((label_info_temp,
                                                              np.array([filename_, keyword, start_frame,
                                                                        end_frame]).reshape(1, 4)), axis=0)
                            flag_kw = 0
                            flag_kw_us = 0
                        else:
                            file_id += 1
                            filename_ = filename_ + '_' + str(file_id)
                            label_info_temp = np.concatenate((label_info_temp,
                                                              np.array([filename_, keyword, start_frame,
                                                                        end_frame]).reshape(1, 4)), axis=0)
                            prev_word = 'X'
                            flag_kw = 0

                if len(line_) == 1 and 'data' not in line_temp:
                    file_end = file_prev
                    frame_length = file_end[1].astype(np.float)
                    frame_length = int(round(frame_length))
                    flag_frame = 1

                else:
                    file_prev = line_

                if flag_frame:
                    num_kw, _ = np.shape(label_info_temp)
                    label_info = np.concatenate((label_info_temp, np.tile(frame_length, (num_kw, 1))), axis=1)

                    np.savetxt(label_filename, label_info, delimiter=',', fmt='%s')
                    flag_frame = 0
                    label_info_temp = np.empty((0, 4))
            num_files += 1
            if num_files > 1000:
                break