import os
import numpy as np
import glob
import echo_canc_lib as ec
import h5py

path_label = "./data/multi-label/"
path_mfcc = "./data/multi-mfcc/"
path_feature = "./data/multi-feature/"

files = glob.glob(path_feature + 'train/*')
for file_ in files:
    os.remove(file_)

files = glob.glob(path_feature + 'test/*')
for file_ in files:
    os.remove(file_)

labels = ['econom', 'financ', 'movie', 'music', 'news', 'resume', 'scien',
          'sport', 'stop', 'world', 'us']
count = {}
for label in labels:
    count[label] = 0

left_context = 3
right_context = 3
feature_train = np.empty((0, 39*(left_context + right_context + 1)))
feature_test = np.empty((0, 39*(left_context + right_context + 1)))
label_train = np.empty((0, 1))
label_test = np.empty((0, 1))
id_test = np.empty((0, 1))
filename_test = []
filename_list = []
h_train = h5py.File(path_feature + 'train/train.hdf5', 'w')
h_test = h5py.File(path_feature + 'test/test.hdf5', 'w')
total_frame_prev = 0
with open(path_label + 'multi-label_refined', 'r') as f:
    k = 1
    prev_file = '__'
    prev_label = []
    prev_feat = []
    flag_limit = 0
    label_count = np.zeros(len(labels))
    for line in f:
        line_split = line.split(',')
        filename_temp = line_split[0]
        filename_temp = filename_temp.split('/')
        s = '--'
        filename_temp = s.join(filename_temp)[:-5]
        filename = filename_temp[1:] + "_lab"
        print(filename, filename_temp)
        if prev_file != filename and filename not in filename_list:
            k += 1

        if np.mod(k, 10) == 0:
            test = 1
            train = 0
        else:
            test = 0
            train = 1

        print('processing: ', filename, '...')
        print(line_split)
        keyword = line_split[1]

        start_frame = line_split[2]
        end_frame = line_split[3]
        total_frame = line_split[4]

        with open(path_mfcc + filename, 'r') as f:
            lines = f.readlines()
            rows = [line.split() for line in lines]
        rows = np.asarray(rows)
        mfcc_ = rows.astype(np.float)
        num_data, num_feat = np.shape(mfcc_)
        num_data = int(num_data)
        total_frame = int(total_frame)

        if num_data != total_frame:
            total_frame = num_data

        if num_data != total_frame:
            print('something is wrong!!!')
            break

        feature_temp = ec.get_feature_new(mfcc_, left_context, right_context)

        if train:

            if prev_file == filename and total_frame_prev == total_frame:
                label_temp, label_prev = ec.get_label_multi_prev(keyword, start_frame, end_frame, right_context,
                                                                 prev_label)
                prev_label = label_prev
                with h5py.File(path_feature + 'train/train.hdf5', 'r+') as f1:
                    data = f1[filename + '_label']
                    data[...] = label_temp

                with h5py.File(path_feature + 'train/train.hdf5', 'r+') as f2:
                    data = f2[filename + '_feature']
                    data[...] = feature_temp

            elif prev_file != filename and filename in filename_list:
                continue
                # filename_temp2 = filename + "_2"
                #
                # label_temp, label_prev = ec.get_label_multi(keyword, start_frame, end_frame, total_frame, right_context)
                # prev_label = label_prev
                # dset_train = h_train.create_dataset(filename_temp2 + '_label', data=label_temp)
                # dset_train = h_train.create_dataset(filename_temp2 + '_feature', data=feature_temp)

            else:
                label_temp, label_prev = ec.get_label_multi(keyword, start_frame, end_frame, total_frame, right_context)
                prev_label = label_prev
                dset_train = h_train.create_dataset(filename + '_label', data=label_temp)
                dset_train = h_train.create_dataset(filename + '_feature', data=feature_temp)

        else:

            if prev_file == filename and total_frame_prev == total_frame:
                label_temp, label_prev = ec.get_label_multi_prev(keyword, start_frame, end_frame, right_context,
                                                                 prev_label)
                prev_label = label_prev
                with h5py.File(path_feature + 'test/test.hdf5', 'r+') as f3:
                    data = f3[filename + '_label']
                    data[...] = label_temp

                with h5py.File(path_feature + 'test/test.hdf5', 'r+') as f4:
                    data = f4[filename + '_feature']
                    data[...] = feature_temp

            elif prev_file != filename and filename in filename_list:
                continue

            else:
                label_temp, label_prev = ec.get_label_multi(keyword, start_frame, end_frame, total_frame, right_context)
                prev_label = label_prev
                dset_test = h_test.create_dataset(filename + '_label', data=label_temp)
                dset_test = h_test.create_dataset(filename + '_feature', data=feature_temp)

        filename_list = np.append(filename_list, filename)
        prev_file = filename
        total_frame_prev = total_frame
h_train.close()
h_test.close()
print(count)
        # if k >= 24:
        #     break

        # with open(path_feature + filename + '_feature') as fx:
        #     lines = fx.readlines()
        #     rows = [line.split() for line in lines]
        # rows = np.asarray(rows)
        # zz = rows.astype(np.float)
        # print(zz.shape)