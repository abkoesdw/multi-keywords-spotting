import os
import numpy as np
import glob
import echo_canc_lib as ec

label = 'scien/'
path_label = "./data/" + label + "label/"
path_mfcc = "./data/" + label + "feature/"
path_feature = "./data/" + label + "feature/final/"

left_context = 3
right_context = 3
feature_train = np.empty((0, 39*(left_context + right_context + 1)))
feature_test = np.empty((0, 39*(left_context + right_context + 1)))
label_train = np.empty((0, 1))
label_test = np.empty((0, 1))
id_test = np.empty((0, 1))
filename_test = []
print(path_label + label[0:-1] + '_refined')
with open(path_label + label[0:-1] + '_refined') as f:
    k = 1
    prev_file = '__'
    prev_label = []
    prev_feat = []
    for line in f:
        if np.mod(k, 10) == 0:
            test = 1
            train = 0
        else:
            test = 0
            train = 1

        line_split = line.split(',')
        filename = line_split[0]
        print('processing: ', filename, '...')
        start_frame = {}
        end_frame = {}
        m = 1
        for j in range(int(len(line_split)/2 - 1)):
            start_frame[j] = line_split[m]
            end_frame[j] = line_split[m + 1]
            m += 2

        total_frame = line_split[-1]

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
        label_temp = ec.get_label_new(start_frame, end_frame, total_frame, left_context, right_context)

        if train:
            feature_train = np.concatenate((feature_train, feature_temp), axis=0)
            label_train = np.concatenate((label_train, label_temp), axis=0)
        elif test:
            feature_test = np.concatenate((feature_test, feature_temp), axis=0)
            label_test = np.concatenate((label_test, label_temp), axis=0)
            id_test_temp = np.zeros((num_data, 1))
            id_test_temp = id_test_temp[4:-3]
            id_test_temp[0] = 1
            id_test_temp[-1] = -1
            id_test = np.concatenate((id_test, id_test_temp), axis=0)
            filename_test = np.append(filename_test, filename)

        k += 1
        if k >= 2000:
            break

    if not os.path.exists(path_feature):
        os.makedirs(path_feature)

    np.savez(path_feature + label[0:-1] + '.npz', feature_train=feature_train, label_train=label_train,
             feature_test=feature_test, label_test=label_test, id_test=id_test,
             filename_test=filename_test)

