import os
import numpy as np
import glob
import echo_canc_lib as ec

label = 'econom/'
path_txt = "./data/" + label
path_label = "./data/" + label + "label/"
path_mfcc = "./data/" + label + "feature/"

# path_label = "/home/abkoesdw/Documents/cognitech/keyword-project/"
# path_mfcc = "/home/abkoesdw/Documents/cognitech/keyword-project/mfcc_new/"
# path_feature = "/home/abkoesdw/Documents/cognitech/keyword-project/test-new/"

left_context = 3
right_context = 3
feature_train = np.empty((0, 39*(left_context + right_context + 1)))
feature_test = np.empty((0, 39*(left_context + right_context + 1)))
label_train = np.empty((0, 1))
label_test = np.empty((0, 1))
id_test = np.empty((0, 1))
filename_test = []

with open(path_label + label[0:-1]) as f:
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

        filename = line.split(',')[0]
        print('processing: ', filename, '...')
        start_frame = line.split(',')[1]
        end_frame = line.split(',')[2]
        total_frame = line.split(',')[3]
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
        filename_temp = filename.split('_')
        filename_orig = filename_temp[0:-1]

        if filename_orig != prev_file:
            feature_temp = ec.get_feature_new(mfcc_, left_context, right_context)
            label_temp = ec.get_label_new(start_frame, end_frame, total_frame, left_context, right_context)
        else:
            feature_temp = prev_feat
            label_temp = ec.get_label_new2(prev_label, start_frame, end_frame)
            flag_double = 1
        prev_feat = feature_temp
        prev_label = label_temp
        prev_file = filename_orig
        print(feature_temp.shape, label_temp.shape)
        if train:
            if not flag_double:
                feature_train = np.concatenate((feature_train, feature_temp), axis=0)
                label_train = np.concatenate((label_train, label_temp), axis=0)
            else:
                feature_train = prev_feat
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

    np.savez(path_mfcc + label[0:-1] + '_features.npz', feature_train=feature_train, label_train=label_train,
             feature_test=feature_test, label_test=label_test, id_test=id_test,
             filename_test=filename_test)
# print('total files:', k-1)
# print(feature_train.shape, label_train.shape)
# print(feature_test.shape, label_test.shape)
# print(id_test.shape, id_test[0], id_test[-1])
# print(filename_test)