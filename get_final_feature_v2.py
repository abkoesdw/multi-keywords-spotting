import os
import numpy as np
import glob
import echo_canc_lib as ec
import h5py
import bcolz
import shutil
from keras.utils.np_utils import to_categorical
path_label = "./data/multi-label/"
path_mfcc = "./data/multi-mfcc/"
path_feature = "./data/multi-feature/"

# files = glob.glob(path_feature + 'train/*')
# for file_ in files:
#     os.remove(file_)
#
# files = glob.glob(path_feature + 'test/*')
# for file_ in files:
#     os.remove(file_)

labels = ['econom', 'financ', 'movie', 'music', 'news', 'resume', 'scien',
          'sport', 'stop', 'world', 'us']

left_context = 3
right_context = 3

# h_train = h5py.File(path_feature + 'train/train.hdf5', 'w')
# h_test = h5py.File(path_feature + 'test/test.hdf5', 'w')
if os.path.exists("./data/multi-feature/train_bcolz/label_train.bc"):
    shutil.rmtree("./data/multi-feature/train_bcolz/label_train.bc")
if os.path.exists("./data/multi-feature/train_bcolz/feature_train.bc"):
    shutil.rmtree("./data/multi-feature/train_bcolz/feature_train.bc")
if os.path.exists("./data/multi-feature/test_bcolz/label_test.bc"):
    shutil.rmtree("./data/multi-feature/test_bcolz/label_test.bc")
if os.path.exists("./data/multi-feature/test_bcolz/feature_test.bc"):
    shutil.rmtree("./data/multi-feature/test_bcolz/feature_test.bc")
num_label = 0
num_feature = 0
with open(path_label + 'multi-label_refined', 'r') as f:
    k = 1
    m = 0
    for line in f:
        line_split = line.split(',')
        filename_temp = line_split[0]
        filename_temp = filename_temp.split('/')
        s = '--'
        filename_temp = s.join(filename_temp)[:-5]
        filename = filename_temp[1:] + "_lab"

        if np.mod(k, 10) == 0:
            test = 1
            train = 0
        else:
            test = 0
            train = 1

        # print('processing: ', filename, '...')

        total_frame = str(line_split[-1][:-1])

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
            print(num_data, total_frame)
            break

        feature_temp = ec.get_feature_new(mfcc_, left_context, right_context)
        label_temp = ec.get_label_multi(line_split, total_frame, right_context)
        # label_temp = to_categorical(label_temp, num_classes=12)
        num_data1, _ = feature_temp.shape
        num_data2 = label_temp.shape
        num_label += num_data2[0]
        num_feature += num_data1
        # print(feature_temp.shape, label_temp.shape)
        if num_data1 != num_data2[0]:
            print('found something wrong')
            print(num_data1, num_label)
            print(feature_temp.shape, label_temp.shape)
            break
        if train:
            if k == 1:
                h1 = bcolz.carray(label_temp, rootdir="./data/multi-feature/train_bcolz/label_train.bc")
                h2 = bcolz.carray(feature_temp, rootdir="./data/multi-feature/train_bcolz/feature_train.bc")
                h1.flush()
                h2.flush()
            else:
                h1.append(label_temp)
                h1.flush()
                h2.append(feature_temp)
                h2.flush()
            # dset_train = h_train.create_dataset(filename + '_label', data=label_temp)
            # dset_train = h_train.create_dataset(filename + '_feature', data=feature_temp)

        else:
            # dset_test = h_test.create_dataset(filename + '_label', data=label_temp)
            # dset_test = h_test.create_dataset(filename + '_feature', data=feature_temp)
            if m == 0:
                h3 = bcolz.carray(label_temp, rootdir="./data/multi-feature/test_bcolz/label_test.bc")
                h4 = bcolz.carray(feature_temp, rootdir="./data/multi-feature/test_bcolz/feature_test.bc")
                h3.flush()
                h4.flush()
            else:
                h3.append(label_temp)
                h3.flush()
                h4.append(feature_temp)
                h4.flush()
            m += 1

        k += 1
        # if k >= 100:
        #     break
# h_train.close()
# h_test.close()
print(num_label, num_feature)

