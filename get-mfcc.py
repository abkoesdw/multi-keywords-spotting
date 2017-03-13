import echo_canc_lib as ec
import os
import numpy as np
import glob

path_txt = "./data/econom/"
path_label = "./data/econom/"
path_mfcc = "./data/econom/feature/"

if not os.path.exists(path_mfcc):
    os.makedirs(path_mfcc)

files = glob.glob(path_mfcc + '*')
for file_ in files:
    os.remove(file_)

with open(path_label + 'label_file') as f:
    labels = f.readlines()

with open(path_txt + 'trigger.mfc') as f:
    data = f.readlines()
data = np.asarray(data)
flag_file = 0
count = 0
mfcc = np.empty((39, 0))
filename = '______'
i = 0
file_id = 0
while i < len(data):
    x = data[i].split()
    if x[0] == 'FILE:':
        file_id += 1
        prev_fn = filename
        np.savetxt(path_mfcc + filename, mfcc.T)
        filename_temp = x[1].split('/')[-1:]
        filename_temp = np.asarray(filename_temp).astype(np.str)

        filename = filename_temp[0][:-4] + '_' + str(file_id)

        print(filename)

        mfcc = np.empty((39, 0))
    else:
        x = np.asarray(x).astype(np.float)
        mfcc = np.concatenate((mfcc, x.reshape(39, 1)), axis=1)

    if file_id == 2379 and i == len(data) - 1:
        print('at 2379', filename)
        np.savetxt(path_mfcc + filename, mfcc.T)
    i += 1

print(file_id)
