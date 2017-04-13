import echo_canc_lib as ec
import os
import numpy as np
import glob

labels = ['econom', 'financ', 'movie', 'music', 'news', 'resume', 'scien',
          'sport', 'stop', 'world', 'us']
labels_list = ['econom', 'financ', 'movie', 'music', 'news', 'resume', 'scien',
               'sport', 'stop', 'world']
path_label = "./data/multi-label/"

path_mfcc = "./data/multi-mfcc/small/"
if not os.path.exists(path_mfcc):
        os.makedirs(path_mfcc)

if not os.path.exists(path_mfcc):
    os.makedirs(path_mfcc)

files = glob.glob(path_mfcc + '*')
for file_ in files:
    os.remove(file_)

for label in labels:
    path_txt = "./data/" + label + '/'
    mfcc_files = glob.glob(path_txt + "*.mfc")

    print(np.asarray(mfcc_files).T)

    with open(path_label + 'multi-label_refined_small') as f:
        labels = f.readlines()
        num_files = len(labels)

    mfcc = np.empty((39, 0))
    file_id = 0
    for mfcc_file in mfcc_files:
        print(mfcc_file)

        with open(mfcc_file) as f:
            filename = '______'
            for line in f:
                x = line.split()
                if x[0] == 'FILE:':
                    # if file_id >= 2000:
                    #     break
                    file_id += 1
                    np.savetxt(path_mfcc + filename, mfcc.T)
                    filename_temp = x[1].split('/')

                    s = "--"
                    filename_temp = s.join(filename_temp)
                    # filename_temp = np.asarray(filename_temp).astype(np.str)

                    filename = filename_temp[:-4] + "_lab"

                    # print(filename)

                    mfcc = np.empty((39, 0))
                else:
                    x = np.asarray(x).astype(np.float)
                    mfcc = np.concatenate((mfcc, x.reshape(39, 1)), axis=1)
        np.savetxt(path_mfcc + filename, mfcc.T)

    os.remove(path_mfcc + '______')
        # if file_id == num_files and i == num_lines - 1:
        #     print('at', str(num_files), filename)

#
