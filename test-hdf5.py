import numpy as np
import h5py
from keras.utils.io_utils import HDF5Matrix

d1 = 10
d2 = 100

h = h5py.File('./data/data.hdf5', 'w')
dset = h.create_dataset('data1', data=d1)
dset = h.create_dataset('data2', data=d2)
# dset = h.create_dataset('data', data=d2)
# print(dset)
h.close()

with h5py.File('./data/data.hdf5', 'r') as f:
    for file in f:
        X_data = f[file]
        print(X_data.value)
    # model.predict(X_data)


with h5py.File('./data/data.hdf5', 'r+') as f:
    for file in f:
        X_data = f[file]
        X_data[...] = 1000
        print(X_data.value)
    # model.predict(X_data)

with h5py.File('./data/data.hdf5', 'r') as f:
    for file in f:
        X_data = f[file]
        print(X_data.value)
    # model.predict(X_data)
