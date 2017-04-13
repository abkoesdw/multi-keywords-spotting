import bcolz
import numpy as np
import bcolzarrayiterator as bci

Y_train = bcolz.open("./data/multi-feature/train_bcolz/label_train.bc", mode='r')
X_train = bcolz.open("./data/multi-feature/train_bcolz/feature_train.bc", mode='r')
print(Y_train)
# atches = bci.BcolzArrayIterator(X, Y, batch_size=X.chunklen * 10, shuffle=True)



