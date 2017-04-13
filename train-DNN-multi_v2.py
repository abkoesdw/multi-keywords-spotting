import numpy as np
import os
from keras.models import Sequential
from keras.layers import Dense, Dropout
import time
from sklearn.metrics import confusion_matrix, classification_report
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from keras.optimizers import SGD
from keras.utils.np_utils import to_categorical
import h5py
import multi_keywords_func as mkf
import bcolz
import bcolzarrayiterator as bci
unused_var = os.system('clear')
today_ = time.strftime("%Y-%m-%d")

path_feature = "./data/multi-feature/"
path_model = "./data/multi-model/"

if not os.path.exists(path_model):
    os.makedirs(path_model)

labels = ['filler', 'econom', 'financ', 'movie', 'music', 'news', 'resume', 'scien',
          'sport', 'stop', 'world', 'us']
Y_train = bcolz.open("./data/multi-feature/train_bcolz/label_train.bc", mode='r')
Y_train = to_categorical(Y_train, num_classes=12)
X_train = bcolz.open("./data/multi-feature/train_bcolz/feature_train.bc", mode='r')
num_train, _ = X_train.shape
Y_test = bcolz.open("./data/multi-feature/test_bcolz/label_test.bc", mode='r')
Y_test = to_categorical(Y_test, num_classes=12)
X_test = bcolz.open("./data/multi-feature/test_bcolz/feature_test.bc", mode='r')

# k = 0
# for label_ in labels:
#     idx_ = np.where(Y_test[:] == k)[0]
#     print("len " + label_, ":", len(idx_))
#     k += 1
# print(np.unique(Y_train), np.unique(Y_test))
num_test, __ = X_test.shape
# batch_size = 1200
batch_size = X_train.chunklen * 1000
batches_train = bci.BcolzArrayIterator(X_train, Y_train, batch_size=batch_size, shuffle=True)
batches_test = bci.BcolzArrayIterator(X_test, Y_test, batch_size=batch_size, shuffle=True)

init = 'uniform'
neurons = 150
drop_out = 0.1
model = Sequential()
model.add(Dense(neurons, input_dim=273, kernel_initializer=init, activation='relu'))
model.add(Dropout(drop_out))

num_hid_layer = 3

# 200x3
for i in range(num_hid_layer-1):
    model.add(Dense(neurons, kernel_initializer=init, activation='relu'))
    model.add(Dropout(drop_out))

model.add(Dense(12, kernel_initializer=init, activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

nb_epoch = 200

stopper = EarlyStopping(monitor='val_loss', min_delta=0.0, patience=6, verbose=0, mode='auto')
#
filepath = (path_model + "model.best.hdf5")
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
# model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=nb_epoch, batch_size=batch_size, shuffle=True,
#           verbose=2, callbacks=[stopper, checkpoint])
model.fit_generator(batches_train, steps_per_epoch=num_train/batch_size, epochs=nb_epoch,
                    verbose=2, callbacks=[stopper, checkpoint],
                    validation_data=batches_test, validation_steps=num_test/batch_size)

model.load_weights(filepath)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
model.save(path_model + 'model_multi' + str(neurons) + 'x' + str(num_hid_layer) + '.h5')

model_json = model.to_json()
with open(path_model + "model_" + str(neurons) + 'x' + str(num_hid_layer) + ".json", "w") as json_file:
    json_file.write(model_json)

# evaluate the model
# y_est_test = model.predict_generator(batches_test, steps=num_test/batch_size)
y_est_test = model.predict(X_test, batch_size=batch_size)
y_est_test = np.argmax(y_est_test, axis=1)
y_test = np.argmax(Y_test, axis=1)
accuracy_test = float(len(np.where(y_est_test == y_test)[0]))/len(y_test) * 100
print("\n Testing accuracy:", accuracy_test, "%")

labels = ['filler', 'econom', 'financ', 'movie', 'music', 'news', 'resume', 'scien',
          'sport', 'stop', 'world', 'us']
print("Classification report - Testing")
print("-----------------------------------------------------")
print(classification_report(y_test, y_est_test, target_names=labels))
print("-----------------------------------------------------")
print("Confusion matrix - Testing")
print("--------------------------")
print(confusion_matrix(y_test, y_est_test, labels=range(12)))
print("--------------------------")

