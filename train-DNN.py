import numpy as np
import os
from keras.models import Sequential
from keras.layers import Dense, Dropout
import time
from sklearn.metrics import confusion_matrix, classification_report
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import SGD
from keras.utils.np_utils import to_categorical

unused_var = os.system('clear')
today_ = time.strftime("%Y-%m-%d")

label = 'movie/'
path_feature = "./data/" + label + "feature/final/"
path_model = "./model/" + label
if not os.path.exists(path_model):
    os.makedirs(path_model)
data = np.load(path_feature + label[0:-1] + '.npz')

x_train = data['feature_train']
y_train = data['label_train']
y_train = to_categorical(np.int16(y_train), nb_classes=2)

x_test = data['feature_test']
y_test = data['label_test']
y_test = to_categorical(np.int16(y_test), nb_classes=2)

_, num_feat = np.shape(x_train)
init = 'uniform'
neurons = 512*3
drop_out = 0.2
model = Sequential()

model.add(Dense(neurons, input_dim=num_feat, init=init, activation='relu'))
model.add(Dropout(drop_out))

num_hid_layer = 3
# 512x3
for i in range(num_hid_layer-1):
    model.add(Dense(neurons, init=init, activation='relu'))
    model.add(Dropout(drop_out))

model.add(Dense(2, init=init, activation='softmax'))

# compile the model
sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
# model.summary()
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# train the model
nb_epoch = 200
batch_size = 1000
callbacks = EarlyStopping(monitor='val_loss', min_delta=0.0, patience=6, verbose=0, mode='auto')
filepath = (path_model + "model.best.hdf5")
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
model.fit(x_train, y_train, validation_data=(x_test, y_test), nb_epoch=nb_epoch, batch_size=batch_size, shuffle=True,
          verbose=2, callbacks=[callbacks, checkpoint])

# save the model
model.load_weights(filepath)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
model.save(path_model + 'model_' + str(neurons) + 'x' + str(num_hid_layer) + '.h5')


model_json = model.to_json()
with open(path_model + "model_" + str(neurons) + 'x' + str(num_hid_layer) + ".json", "w") as json_file:
    json_file.write(model_json)

# prediction
y_est_train = model.predict(x_train, batch_size=batch_size)
y_est_train = np.argmax(y_est_train, axis=1)
y_train = np.argmax(y_train, axis=1)
accuracy_train = float(len(np.where(y_est_train == y_train)[0]))/len(y_train) * 100

print("\n Training accuracy:", accuracy_train, "%")
print("Classification report - Training")
print("-----------------------------------------------------")
print(classification_report(y_train, y_est_train, target_names=['filler', label[:-1]]))
print("-----------------------------------------------------")
print("Confusion matrix - Training")
print("--------------------------")
print(confusion_matrix(y_train, y_est_train, labels=[0, 1]))
print("--------------------------")

# evaluate the model
y_est_test = model.predict(x_test, batch_size=batch_size)
y_est_test = np.argmax(y_est_test, axis=1)
y_test = np.argmax(y_test, axis=1)
accuracy_test = float(len(np.where(y_est_test == y_test)[0]))/len(y_test) * 100
print("\n Testing accuracy:", accuracy_test, "%")

print("Classification report - Testing")
print("-----------------------------------------------------")
print(classification_report(y_test, y_est_test, target_names=['filler', label[:-1]]))
print("-----------------------------------------------------")
print("Confusion matrix - Testing")
print("--------------------------")
print(confusion_matrix(y_test, y_est_test, labels=[0, 1]))
print("--------------------------")

