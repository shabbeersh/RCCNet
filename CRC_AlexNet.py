from __future__ import print_function
import numpy as np
import os
import time,keras
from keras.callbacks import CSVLogger
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, ZeroPadding2D,MaxPooling2D, BatchNormalization,Input
from keras.models import Model
from keras.models import Sequential
from keras.callbacks import ReduceLROnPlateau, CSVLogger,EarlyStopping,ModelCheckpoint
from keras.utils import np_utils
import re
from sklearn.cross_validation import train_test_split
from sklearn.utils import shuffle
import h5py
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

PATH = "/home/shabbeer/CV_Course/Shabbeer/crchistophenotypes32_32"
print("PWD", PATH)

from keras import backend as K

def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(data, key=alphanum_key)


# Define data path
data_path = PATH
data_dir_list = sorted_alphanumeric(os.listdir(data_path))
print(data_dir_list)

img_data_list = []

for dataset in sorted_alphanumeric(data_dir_list):
    img_list = sorted_alphanumeric(os.listdir(data_path + '/' + dataset))
    print('Loaded the images of dataset-' + '{}\n'.format(dataset))
    for img in img_list:
        print(img)
        img_path = data_path + '/' + dataset + '/' + img
        img = image.load_img(img_path, target_size=(33,33))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        #     x = x/255
        print('Input image shape:', x.shape)
        img_data_list.append(x)

img_data = np.array(img_data_list)
# img_data = img_data.astype('float32')
print(img_data.shape)
img_data = np.rollaxis(img_data, 1, 0)
print(img_data.shape)
img_data = img_data[0]
print(img_data.shape)

num_classes = 4
num_of_samples = img_data.shape[0]
print("sample", num_of_samples)
labels = np.ones((num_of_samples,), dtype='int64')
labels[0:7722] = 0
labels[7722:13434] = 1
labels[13434:20225] = 2
labels[20225:] = 3
names = ['epithelial', 'fibroblast', 'inflammatory', 'others']

Y = np_utils.to_categorical(labels, num_classes)
x, y = shuffle(img_data, Y, random_state=2)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)

model = Sequential()
model.add(ZeroPadding2D(1, input_shape=(33,33,3)))
model.add(Conv2D(96, (5,5), activation='relu'))
model.add(BatchNormalization())

model.add(Conv2D(256, (5, 5), activation='relu'))
model.add(BatchNormalization())

model.add(Dropout(0.3, name='dropout_1'))
model.add(MaxPooling2D(pool_size=(2, 2), name='block1_maxpooling2'))
model.add(Conv2D(384, (3, 3), activation='relu', name='block1_conv3', padding='same'))

model.add(Conv2D(384, (3, 3), activation='relu', name='block1_conv4', padding='same'))
model.add(Conv2D(256, (3, 3), activation='relu', name='block1_conv5', padding='same'))
model.add(Dropout(0.4, name='dropout_2'))
model.add(Flatten(name='flatten'))
# model.add(Dense(512, activation='relu', name='fc'))
# model.add(Dropout(0.5, name='dropout_3'))
model.add(Dense(4096, activation='relu', name='fc2'))
model.add(Dropout(0.5, name='dropout_4'))
model.add(Dense(4096, activation='relu', name='fc1'))
model.add(Dropout(0.5, name='dropout_5'))
model.add(Dense(4, activation='softmax', name='predictions'))
model.summary()

batch_size = 32
num_classes = 4
epochs = 500
data_augmentation = False

lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=2, min_lr=0.5e-6)
csv_logger = CSVLogger('/home/ravi/Desktop/Shabbeer/AlexNet_acc_DA.csv')

# save_dir = os.path.join(os.getcwd(), 'saved_models')
adam = Adam(lr=0.00006, beta_1=0.9, beta_2=0.99, epsilon=None, decay=1e-6, amsgrad=False)

# Let's train the model using Adam
model.compile(loss='categorical_crossentropy',
              optimizer=adam,
              metrics=['accuracy'])

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

import time
start = time.time()

if not data_augmentation:
    print('Not using data augmentation.')
    history = model.fit(X_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(X_test, y_test),
              shuffle=True, callbacks=[lr_reducer,csv_logger])
else:
    print('Using real-time data augmentation.')
    datagen = ImageDataGenerator(
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True)  # randomly flip images

    datagen.fit(X_train)
    history = model.fit_generator(datagen.flow(X_train, y_train,
                                            batch_size=batch_size),
                               epochs=epochs,
                               validation_data=(X_test, y_test), callbacks=[lr_reducer, csv_logger])

print('------------Training time is seconds:%s',time.time()-start)

scores = model.evaluate(X_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
import matplotlib.pyplot as plt
print('Max test accuracy:', max(history.history['val_acc']))
print(history.history.keys())
# # summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
