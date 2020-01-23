import keras
from keras.models import Sequential
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D,AveragePooling2D,GlobalAveragePooling2D
from keras import regularizers, optimizers
import numpy as np
from keras.layers import Add
from keras.layers import Input
from keras.models import Model
from keras.layers import Flatten
from keras.utils import plot_model
from keras.preprocessing import image
import os
import re
from sklearn.cross_validation import train_test_split
from keras.applications.imagenet_utils import preprocess_input
from sklearn.utils import shuffle
from keras import backend as K
from keras.callbacks import ReduceLROnPlateau, CSVLogger,EarlyStopping,ModelCheckpoint
import keras
from keras.preprocessing.image import ImageDataGenerator

PATH = "/home/shabbeer/CV_Course/crchistophenotypes"
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

for dataset in data_dir_list:
    img_list = sorted_alphanumeric(os.listdir(data_path + '/' + dataset))
    print('Loaded the images of dataset-' + '{}\n'.format(dataset))
    for img in img_list:
        print(img)
        img_path = data_path + '/' + dataset + '/' + img
        img = image.load_img(img_path, target_size=(32, 32))
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
labels[0:7721] = 0
labels[7722:13433] = 1
labels[13434:20224] = 2
labels[20225:] = 3
names = ['epithelial', 'fibroblast', 'inflammatory', 'others']

Y = np_utils.to_categorical(labels, num_classes)
x, y = shuffle(img_data, Y, random_state=2)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)

def initial_conv(Input, filters, stride=1, kernel_size=7):
    x = Conv2D(filters, kernel_size=(kernel_size, kernel_size), strides=(stride, stride), padding="same")(Input)

    x = BatchNormalization()(x)

    x = Activation('relu')(x)
    return x


def expand_conv_basic_block(Input, filters, stride=1, dropout=0.0):
    Init = Input

    # First conv which is used to downsample the image
    x = Conv2D(filters, kernel_size=(3, 3), strides=(stride, stride), padding="same")(Input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Optional Dropout layer
    if (dropout > 0.0):
        x = Dropout(dropout)(x)

    x = Conv2D(filters, kernel_size=(3, 3), strides=(1, 1), padding="same")(x)
    x = BatchNormalization()(x)

    # Projection shortcut to make skip connection(Paper terminology)
    skip_conv = Conv2D(filters, kernel_size=(1, 1), strides=(stride, stride), padding="same")(Input)
    skip = BatchNormalization()(skip_conv)

    # Skip connection
    x = Add()([x, skip])
    return x


def normal_conv_basic_block(Input, filters, stride=1, dropout=0.0):
    x = Conv2D(filters, kernel_size=(3, 3), strides=(stride, stride), padding="same")(Input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Optional Dropout layer
    if (dropout > 0.0):
        x = Dropout(dropout)(x)

    x = Conv2D(filters, kernel_size=(3, 3), strides=(stride, stride), padding="same")(x)
    x = BatchNormalization()(x)

    # Identity skip connection
    x = Add()([x, Input])

    return x


def expand_conv_bottleneck_block(Input, filters, stride=1, dropout=0.0):
    # Contracting 1*1 conv
    x = Conv2D(filters, kernel_size=(1, 1), strides=(stride, stride), padding="same")(Input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # if(dropout > 0.0):
    #   x = Dropout(dropout)(x)

    # Depth preserving 3*3 conv
    x = Conv2D(filters, kernel_size=(3, 3), strides=(1, 1), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # if(Dropout > 0.0):
    #   x = Dropout(dropout)(x)

    # Expanding 1*1 Conv
    x = Conv2D(filters * 4, kernel_size=(1, 1), strides=(1, 1), padding="same")(x)
    x = BatchNormalization()(x)

    # Projection shortcut
    skip_conv = Conv2D(filters * 4, kernel_size=(1, 1), strides=(stride, stride), padding="same")(Input)
    skip = BatchNormalization()(skip_conv)

    # Skip connection
    x = Add()([x, skip])

    return x

def normal_conv_bottleneck_block(Input, filters, stride=1, dropout=0.0):
    # Contracting 1*1 conv
    x = Conv2D(filters, kernel_size=(1, 1), strides=(stride, stride), padding="same")(Input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # if(dropout > 0.0):
    #   x = Dropout(dropout)(x)

    # Depth preserving 3*3 Conv
    x = Conv2D(filters, kernel_size=(3, 3), strides=(stride, stride), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # if(Dropout > 0.0):
    #    x = Dropout(dropout)(x)

    # Expanding 1*1 Conv
    x = Conv2D(filters * 4, kernel_size=(1, 1), strides=(stride, stride), padding="same")(x)
    x = BatchNormalization()(x)

    # Identity skip connection
    x = Add()([x, Input])

    return x


def build_basic_resnet(h, w, no_of_outputs, r1, r2, r3, r4, first_conv_stride=2, first_max_pool=True,
                       first_conv_kernel_size=7):
    # Creating input tensor
    inputs = Input(shape=(h, w, 3), name="image_input")

    # Inital Conv block
    x = initial_conv(inputs, 64, first_conv_stride, first_conv_kernel_size)

    # Optional Max pooling layer
    if (first_max_pool):
        x = MaxPooling2D(pool_size=(2, 2))(x)

    # Expanding block1 with projection shortcut
    x = expand_conv_basic_block(x, 64, 1)
    x = Activation('relu')(x)

    # Repeating block of Conv1
    for i in range(r1 - 1):
        x = normal_conv_basic_block(x, 64)
        x = Activation('relu')(x)

    # Expanding block2 with projection shortcut
    x = expand_conv_basic_block(x, 128, 2)
    x = Activation('relu')(x)

    # Repeating block of Conv2
    for i in range(r2 - 1):
        x = normal_conv_basic_block(x, 128)
        x = Activation('relu')(x)

    # Expanding block3 with projection shortcut
    x = expand_conv_basic_block(x, 256, 2)
    x = Activation('relu')(x)

    # Repeating block of Conv3
    for i in range(r3 - 1):
        x = normal_conv_basic_block(x, 256)
        x = Activation('relu')(x)

    # Expanding block4 with projection shortcut
    x = expand_conv_basic_block(x, 512, 2)
    x = Activation('relu')(x)

    # Repeating block of Conv3
    for i in range(r4 - 1):
        x = normal_conv_basic_block(x, 512)
        x = Activation('relu')(x)

    shape = K.int_shape(x)

    # Average pooling layer
    x = AveragePooling2D(pool_size=(shape[1], shape[2]),
                         strides=(1, 1))(x)
    # x = GlobalAveragePooling2D()(x)
    x = Flatten()(x)

    # Classifier Block
    x = Dense(no_of_outputs, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=x)
    return model


def build_bottleneck_resnet(h, w, no_of_outputs, r1, r2, r3, r4, first_conv_stride=2, first_max_pool=True,
                            first_conv_kernel_size=7):
    # Creating input tensor
    inputs = Input(shape=(h, w, 3), name="image_input")

    # Inital Conv block
    x = initial_conv(inputs, 64, first_conv_stride, first_conv_kernel_size)

    # Optional Max pooling layer
    if (first_max_pool):
        x = MaxPooling2D(pool_size=(2, 2))(x)

    # Expanding block1 with projection shortcut
    x = expand_conv_bottleneck_block(x, 64, 1)
    x = Activation('relu')(x)

    # Repeating block of Conv1
    for i in range(r1 - 1):
        x = normal_conv_bottleneck_block(x, 64)
        x = Activation('relu')(x)

    # Expanding block2 with projection shortcut
    x = expand_conv_bottleneck_block(x, 128, 2)
    x = Activation('relu')(x)

    # Repeating block of Conv2
    for i in range(r2 - 1):
        x = normal_conv_bottleneck_block(x, 128)
        x = Activation('relu')(x)

    # Expanding block3 with projection shortcut
    x = expand_conv_bottleneck_block(x, 256, 2)
    x = Activation('relu')(x)

    # Repeating block of Conv3
    for i in range(r3 - 1):
        x = normal_conv_bottleneck_block(x, 256)
        x = Activation('relu')(x)

    # Expanding block4 with projection shortcut
    x = expand_conv_bottleneck_block(x, 512, 2)
    x = Activation('relu')(x)

    # Repeating block of Conv4
    for i in range(r4 - 1):
        x = normal_conv_bottleneck_block(x, 512)
        x = Activation('relu')(x)

    shape = K.int_shape(x)

    # Average pooling layer
    x = AveragePooling2D(pool_size=(shape[1], shape[2]),
                         strides=(1, 1))(x)
    # x = GlobalAveragePooling2D()(x)

    # Classifier Block
    x = Flatten()(x)
    x = Dense(no_of_outputs, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=x)
    return model



model = build_bottleneck_resnet(32,32,4,3,4,6,3,2,True,7)

model.summary()

plot_model(model,"ResNet50.png",show_shapes=True)


model.compile(loss='categorical_crossentropy',
        optimizer="Adam",
        metrics=['accuracy'])


lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=2, min_lr=0.5e-6)
csv_logger = CSVLogger('/home/shabbeer/Desktop/CV_Project_imp_contribution/IEEE Journal/without Data Augmentation/ResNet-50/history_NDA_acc.csv')
model_chekpoint = ModelCheckpoint("ResNet50_F1score_DA.hdf5",monitor = 'val_loss',verbose = 1,save_best_only=True)


batch_size = 64
data_augmentation = False
epochs = 500

if data_augmentation:
    print("-------------Using Data augmentation------------")
    # This will do preprocessing and realtime data augmentation:
    datagen = ImageDataGenerator(
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True)  # randomly flip images

    datagen.fit(x_train)
    history = model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                        steps_per_epoch=x_train.shape[0] // batch_size,
                        epochs=epochs, verbose=1, validation_data=(x_test, y_test),
                        callbacks=[lr_reducer,  csv_logger, model_chekpoint])

else:
    print("-----Not Using Data augmentation---------------")
    history = model.fit(x_train, y_train,
              batch_size=batch_size * 4,
              epochs=epochs,
              validation_data=(x_test, y_test),
              shuffle=True, callbacks=[lr_reducer,  csv_logger, model_chekpoint])
scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
import matplotlib.pyplot as plt
print("Max Test accuracy", max(history.history['val_acc']))
import matplotlib.pyplot  as plt
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model Accuracy')
plt.ylabel('accuracy')
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
