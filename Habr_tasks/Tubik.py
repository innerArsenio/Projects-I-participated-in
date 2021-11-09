from __future__ import print_function
import os
import zipfile
import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from keras import layers
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.densenet import DenseNet121
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from keras import backend as K
from keras.models import Sequential
from keras.layers            import MaxPooling2D
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers import Adam

from keras.models import load_model
from keras.callbacks import Callback
from shutil import copyfile, rmtree

from keras.layers import Dense,Dropout,Activation,Flatten,BatchNormalization
from keras.layers import Conv2D,MaxPooling2D


import keras


# Kaggle dataset
normal_path = "TB_Chest_Radiography_Database/Normal"
tb_path = "TB_Chest_Radiography_Database/Tuberculosis"
print(f"Normal X-rays {len(os.listdir(normal_path))}")
print(f"TB X-rays {len(os.listdir(tb_path))}")


# make directories for storing training and test sets
try:
    parent_dir = './tb-xray-database/'
    os.mkdir(parent_dir)
    training_path = os.path.join(parent_dir, "training")
    testing_path = os.path.join(parent_dir, "testing")
    os.mkdir(training_path)
    os.mkdir(testing_path)
    os.mkdir(os.path.join(training_path, "normal"))
    os.mkdir(os.path.join(training_path, "tb"))
    os.mkdir(os.path.join(testing_path, "normal"))
    os.mkdir(os.path.join(testing_path, "tb"))
except OSError:
    print("couldn't make it")
    pass



# split into training and test sets

def split_data(SOURCE, TRAINING, TESTING, SPLIT_SIZE):
    path_list = os.listdir(SOURCE)# file names in SOURCE
    length = len(path_list)
    count = 0
    for file in random.sample(path_list, length):
        file_path = os.path.join(SOURCE, file)
        if (os.path.getsize(file_path) != 0):
            if (count < SPLIT_SIZE*length):
                copyfile(file_path, os.path.join(TRAINING, file))
                count+=1
            else:
                copyfile(file_path, os.path.join(TESTING, file))


NORMAL_SOURCE_DIR = "TB_Chest_Radiography_Database/Normal"
TRAINING_NORMAL_DIR = os.path.join(training_path, "normal")
TESTING_NORMAL_DIR = os.path.join(testing_path, "normal")
TB_SOURCE_DIR = "TB_Chest_Radiography_Database/Tuberculosis"
TRAINING_TB_DIR = os.path.join(training_path, "tb")
TESTING_TB_DIR = os.path.join(testing_path, "tb")

split_size = .7
split_data(NORMAL_SOURCE_DIR, TRAINING_NORMAL_DIR, TESTING_NORMAL_DIR, split_size)
split_data(TB_SOURCE_DIR, TRAINING_TB_DIR, TESTING_TB_DIR, split_size)



parent_dir = './tb-xray-database/'
training_path = os.path.join(parent_dir, "training")
testing_path = os.path.join(parent_dir, "testing")
NORMAL_SOURCE_DIR = "TB_Chest_Radiography_Database/Normal"
TRAINING_NORMAL_DIR = os.path.join(training_path, "normal")
TESTING_NORMAL_DIR = os.path.join(testing_path, "normal")
TB_SOURCE_DIR = "TB_Chest_Radiography_Database/Tuberculosis"
TRAINING_TB_DIR = os.path.join(training_path, "tb")
TESTING_TB_DIR = os.path.join(testing_path, "tb")




print(len(os.listdir(TRAINING_NORMAL_DIR)))
print(len(os.listdir(TESTING_NORMAL_DIR)))
print(len(os.listdir(TRAINING_TB_DIR)))
print(len(os.listdir(TESTING_TB_DIR)))



# get training and testing generators
training_path = os.path.join(parent_dir, "training")
testing_path = os.path.join(parent_dir, "testing")
train_datagen = ImageDataGenerator(samplewise_center=True, samplewise_std_normalization= True)
train_generator = train_datagen.flow_from_directory(training_path, color_mode='grayscale', batch_size = 32, target_size = (320,320), class_mode = 'binary')
test_datagen = ImageDataGenerator(samplewise_center=True, samplewise_std_normalization= True)
test_generator = test_datagen.flow_from_directory(testing_path, color_mode='grayscale', batch_size = 32, target_size = (320,320), class_mode = 'binary')



#############################################################
#############################################################
#############################################################





num_classes = 2
img_rows,img_cols = 250,250 # Resizing images to 250x250
batch_size = 16


train_data_dir = 'tb-xray-database/training'
validation_data_dir = 'tb-xray-database/testing'
 
train_datagen = ImageDataGenerator(
					rescale=1./255,
					rotation_range=30,
					shear_range=0.3,
					zoom_range=0.3,
					width_shift_range=0.4,
					height_shift_range=0.4,
					horizontal_flip=True,
					fill_mode='nearest')

validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
					train_data_dir,
					color_mode='grayscale',
					target_size=(img_rows,img_cols),
					batch_size=batch_size,
					class_mode='categorical',
					shuffle=True)

validation_generator = validation_datagen.flow_from_directory(
							validation_data_dir,
							color_mode='grayscale',
							target_size=(img_rows,img_cols),
							batch_size=batch_size,
							class_mode='categorical',
							shuffle=True)

print("lalala")
model = Sequential()

# Block-1

model.add(Conv2D(32,(3,3),padding='same',kernel_initializer='he_normal',input_shape=(img_rows,img_cols,1)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(32,(3,3),padding='same',kernel_initializer='he_normal',input_shape=(img_rows,img_cols,1)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

# Block-2 

model.add(Conv2D(64,(3,3),padding='same',kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(64,(3,3),padding='same',kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

# Block-3

model.add(Conv2D(128,(3,3),padding='same',kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(128,(3,3),padding='same',kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

# Block-4 

model.add(Conv2D(256,(3,3),padding='same',kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(256,(3,3),padding='same',kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

# Block-5

model.add(Flatten())
model.add(Dense(64,kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

# Block-6

model.add(Dense(64,kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

# Block-7

model.add(Dense(num_classes,kernel_initializer='he_normal'))
model.add(Activation('softmax'))

print(model.summary())

from tensorflow.keras.optimizers import RMSprop,SGD,Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

checkpoint = ModelCheckpoint('Tubik_model.h5',
                             monitor='val_loss',
                             mode='min',
                             save_best_only=True,
                             verbose=1)

earlystop = EarlyStopping(monitor='val_loss',
                          min_delta=0,
                          patience=9,
                          verbose=1,
                          restore_best_weights=True
                          )

reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                              factor=0.2,
                              patience=3,
                              verbose=1,
                              min_delta=0.0001)

callbacks = [earlystop,checkpoint,reduce_lr]

model.compile(loss='categorical_crossentropy',
              optimizer = Adam(lr=0.001),
              metrics=['accuracy'])

nb_train_samples = 2940
nb_validation_samples = 1260
epochs=20

history=model.fit_generator(
                train_generator,
                steps_per_epoch=nb_train_samples//batch_size,
                epochs=epochs,
                callbacks=callbacks,
                validation_data=validation_generator,
                validation_steps=nb_validation_samples//batch_size)







