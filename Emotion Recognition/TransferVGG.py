from keras.layers import Input, Lambda, Dense, Flatten
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
import numpy as np
from glob import glob
import matplotlib.pyplot as plt


from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

train_path = 'D:/ConvolutionaNeuralNetwork/Emotion Dataset/AffectNet/UpdatedShrinkenTrain'
valid_path = 'D:/ConvolutionaNeuralNetwork/Emotion Dataset/AffectNet/UpdatedShrinkenTest'

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   rotation_range=30,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   width_shift_range=0.4,
					                         height_shift_range=0.4,
                                   horizontal_flip = True,
                                   fill_mode='nearest'
                                  )

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory(train_path,
                                                 #color_mode='grayscale',
                                                 target_size = (224, 224),
                                                 batch_size = 16,
                                                 class_mode = 'categorical',
                                                 shuffle=True)

test_set = test_datagen.flow_from_directory(valid_path,
                                            #color_mode='grayscale',
                                            target_size = (224, 224),
                                            batch_size = 16,
                                            class_mode = 'categorical',
                                            shuffle=True)


# re-size all the images to this
IMAGE_SIZE = [224, 224]



# add preprocessing layer to the front of VGG
#vgg = VGG16(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)

modelGit = tf.keras.models.load_model('expression.model')
modelGit.summary()
print("AAAAAAAAAAAAAAAAA")
modelGit._layers.pop()
modelGit._layers.pop()
modelGit.summary()
# don't train existing weights
for layer in modelGit.layers:
  layer.trainable = False
  

  
  # useful for getting number of classes
folders = glob('D:/ConvolutionaNeuralNetwork/Emotion Dataset/AffectNet/UpdatedShrinkenTrain/*')
  

# our layers - you can add more if you want
#x = Flatten()(modelGit.output)
# x = Dense(1000, activation='relu')(x)
#prediction = Dense(len(folders), activation='softmax')(x)

x=modelGit.layers[-1].output
prediction = Dense(len(folders), activation='softmax', name='dense_2')(x)
# create a model object
model = Model(inputs=modelGit.input, outputs=prediction)

# view the structure of the model
model.summary()

# tell the model what cost and optimization method to use
"""
model.compile(
  loss='categorical_crossentropy',
  optimizer='adam',
  metrics=['accuracy']
)
"""










from keras.optimizers import RMSprop,SGD,Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

nb_train_samples = 283901
nb_validation_samples = 3500
epochs=70
batch_size = 16


checkpoint = ModelCheckpoint('Git_vgg.h5',
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


history=model.fit_generator(
                training_set,
                steps_per_epoch=nb_train_samples//batch_size,
                epochs=epochs,
                callbacks=callbacks,
                validation_data=test_set,
                validation_steps=nb_validation_samples//batch_size)















# fit the model
"""
r = model.fit_generator(
  training_set,
  validation_data=test_set,
  epochs=5,
  steps_per_epoch=len(training_set),
  validation_steps=len(test_set)
)
"""


"""

# loss
plt.plot(r.history['loss'], label='train loss')
plt.plot(r.history['val_loss'], label='val loss')
plt.legend()
plt.show()
plt.savefig('LossVal_loss')

# accuracies
plt.plot(r.history['acc'], label='train acc')
plt.plot(r.history['val_acc'], label='val acc')
plt.legend()
plt.show()
plt.savefig('AccVal_acc')

import tensorflow as tf

from keras.models import load_model

model.save('facefeatures_new_model.h5')

"""















































