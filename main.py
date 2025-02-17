#Import Libraries
import matplotlib.pyplot as plt
import numpy as np
import cv2
import pandas as pd
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping,ReduceLROnPlateau
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
#declare a variable workingDir which is of type string
#It should contain the path to data folder for this application
train_dir = ''
val_dir = ''
num_train = 28709
num_val = 7178
batch_size = 64
num_epoch = 70
MODEL = None
train_generator = None
validation_generator = None
def buildCNNModel():
  #Convolution Neural Network
  model = Sequential()
  model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
  model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Dropout(0.25))
  model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Dropout(0.25))
  model.add(Flatten())
  model.add(Dense(1024, activation='relu'))
  model.add(Dropout(0.5))
  model.add(Dense(7, activation='softmax'))
  model.compile(loss='categorical_crossentropy',
                optimizer=Adam(learning_rate=0.001),
                metrics=['accuracy']
                )
  print(model.summary())
  return model
#1
def loadDataForTraining(workingDir:str):
  global train_dir, val_dir
  train_dir = f'{workingDir}/data/train' #Replace the Path
  val_dir = f'{workingDir}/data/test' #Replace the Path
#2
def processDataForTraining():
  global train_generator
  global validation_generator
  #We normalize the images to a standard size and create a series of batch
  datagen = ImageDataGenerator(rescale=1./255)
  train_generator = datagen.flow_from_directory(
        train_dir,
        target_size=(48,48),
        batch_size=batch_size,
        color_mode="grayscale",
        class_mode='categorical')
  validation_generator = datagen.flow_from_directory(
          val_dir,
          target_size=(48,48),
          batch_size=batch_size,
          color_mode="grayscale",
          class_mode='categorical')
  print("Data has been processed")
#3
def trainTheModel():
  print("Started Training Model")
  global MODEL
  # model.compile(loss='categorical_crossentropy',optimizer=Adam(lr=0.0001, decay=1e-6),metrics=['accuracy'])
  #Transfer learning: We use learnings from previously trained model as base in this application.
  checkpoint = ModelCheckpoint(
                                'emotion_face_mobilNet.keras',
                                monitor='val_loss',
                                mode='min',
                                save_best_only=True,
                                verbose=1)
  earlystop = EarlyStopping(
                            monitor='val_loss',
                            min_delta=0,
                            patience=10,
                            verbose=1,restore_best_weights=True)
  learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc',
                                              patience=5,
                                              verbose=1,
                                              factor=0.2,
                                              min_lr=0.0001)
  callbackList = [earlystop,checkpoint,learning_rate_reduction]
  MODEL.fit(
        train_generator,
        steps_per_epoch=num_train // batch_size,
        epochs=num_epoch,
        validation_data=validation_generator,
        validation_steps=num_val // batch_size)
#4
def evaluateModel():
    # summarize history for accuracy
    pd.DataFrame(MODEL.history.history)[['accuracy','val_accuracy']].plot()
    plt.title("Accuracy")
    plt.show()
    # summarize history for loss
    pd.DataFrame(MODEL.history.history)[['loss','val_loss']].plot()
    plt.title("Loss")
    plt.show()
#5
def saveModel():
  MODEL.save(r'C:\Users\Om\Downloads\FacialExpressionAnalysis\FacialExpressionAnalysis\EmojiScavenger.kerasr')
MODEL = buildCNNModel()
loadDataForTraining(r'C:\Users\Om\Downloads\FacialExpressionAnalysis\FacialExpressionAnalysis\Emotion-detection-master')
processDataForTraining()
trainTheModel()
evaluateModel()
saveModel()


