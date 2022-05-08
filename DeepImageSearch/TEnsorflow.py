import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Add
from tensorflow.keras.layers import Input, Lambda, Dense, Flatten ,Conv2D, MaxPool2D,Dropout,LeakyReLU,BatchNormalization,GlobalAveragePooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras import optimizers
from datetime import datetime
import matplotlib.pyplot as plt
from keras.preprocessing import image
from keras.models import load_model
import os
from DeepImageSearch.utils.allutils import util
import time
import argparse


class _Model:
    @staticmethod
    def CreateModel(out_class):
        model = Sequential()
        # Performing 2dconvolution followed by BatchNormalization and Dropout
        model.add(Conv2D(32,(5,5),input_shape=(224,224,3), activation="relu"))
        model.add(Conv2D(32,(3,3), padding="same", activation=LeakyReLU(alpha=0.01)))
        model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

        model.add(Conv2D(64,(3,3), padding="same", activation=LeakyReLU(alpha=0.01)))
        model.add(Conv2D(64,(3,3), padding="same", activation=LeakyReLU(alpha=0.01)))
        model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

        model.add(Conv2D(128,(3,3), padding="same",activation=LeakyReLU(alpha=0.01)))
        model.add(Conv2D(128,(3,3), padding="same",activation=LeakyReLU(alpha=0.01)))
        model.add(BatchNormalization())
        model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

        model.add(Conv2D(256,(3,3), padding="same", activation=LeakyReLU(alpha=0.01)))
        model.add(Conv2D(256,(3,3), padding="same", activation=LeakyReLU(alpha=0.01)))
        model.add(BatchNormalization())
        model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))


        model.add(Conv2D(256,(3,3), padding="same",activation=LeakyReLU(alpha=0.01)))
        model.add(Conv2D(256,(3,3), padding="same",activation=LeakyReLU(alpha=0.01)))
        model.add(BatchNormalization())
        model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

        model.add(Conv2D(1024,(3,3), padding="same",activation=LeakyReLU(alpha=0.01)))
        model.add(BatchNormalization())

        model.add(Flatten())
        model.add(Dropout(rate=0.3))
        model.add(BatchNormalization())
        model.add(Dense(units=500,activation=LeakyReLU(alpha=0.01)))
        model.add(BatchNormalization())
        model.add(Dropout(rate=0.3))
        model.add(Dense(units=1000,activation=LeakyReLU(alpha=0.01)))
        model.add(Dense(units=out_class, activation="softmax"))

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        return model

class CustomModelTensorFlow():
    def __init__(self,config,params):

        params = util.read_yaml(params)
        config = util.read_yaml(config)

        artifact_dir = config['artifacts']['artifactdir']
        imagedir_name = config['artifacts']["image_dir"]
        preprocessed = config['artifacts']['preprocessed']
        train_dir_name = config['artifacts']["train_dir"]
        val_dir_name = config['artifacts']["val_dir"]
        metadata_dir = config['artifacts']["meta_data_dir"]
        model_name = config['artifacts']['tensoflowmodel']
        historyfile = config['artifacts']['tensorflowhistory']

        self.train_dir = os.path.join(artifact_dir, imagedir_name,preprocessed,train_dir_name)
        self.val_dir = os.path.join(artifact_dir, imagedir_name,preprocessed,val_dir_name)
        self.modelfilename = os.path.join(artifact_dir, metadata_dir, model_name)
        self.historypath = os.path.join(artifact_dir,metadata_dir,historyfile)

        self.batchsize = params['batch_size']
        self.epochs = params['epoch']
        self.lr = params['lr']
        no_of_class = params['class']
        self.model = _Model.CreateModel(no_of_class)


    @classmethod
    def _train_from_directory(cls,train_dir,val_dir,batchsize):
        train_datagen = ImageDataGenerator(rescale=1 / 255.0)
        input_shape = (224, 224)

        train_set = train_datagen.flow_from_directory(train_dir, target_size=input_shape, batch_size=batchsize,
                                                      shuffle=True )
        val_set = train_datagen.flow_from_directory(val_dir, target_size=input_shape, batch_size=batchsize,
                                                    shuffle=True)

        return train_set, val_set


    @classmethod
    def _callbacks(cls,modelfilename, early_stopping=None, lr_reducer=None):
        callbacks = []
        if modelfilename is not None:
            checkpoint = ModelCheckpoint(filepath=modelfilename, verbose=1, save_best_only=True)
            callbacks.append(checkpoint)

        if early_stopping:
            early_stopping = EarlyStopping(monitor='val_loss',
                                           min_delta=0,
                                           patience=5,
                                           verbose=1,
                                           restore_best_weights=True
                                           )
            callbacks.append(early_stopping)
        if lr_reducer:
            lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                                           cooldown=0,
                                           patience=5,
                                           min_lr=0.5e-6)

            callbacks.append(lr_reducer)

        return callbacks

    def fit(self,):

        since = time.time()

        train_set, val_set = self._train_from_directory(self.train_dir,self.val_dir,self.batchsize)
        history = self.model.fit(
            train_set,
            validation_data=val_set,
            epochs=self.epochs,
            steps_per_epoch=2,
            validation_steps=val_set.n // val_set.batch_size,
            callbacks=self._callbacks(self.modelfilename))

        time_elapsed = time.time() - since
        print(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s Epoch: {self.epochs}")
        util.dump_pickle(self.historypath,history.history)

def maintensor():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',"-c", type=str, default="config/config.yaml", help='ROOT/config/config.yaml')
    parser.add_argument('--params',"-p", type=str, default="params.yaml", help='ROOT/params.yaml')
    opt = parser.parse_args()
    model = CustomModelTensorFlow(**vars(opt))
    model.fit()

if __name__ == '__main__':
    maintensor()