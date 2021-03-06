import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras

from keras.models import Model
from keras import optimizers
from keras.layers import *
from keras import regularizers
from keras import initializers
from keras import callbacks
from keras.utils.vis_utils import plot_model, model_to_dot
from IPython.display import Image, SVG


class Segnet:
    def __init__(self,X_train,y_train):
        self.X_train = X_train
        self.y_train = y_train

    def net(self):

        Input_layer = Input(self.X_train.shape)                    # (k,224,224,3)

        # encoder
        conv1 = convolutional.Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(Input_layer)
        bn = normalization.BatchNormalization()(conv1)
        conv1 = convolutional.Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(bn)
        bn = normalization.BatchNormalization()(conv1)
        max_pooling1 = pooling.MaxPooling2D(pool_size=(2, 2))(bn)  # (k,112,112,64)

        conv2 = convolutional.Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu')(max_pooling1)
        bn = normalization.BatchNormalization()(conv2)
        conv2 = convolutional.Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu')(bn)
        bn = normalization.BatchNormalization()(conv2)
        max_pooling2 = pooling.MaxPooling2D(pool_size=(2, 2))(bn)  # (k,56,56,128)

        conv3 = convolutional.Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu')(max_pooling2)
        bn = normalization.BatchNormalization()(conv3)
        conv3 = convolutional.Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu')(bn)
        bn = normalization.BatchNormalization()(conv3)
        conv3 = convolutional.Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu')(bn)
        bn = normalization.BatchNormalization()(conv3)
        max_pooling3 = pooling.MaxPooling2D(pool_size=(2, 2))(bn)  # (k,28,28,256)

        conv4 = convolutional.Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu')(max_pooling3)
        bn = normalization.BatchNormalization()(conv4)
        conv4 = convolutional.Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu')(bn)
        bn = normalization.BatchNormalization()(conv4)
        conv4 = convolutional.Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu')(bn)
        bn = normalization.BatchNormalization()(conv4)
        max_pooling4 = pooling.MaxPooling2D(pool_size=(2, 2))(bn)  # (k,14,14,512)

        conv5 = convolutional.Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu')(max_pooling4)
        bn = normalization.BatchNormalization()(conv5)
        conv5 = convolutional.Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu')(bn)
        bn = normalization.BatchNormalization()(conv5)
        conv5 = convolutional.Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu')(bn)
        bn = normalization.BatchNormalization()(conv5)
        max_pooling5 = pooling.MaxPooling2D(pool_size=(2, 2))(bn)  # (k,7,7,512)

        # decoder
        upsample1 = convolutional.UpSampling2D(size=(2, 2))(max_pooling5)  # (k,14,14,512)
        conv6 = convolutional.Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu')(upsample1)
        bn = normalization.BatchNormalization()(conv6)
        conv6 = convolutional.Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu')(bn)
        bn = normalization.BatchNormalization()(conv6)
        conv6 = convolutional.Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu')(bn)
        bn = normalization.BatchNormalization()(conv6)

        upsample2 = convolutional.UpSampling2D(size=(2, 2))(bn)  # (k,28,28,512)
        conv7 = convolutional.Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu')(upsample2)
        bn = normalization.BatchNormalization()(conv7)
        conv7 = convolutional.Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu')(bn)
        bn = normalization.BatchNormalization()(conv7)
        conv7 = convolutional.Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu')(bn)
        bn = normalization.BatchNormalization()(conv7)

        upsample3 = convolutional.UpSampling2D(size=(2, 2))(bn)  # (k,56,56,512)
        conv8 = convolutional.Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu')(upsample3)
        bn = normalization.BatchNormalization()(conv8)
        conv8 = convolutional.Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu')(bn)
        bn = normalization.BatchNormalization()(conv8)
        conv8 = convolutional.Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu')(bn)
        bn = normalization.BatchNormalization()(conv8)

        upsample4 = convolutional.UpSampling2D(size=(2, 2))(bn)  # (k,112,112,256)
        conv9 = convolutional.Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu')(upsample4)
        bn = normalization.BatchNormalization()(conv9)
        conv9 = convolutional.Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu')(bn)
        bn = normalization.BatchNormalization()(conv9)

        upsample5 = convolutional.UpSampling2D(size=(2, 2))(bn)  # (k,224,224,128)
        conv10 = convolutional.Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(upsample5)
        bn = normalization.BatchNormalization()(conv10)
        conv10 = convolutional.Conv2D(filters=11, kernel_size=(3, 3), padding='same', activation='relu')(bn)
        bn = normalization.BatchNormalization()(conv10)

        Output_layer = Activation('softmax')(bn)

        self.Input_layer = Input_layer
        self.Output_layer = Output_layer

    def make_model(self):

        sgd = optimizers.SGD(lr=1e-5, decay=1e-6, momentum=0.9, nesterov=True)
        self.model = Model(inputs=self.Input_layer, outputs=self.Output_layer)
        self.model.compile(loss='categorical_crossentropy',
                      optimizer=sgd,
                      metrics=['accuracy'])

    def net_visualization(self,graph_save=False):

        if graph_save == True:
            plot_model(self.model, to_file='model.png')

        SVG(model_to_dot(self.model, show_shapes=True).create(prog='dot', format='svg'))

    def train(self,batch_size=64,epoch=5,verbose=2,):

        self.history = self.model.fit(x=self.X_train, y=self.y_train, batch_size=batch_size, epochs=epoch, verbose=verbose,
                  callbacks=[callbacks.ReduceLROnPlateau(patience=1, cooldown=10)])

    def evaluate(self):

        pass