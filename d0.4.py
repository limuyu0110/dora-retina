#-*- coding:utf-8 _*-  
""" 
@author:limuyu
@file: trial.py 
@time: 2019/07/20
@contact: limuyu0110@pku.edu.cn

"""

import pandas as pd
import numpy as np
import tensorflow as tf
import cv2
import os
from tqdm import tqdm

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score

from imgaug import augmenters as iaa

import keras.backend as K
from keras.models import Sequential, load_model, Model
from keras.optimizers import Adam, SGD
from keras.layers import (
    Activation,
    Dropout,
    Flatten,
    Dense,
    GlobalMaxPooling2D,
    BatchNormalization,
    Input,
    Conv2D,
    GlobalAveragePooling2D
)
from keras.applications.resnet50 import ResNet50
from keras.utils import (
    to_categorical,
    Sequence
)
from keras.callbacks import (
    ModelCheckpoint,
    LearningRateScheduler,
    EarlyStopping,
    ReduceLROnPlateau,
    CSVLogger,
    Callback
)


epochs = 30
warmup_batch_size = 64
batch_size = 16
SIZE = 300

# checkpoint = ModelCheckpoint('./trial.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min', save_weights_only=True)
reducelr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, verbose=4, mode='min', epsilon=0.00001)
early = EarlyStopping(monitor='val_loss', mode='min', patience=9)
csv_logger = CSVLogger(filename='./d0.4/training_log.csv', separator=',', append=True)


sometimes = lambda aug: iaa.Sometimes(0.5, aug)
seq = iaa.Sequential([
    sometimes(
        iaa.OneOf([
            iaa.Add((-10, 10), per_channel=0.5),
            iaa.Multiply((0.9, 1.1), per_channel=0.5),
            iaa.ContrastNormalization((0.9, 1.1), per_channel=0.5)
        ])
    ),
    iaa.Fliplr(0.5),
    iaa.Crop(percent=(0, 0.1)),
    iaa.Flipud(0.5)
], random_order=True)


def load_data():
    df_train = pd.read_csv('./data/raw/train.csv')
    # df_test = pd.read_csv('./data/raw/test.csv')

    x = df_train['id_code']
    y = df_train['diagnosis']

    x, y = shuffle(x, y, random_state=8)

    print(y.shape)
    y = to_categorical(y, num_classes=5)
    train_x, valid_x, train_y, valid_y = train_test_split(x, y, test_size=0.15, stratify=y, random_state=8)

    return train_x, valid_x, train_y, valid_y


def create_model(input_shape, n_out):
    input_tensor = Input(shape=input_shape)
    base_model = ResNet50(
        include_top=False,
        weights=None,
        input_tensor=input_tensor
    )
    base_model.load_weights('./data/resnet50/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5')
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dropout(0.4)(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.4)(x)
    final_output = Dense(n_out, activation='softmax', name='final_output')(x)
    model = Model(input_tensor, final_output)

    return model


class My_Generator(Sequence):
    def __init__(self, image_filenames, labels, batch_size, is_train=False, mix=False, augment=False):
        self.image_filenames, self.labels = image_filenames, labels
        self.batch_size = batch_size
        self.is_train = is_train
        self.is_augment = augment
        if self.is_train:
            self.on_epoch_end()
        self.is_mix = mix

    def __len__(self):
        return int(np.ceil(len(self.image_filenames) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.image_filenames[idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_y = self.labels[idx * self.batch_size: (idx + 1) * self.batch_size]

        if self.is_train:
            return self.train_generate(batch_x, batch_y)
        return self.valid_generate(batch_x, batch_y)

    def on_epoch_end(self):
        if self.is_train:
            self.image_filenames, self.labels = shuffle(self.image_filenames, self.labels)

    def mix_up(self, x, y):
        lam = np.random.beta(0.2, 0.4)
        ori_index = np.arange(int(len(x)))
        index_array = np.arange(int(len(x)))
        np.random.shuffle(index_array)

        mixed_x = lam * x[ori_index] + (1 - lam) * x[index_array]
        mixed_y = lam * y[ori_index] + (1 - lam) * y[index_array]

        return mixed_x, mixed_y

    def train_generate(self, batch_x, batch_y):
        batch_images = []
        for (sample, label) in zip(batch_x, batch_y):
            img = cv2.imread('./data/raw/train_images/' + sample + '.png')
            img = cv2.resize(img, (SIZE, SIZE))
            if self.is_augment:
                img = seq.augment_image(img)
            batch_images.append(img)
        batch_images = np.array(batch_images, np.float32) / 255.0
        batch_y = np.array(batch_y, np.float32)
        if self.is_mix:
            batch_images, batch_y = self.mix_up(batch_images, batch_y)

        return batch_images, batch_y

    def valid_generate(self, batch_x, batch_y):
        batch_images = []
        for (sample, label) in zip(batch_x, batch_y):
            img = cv2.imread('./data/raw/train_images/' + sample + '.png')
            img = cv2.resize(img, (SIZE, SIZE))
            batch_images.append(img)
        batch_images = np.array(batch_images, np.float32) / 255.0
        batch_y = np.array(batch_y, np.float32)
        return batch_images, batch_y


class QWKEvaluation(Callback):
    def __init__(self, validation_data=(), batch_size=64, interval=1):
        super(Callback, self).__init__()

        self.interval = interval
        self.batch_size = batch_size
        self.valid_generator, self.y_val = validation_data
        self.history = []

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict_generator(generator=self.valid_generator,
                                                  steps=np.ceil(float(len(self.y_val)) / float(self.batch_size)),
                                                  workers=1, use_multiprocessing=True,
                                                  verbose=1)

            def flatten(y):
                return np.argmax(y, axis=1).reshape(-1)
                # return np.sum(y.astype(int), axis=1) - 1

            score = cohen_kappa_score(flatten(self.y_val),
                                      flatten(y_pred),
                                      labels=[0, 1, 2, 3, 4],
                                      weights='quadratic')
            #             print(flatten(self.y_val)[:5])
            #             print(flatten(y_pred)[:5])
            print("\n epoch: %d - QWK_score: %.6f \n" % (epoch + 1, score))
            self.history.append(score)
            if score >= max(self.history):
                print('save checkpoint: ', score)
                self.model.save('./d0.4/Resnet50_bestqwk.h5')


if __name__ == '__main__':
    train_x, valid_x, train_y, valid_y = load_data()
    train_generator = My_Generator(train_x, train_y, warmup_batch_size, is_train=True)
    train_mixup = My_Generator(train_x, train_y, batch_size, is_train=True, mix=False, augment=True)
    valid_generator = My_Generator(valid_x, valid_y, batch_size, is_train=False)
    qwk = QWKEvaluation(validation_data=(valid_generator, valid_y), batch_size=batch_size, interval=1)

    model = create_model(
        input_shape=(SIZE, SIZE, 3),
        n_out=5
    )

    for layer in model.layers:
        layer.trainable = False

    for i in range(-5, 0):
        model.layers[i].trainable = True


    model.compile(
        loss='categorical_crossentropy',
        optimizer=SGD(),
    )

    model.fit_generator(
        train_generator,
        steps_per_epoch=np.ceil(float(len(train_y)) / float(warmup_batch_size)),
        epochs=2,
        workers=12,
        use_multiprocessing=True,
        verbose=1
    )

    for layer in model.layers:
        layer.trainable = True

    callbacks_list = [csv_logger, reducelr, early, qwk]
    model.compile(
        loss='categorical_crossentropy',
        optimizer=SGD(),
        metrics=['accuracy']
    )

    model.fit_generator(
        train_mixup,
        steps_per_epoch=np.ceil(float(len(train_x)) / float(batch_size)),
        validation_data=valid_generator,
        validation_steps=np.ceil(float(len(valid_x)) / float(batch_size)),
        epochs=epochs,
        verbose=1,
        workers=12,
        use_multiprocessing=True,
        callbacks=callbacks_list
    )
