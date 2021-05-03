# -*- coding: utf-8 -*-
import os, glob, numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import tensorflow as tf
import set_data


def normalize_data():
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.compat.v1.Session(config=config)

    X_train, X_test, y_train, y_test = np.load('./numpy_data/multi_image_data.npy', allow_pickle=True)
    print(X_train.shape)
    print(X_train.shape[0])

    # 일반화
    X_train = X_train.astype(float) / 255
    X_test = X_test.astype(float) / 255
    return X_train, X_test, y_train, y_test


def train_model():
    X_train, X_test, y_train, y_test = normalize_data()

    categories, test_for_categories_list = set_data.get_categories()
    nb_classes = len(categories)

    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding="same", input_shape=X_train.shape[1:], activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding="same", activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model_dir = './model'

    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    model_path = model_dir + '/multi_img_classification.model'
    checkpoint = ModelCheckpoint(filepath=model_path, monitor='val_loss', verbose=1, save_best_only=True)
    early_stopping = EarlyStopping(monitor='val_loss', patience=6)

    model.summary()

    # 데이터셋이 적어서 validation을 그냥 test 데이터로 했습니다.
    # 데이터셋이 충분하시면 이렇게 하시지 마시고 validation_split=0.2 이렇게 하셔서 테스트 셋으로 나누시길 권장합니다.
    # history = model.fit(X_train, y_train, batch_size=32, epochs=50, validation_data=(X_test, y_test), callbacks=[checkpoint, early_stopping])
    history = model.fit(X_train, y_train, batch_size=32, epochs=50, validation_split=0.2,
                        callbacks=[checkpoint, early_stopping])

    print("정확도 : %.4f" % (model.evaluate(X_test, y_test)[1]))
    return history


def show_graph(history):

    y_vloss = history.history['val_loss']
    y_loss = history.history['loss']

    x_len = np.arange(len(y_loss))

    plt.plot(x_len, y_vloss, marker='.', c='red', label='val_set_loss')
    plt.plot(x_len, y_loss, marker='.', c='blue', label='train_set_oss')
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.grid()
    plt.show()


# 메인함수
if __name__ == '__main__':
    history = train_model()
    show_graph(history)