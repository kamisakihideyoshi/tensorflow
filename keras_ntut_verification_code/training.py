from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.layers import Conv1D, Conv2D
from keras.layers import MaxPooling1D, MaxPooling2D
from sklearn.model_selection import train_test_split

import string
import os
import keras
import pandas as pd
import numpy as np
import cv2

shape = (40, 40, 1)
class_num = len(string.ascii_uppercase)

# 強制 numpy 顯示完整的陣列
np.set_printoptions(threshold=np.inf)


def build(model):
    '''建立模型'''
    model.add(MaxPooling2D(pool_size=(2, 2), input_shape=shape))
    model.add(Conv2D(64, (4, 4), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (4, 4), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(128, (4, 4), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(1024))
    model.add(Activation('relu'))
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dense(class_num))
    model.add(Activation('softmax'))
    model.summary()
    return model


def read_image(path):
    '''從指定路徑讀取圖片
    path -- 圖片的路徑 (包含副檔名)
    '''
    im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    im = im.reshape(im.shape[0], im.shape[1], 1)
    # 將圖片二值化xxx
    im //= 255
    return im


def read_data():
    '''從 csv 檔讀取 feature 跟 label
    csv -- 指定的 csv 檔案路徑 (包含副檔名)
    '''
    data = pd.DataFrame()
    for t in string.ascii_uppercase:
        image_path = 'testing/'+t+'/'
        print('Loading images from:', image_path)
        dir_list = os.listdir(image_path)
        images = [image_path + image_name for image_name in dir_list if image_name[-4:] == '.png']
        d = pd.DataFrame(zip(images, [ord(t)-65]*len(images)), columns=['feature', 'label'])
        data = data.append(d)

    print('Data shuffling')
    # 打亂 Dataset 的順序
    data = data.sample(frac=1)
    feature = [read_image(path) for path in data['feature'].values]
    feature = np.asarray(feature)
    label = data['label'].values
    return feature, label


def train(model, feature, label, epochs):
    '''訓練指定的模型，並進行驗證'''
    # 分割 Dataset
    train_feature, test_feature, train_label, test_label = train_test_split(
        feature, label, test_size=0.2)

    train_label = keras.utils.to_categorical(train_label, class_num)
    test_label = keras.utils.to_categorical(test_label, class_num)

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    model.fit(train_feature, train_label,
              batch_size=300, epochs=epochs, verbose=1, validation_split=0.2, shuffle=True)

    while input('One more?') == '':
        model.fit(train_feature, train_label,
              batch_size=300, epochs=1, verbose=1, shuffle=True)

    result = model.evaluate(test_feature, test_label, verbose=1)
    print('Test lost:', result[0], ', accuracy:', result[1])
    return model


def main():
    feature, label = read_data()
    model = Sequential()
    model = build(model)
    model = train(model, feature, label, epochs=5)
    model.save('nportal_splitted.h5')


if __name__ == '__main__':
    main()
