import math
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pygame
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Activation
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.utils.np_utils import to_categorical


def train_model(trainFile, testFile):
    train = pd.read_csv(trainFile)

    f, ax = plt.subplots(5, 5)
    for i in range(1, 26):
        data = train.iloc[i, 1:785].values
        nrows, ncols = 28, 28
        grid = data.reshape((nrows, ncols))
        n = math.ceil(i / 5) - 1
        m = [0, 1, 2, 3, 4] * 5
        ax[m[i - 1], n].imshow(grid)

    dataTest = pd.read_csv(testFile)
    trainNumbers = train['label']
    train = train.drop(labels=['label'], axis=1)
    trainNumbers = to_categorical(trainNumbers, num_classes=10)

    train = train / 255
    test = dataTest / 255
    train = train.values.reshape(-1, 28, 28, 1)
    test = test.values.reshape(-1, 28, 28, 1)

    model = Sequential()
    model.add(Conv2D(24, (3, 3), padding='same', input_shape=(28, 28, 1)))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    datagen = ImageDataGenerator(
        featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        rotation_range=5,
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=False,
        vertical_flip=False)

    datagen.fit(train)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    model.fit(train, trainNumbers, epochs=5, batch_size=120)
    return model


def make_predict(model):
    img = mpimg.imread('image.png')[..., 1]
    img = cv2.resize(img, dsize=(28, 28))
    img = np.array(img).reshape(1, 28, 28, 1)
    pred = model.predict(img)
    print(pred)
    return np.argmax(pred, axis=1)[0]


def check_model(model):
    pygame.init()
    scr = pygame.display.set_mode((600, 400))
    scr.fill((0, 0, 0))
    pygame.display.update()
    txt = pygame.font.Font(None, 30)
    clock = pygame.time.Clock()
    FPS = 60

    p = True
    checker = False
    while p:
        for i in pygame.event.get():
            if i.type == pygame.QUIT:
                p = False
            if i.type == pygame.MOUSEBUTTONDOWN:
                if i.button == 1:
                    checker = True
                if i.button == 3:
                    pygame.image.save(scr, 'image.png')
                    prediction = make_predict(model)
                    text = txt.render('Prediction:{0}'.format(prediction), True, (255, 255, 255))
                    scr.blit(text, (16, 16))
                    pygame.display.update()
            if i.type == pygame.KEYDOWN:
                if i.key == pygame.K_SPACE:
                    scr.fill((0, 0, 0))
            if i.type == pygame.MOUSEBUTTONUP:
                if i.button == 1:
                    checker = False
            if i.type == pygame.MOUSEMOTION and checker == True:
                pygame.draw.circle(scr, (255, 255, 255), i.pos, 10)

        clock.tick(FPS)
        pygame.display.update()


if __name__ == '__main__':
    # train = r'../Neironka/train.csv'
    # test = r'../Neironka/test.csv'
    # model = train_model(train, test)
    # model.save('train_model.h5')

    model = tf.keras.models.load_model('train_model.h5')
    check_model(model)
