from captcha.image import ImageCaptcha
import numpy as np
from PIL import Image
import random
import uuid
import os
import keras
from keras.models import Sequential
from keras.layers import Dropout, Dense, Flatten, Conv2D, MaxPooling2D

# global settings
batch_size = 128
epochs = 12
width = 60
height = 100
number_classes = 10
weights_h5_file = "weights.h5"
input_shape = (width, height, 1)

def _gen_one_captcha(folder):
    image = ImageCaptcha(width=width, height=height)
    d = random.randint(0, 9)
    img = image.generate_image(str(d))
    img = img.convert('L')
    guid = uuid.uuid4()
    path = "{}/{}-{}.png".format(folder, d, guid)
    img.save(path)
    return path

def gen_captcha():
    for i in range(10000):
        _gen_one_captcha("train")
    for i in range(1000):
        _gen_one_captcha("test")

def load_data():
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    for f in os.listdir("train/"):
        im = Image.open(os.path.join("train", f))
        x_train.append(np.asarray(im, dtype=np.float32))
        y_train.append(f.split('-')[0])
    for f in os.listdir("test/"):
        im = Image.open(os.path.join("test", f))
        x_test.append(np.asarray(im, dtype=np.float32))
        y_test.append(f.split('-')[0])
    return (np.asarray(x_train), np.asarray(y_train)), (np.asarray(x_test), np.asarray(y_test))

def build_model():
    layers = [
        Conv2D(64, (3, 3), activation='relu', input_shape=input_shape),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2,2)),
        Dropout(.25),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(.5),
        Dense(number_classes, activation='softmax')
    ]
    model = Sequential(layers)
    model.compile(loss=keras.losses.categorical_crossentropy,
                optimizer=keras.optimizers.Adadelta(),
                metrics=["accuracy"])
    return model

def normalize_data():
    (x_train, y_train), (x_test, y_test) = load_data()
    x_train /= 255
    x_test /= 255

    x_train = x_train.reshape(x_train.shape[0], width, height, 1)
    x_test = x_test.reshape(x_test.shape[0], width, height, 1)

    y_train = keras.utils.to_categorical(y_train, number_classes)
    y_test = keras.utils.to_categorical(y_test, number_classes)
    return (x_train, y_train), (x_test, y_test)
    

def train(save_weights=True):

    (x_train, y_train), (x_test, y_test) = normalize_data()

    model = build_model()
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test))
    if save_weights:
        model.save_weights(weights_h5_file)
    return model

def evaluate(model):
    (x_train, y_train), (x_test, y_test) = normalize_data()
    score = model.evaluate(x_test, y_test, verbose=2)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

def predict(model, img):
    im = Image.open(img)
    data = np.asarray(im, dtype=np.float32)
    data = data.reshape(1, width, height, 1)
    return model.predict_classes(data)[0]
    
def _get_random_test_input():
    files = os.listdir("test/")
    return "test/{}".format(random.choice(files))

def load_trained_model():
    if os.path.exists("train/"):
        pass
    else:
        os.mkdir("train")
        os.mkdir("test")
        gen_captcha()
    if not os.path.exists(weights_h5_file):
        # means have the saved trained weights
        train()
    # load the trained weights
    model = build_model()
    model.load_weights(weights_h5_file)
    return model

if __name__ == "__main__":
    model = load_trained_model()
    print predict(model, _get_random_test_input())
