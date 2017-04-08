# Machine Learning Playground

## Structure

1. frontend.py: the flask frontend app to serve a simple UI to play with
2. other py files: the models

## How to run

`python frontend.py` will start the local server and point your browser to http://localhost:5000 to play with.

## Models

1. captcha-cnn.py
    * using `captcha` to generate the training data(each image contains 1 digit from 0 to 9) to train the model and save as h5 file
    * for the web UI, it will generate a new captcha on the fly and try to predict using the trained model
