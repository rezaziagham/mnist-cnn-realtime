import numpy as np

def load_data(path='data/mnist.npz'):
    with np.load(path) as data:
        x_train, y_train = data['x_train'], data['y_train']
        x_test, y_test = data['x_test'], data['y_test']
    return (x_train, y_train), (x_test, y_test)

def preprocess_data(x, y):
    x = x.astype('float32') / 255.0
    x = x.reshape(-1, 28, 28, 1)
    return x, y
