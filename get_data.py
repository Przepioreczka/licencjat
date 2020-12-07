import numpy as np
from tensorflow.keras.utils import to_categorical


def get_training(X, y, combinations, pic_mode):
    X_train = np.concatenate([X[ind] for ind in combinations], axis = 1)
    X_train = np.einsum(pic_mode, X_train)
    y_train = np.concatenate([y[ind] for ind in combinations])
    y_train = to_categorical(y_train).astype(int)
    return X_train, y_train

def get_test(X, y, ind, pic_mode):
    X_test = np.einsum(pic_mode, X[ind])
    y_test = to_categorical(y[ind]).astype(int)
    return X_test, y_test
