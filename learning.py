import numpy as np
import matplotlib.pyplot as plt
from typing import Iterable, Union
import pickle

def fit(model, X_train: Iterable, y_train: Iterable, 
                      X_test: Iterable, y_test: Iterable,
                      batch_size: int, no_epochs: int, 
                      verbosity = 1):
    
    # Fit data to model
    history = model.fit(X_train, y_train,
                batch_size=batch_size,
                epochs=no_epochs,
                verbose=verbosity,
                validation_data=(X_test, y_test))
    return model, history


def evaluate(model, X_val: Iterable, y_val: Iterable, verbose = 1):
    # Generate generalization metrics
    score = model.evaluate(X_val, y_val, verbose=0)
    if verbose:
        print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')
    return score

def plot_history(history, loss_fun: str):
    # Plot history: Categorical crossentropy & Accuracy
    plt.figure(figsize=(12,8))
    plt.plot(history.history['loss'], label=loss_fun + '(training data)')
    plt.plot(history.history['val_loss'], label=loss_fun + '(validation data)')
    plt.plot(history.history['acc'], label='Accuracy (training data)')
    plt.plot(history.history['val_acc'], label='Accuracy (validation data)')
    plt.title('Model performance')
    plt.ylabel('Loss value')
    plt.xlabel('No. epoch')
    plt.legend(loc="upper left")
    plt.show()
    
def create_model_LSTM(sample_shape: Iterable,
                 no_classes: int, kernel: Iterable,
                 loss_fun = 'binary_crossentropy'):
    model = Sequential()
    model.add(ConvLSTM2D(filters = 15, kernel_size = kernel, activation='tanh', dropout = 0.4, return_sequences = True, input_shape = sample_shape,data_format='channels_last',recurrent_activation='hard_sigmoid'))
    # model.add(BatchNormalization())
    # model.add(SpatialDropout3D(1))
    # model.add(Dense(10, activation='tanh'))
    # model.add(ConvLSTM2D(filters = 20, kernel_size = (1,1), activation='tanh', dropout = 0.5, return_sequences = True))
    # model.add(BatchNormalization())
    model.add(Flatten())
    # model.add(Dense(256, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(no_classes, activation='softmax'))
    model.compile(loss=loss_fun, optimizer='adam', metrics=['accuracy'])
    return model