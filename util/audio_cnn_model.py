
from __future__ import print_function
import pandas as pd
import boto
import os
import numpy as np
from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

"""The 1D CNN Model used to predict severity score of depressed patients and depression label. It is the part of multi fusion model.CNN used to classify spectrograms of normal participants (0) or depressed
participants (1). 
"""

prefix = '/content/drive/MyDrive/Colab Notebooks/Depression Detection Code Workspace/Dataset/'
df_train = pd.read_csv(prefix+'train_split_Depression_AVEC2017.csv')

df_test = pd.read_csv(prefix+'dev_split_Depression_AVEC2017.csv')

df_dev = pd.concat([df_train, df_test], axis=0)

K.set_image_data_format('channels_first')
np.random.seed(15)

def preprocess(X_train, X_test):
    """
    Convert from float64 to float32 and normalize normalize to decibels
    relative to full scale (dBFS) for the 4 sec clip.
    """
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    X_train = np.array([(X - X.min()) / (X.max() - X.min()) for X in X_train])
    X_test = np.array([(X - X.min()) / (X.max() - X.min()) for X in X_test])
    return X_train, X_test

def prep_train_test(X_train, y_train, X_test, y_test, nb_classes):
    """
    Prep samples ands labels for Keras input by noramalzing and converting
    labels to a categorical representation.
    """
    print('Train on {} samples, validate on {}'.format(X_train.shape[0],
                                                       X_test.shape[0]))
    X_train, X_test = preprocess(X_train, X_test)
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)

    return X_train, X_test, Y_train, Y_test

def keras_img_prep(X_train, X_test, img_dep, img_rows, img_cols):
    """
    Reshape feature matrices for Keras' expexcted input dimensions.
    For 'th' (Theano) dim_order, the model expects dimensions:
    (# channels, # images, # rows, # cols).
    """
    if K.image_data_format() == 'channels_first':
        X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
        X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
        X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)
    return X_train, X_test, input_shape

def cnn(X_train, y_train, X_test, y_test, batch_size,
        nb_classes, epochs, input_shape):
    """
    The Convolutional Neural Net architecture for classifying the audio clips
    as normal (0) or depressed (1).
    """
    model = Sequential()

    model.add(Conv2D(32, (3, 3), padding='valid', strides=1,
                     input_shape=input_shape, activation='relu'))

    model.add(MaxPooling2D(pool_size=(4, 3), strides=(1, 3)))

    model.add(Conv2D(32, (1, 3), padding='valid', strides=1,
              input_shape=input_shape, activation='relu'))

    model.add(MaxPooling2D(pool_size=(1, 3), strides=(1, 3)))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adadelta',
                  metrics=['accuracy'])

    history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs,
                        verbose=1, validation_data=(X_test, y_test))
    score_train = model.evaluate(X_train, y_train, verbose=0)
    print('Train accuracy:', score_train[1])
    score_test = model.evaluate(X_test, y_test, verbose=0)
    print('Test accuracy:', score_test[1])

    return model, history

def model_performance(model, X_train, X_test, y_train, y_test):
    """
    Evaluation metrics for network performance.
    """
    y_test_pred = model.predict_classes(X_test)
    y_train_pred = model.predict_classes(X_train)
    y_test_pred_proba = model.predict_proba(X_test)
    y_train_pred_proba = model.predict_proba(X_train)
    y_test_1d = y_test[:, 1]
    conf_matrix = standard_confusion_matrix(y_test_1d, y_test_pred)
    print("Confusion Matrix:")
    print(conf_matrix)

    return y_train_pred, y_test_pred, y_train_pred_proba, \
        y_test_pred_proba, conf_matrix

def standard_confusion_matrix(y_test, y_test_pred):
    """
    Make confusion matrix with format:
                  -----------
                  | TP | FP |
                  -----------
                  | FN | TN |
                  -----------

    """
    [[tn, fp], [fn, tp]] = confusion_matrix(y_test, y_test_pred)
    return np.array([[tp, fp], [fn, tn]])

"""
Plots of test/train accuracy, loss, ROC curve.
"""


def plot_accuracy(history):
    """
    Plots train and test accuracy for each epoch.
    """
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('/content/drive/MyDrive/Colab Notebooks/Depression Detection Code Workspace/Experiments/cnn_accuracy1.png')
    plt.close()


def plot_loss(history):
    """
    Plots train and test loss for each epoch.
    """
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('/content/drive/MyDrive/Colab Notebooks/Depression Detection Code Workspace/Experiments/cnn_loss1.png')
    plt.close()


def plot_roc_curve(y_test, y_score):
    """
    Plots ROC curve for final trained model. Code taken from:
    https://vkolachalama.blogspot.com/2016/05/keras-implementation-of-mlp-neural.html
    """
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic curve')
    plt.legend(loc="lower right")
    plt.savefig('/content/drive/MyDrive/Colab Notebooks/Depression Detection Code Workspace/Experiments/cnn_roc1.png')
    plt.close()

if __name__ == '__main__':
    prefix="/content/drive/MyDrive/Colab Notebooks/Depression Detection Code Workspace/randomly_sampled_data/"
    prefix1="/content/drive/MyDrive/Colab Notebooks/Depression Detection Code Workspace/Experiments/"
    print('Retrieving from drive...')
    X_train = np.load(prefix+"train_samples.npz")
    y_train = np.load(prefix+'train_labels.npz')
    X_test = np.load(prefix+'test_samples.npz')
    y_test = np.load(prefix+'test_labels.npz')

    X_train, y_train, X_test, y_test = \
        X_train['arr_0'], y_train['arr_0'], X_test['arr_0'], y_test['arr_0']

    # CNN parameters
    batch_size = 64
    nb_classes = 2
    epochs = 25

    print('Processing images for Keras...')
    X_train, X_test, y_train, y_test = prep_train_test(X_train, y_train,
                                                       X_test, y_test,
                                                       nb_classes=nb_classes)
    img_rows, img_cols, img_depth = X_train.shape[1], X_train.shape[2], 1
    X_train, X_test, input_shape = keras_img_prep(X_train, X_test, img_depth,
                                                  img_rows, img_cols)
    print('Fitting model...')
    model, history = cnn(X_train, y_train, X_test, y_test, batch_size,
                         nb_classes, epochs, input_shape)
    print('Evaluating model...')
    y_train_pred, y_test_pred, y_train_pred_proba, y_test_pred_proba, \
        conf_matrix = model_performance(model, X_train, X_test, y_train, y_test)
    print('Saving model locally...')
    model_name = prefix1+'cnn_audio1.h5'
    model.save(model_name)
    print('Calculating additional test metrics...')
    accuracy = float(conf_matrix[0][0] + conf_matrix[1][1]) / np.sum(conf_matrix)
    precision = float(conf_matrix[0][0]) / (conf_matrix[0][0] + conf_matrix[0][1])
    recall = float(conf_matrix[0][0]) / (conf_matrix[0][0] + conf_matrix[1][0])
    f1_score = 2 * (precision * recall) / (precision + recall)
    print("Accuracy: {}".format(accuracy))
    print("Precision: {}".format(precision))
    print("Recall: {}".format(recall))
    print("F1-Score: {}".format(f1_score))
    print('Saving plots...')
    plot_loss(history)
    plot_accuracy(history)
    plot_roc_curve(y_test[:, 1], y_test_pred_proba[:, 1])

