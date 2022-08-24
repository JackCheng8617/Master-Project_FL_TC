
import numpy as np
import random
# import cv2
import os
# from imutils import paths
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K

print('------------------------------------------------------------')

import itertools
import logging
import tensorflow as tf
import keras.backend as K
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
import numpy as np
import pandas as pd
import six
from keras import optimizers, regularizers
from keras.callbacks import *
from tensorflow.keras.layers import (LSTM, Activation, Add, Conv1D, Conv2D, Dense,
                          Dropout, Embedding, Flatten, GlobalMaxPooling1D,
                          GlobalMaxPooling2D, Input, MaxPooling1D,
                          MaxPooling2D, Reshape, SimpleRNN, UpSampling1D)
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.models import Model, Sequential, load_model
from keras.utils.vis_utils import plot_model
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from tensorflow.python import debug as tf_debug
from sklearn.metrics import classification_report
import time
import datetime
from sklearn import feature_extraction
from sklearn.compose import ColumnTransformer
from tensorflow.keras import regularizers
from sklearn.utils.multiclass import unique_labels
from sklearn.preprocessing import label_binarize
from sklearn import metrics
import gc


#initialize global model
def buildMLPModel(shape, classes, optimizer):
    # model = Sequential()
    # model.add(Dense(256, input_shape=(int(shape),), name='Input'))
    # model.add(LeakyReLU(alpha=0.2))
    # model.add(Dense(256, name='Hiden_1'))
    # model.add(LeakyReLU(alpha=0.2))
    # model.add(Dropout(0.2, name='Dropout_1'))
    # model.add(Dense(256, name='Hiden_2'))
    # model.add(LeakyReLU(alpha=0.2))
    # model.add(Dropout(0.2, name='Dropout_2'))
    # model.add(Dense(256,  name='Hiden_3'))
    # model.add(LeakyReLU(alpha=0.2))
    # model.add(Dropout(0.2, name='Dropout_3'))
    # model.add(Dense(classes, activation='softmax', name='Softmax_classifier'))
    # adam = optimizers.Adam(lr=0.01)
    # model.compile(loss='categorical_crossentropy', 
    #             optimizer=optimizer, metrics=['categorical_accuracy'])
    # model.summary()
    # return model
    model = Sequential()
    model.add(Dense(256, input_shape=(int(shape),), name='Input'))
    model.add(Dense(256, activation='relu', name='Hiden_1'))
    model.add(Dropout(0.2, name='Dropout_1'))
    model.add(Dense(256, activation='relu', name='Hiden_2'))
    model.add(Dropout(0.2, name='Dropout_2'))
    model.add(Dense(256, activation='relu', name='Hiden_3'))
    model.add(Dropout(0.2, name='Dropout_3'))
    model.add(Dense(classes, activation='softmax', name='Softmax_classifier'))
    # adam = optimizers.Adam(lr=0.001)
    model.compile(loss='categorical_crossentropy', 
                optimizer=optimizer, metrics=['categorical_accuracy'])
    model.summary()
    return model

def buildCNNModel(shape, classes, optimizer):
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=2, padding='Same', 
                    activation='relu', input_shape=(int(shape), 1),
                    name='Conv_1'))
    model.add(MaxPooling1D(2, name='Maxpool_1'))
    model.add(Conv1D(filters=64, kernel_size=2, padding='Same',
                    activation='relu', name='Conv_2'))
    model.add(MaxPooling1D(2, name='Maxpool_2'))

    model.add(Flatten(name='Full_connect'))
    model.add(Dense(64, activation='relu', name='relu_FC'))
    model.add(Dense(classes, activation='softmax', name='Softmax_classifier'))
    
    # adam = optimizers.Adam(lr=0.01)#lr=0.01
    model.compile(loss='categorical_crossentropy', 
                optimizer=optimizer, metrics=['categorical_accuracy'])
    model.summary()
    
    return model

def buildLSTMModel(shape, classes, optimizer):
    model = Sequential()
    model.add(LSTM(64, dropout=0.2, recurrent_dropout=0.2, input_shape=(int(shape), 1)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.25, name='Dropout_1'))
    model.add(Dense(classes, activation='softmax', name='Softmax_classifier'))
    # rmSprop = optimizers.RMSprop(lr=0.001)
    model.compile(loss='categorical_crossentropy', 
                optimizer=optimizer, metrics=['categorical_accuracy'])
    model.summary()
    return model

def buildSAEModel(shape, classes, optimizer):
    model = Sequential()
    model.add(Dense(128, activation='relu', input_shape=(int(shape),), name='AE1_input'))
    model.add(Dense(64, activation='relu', name='AE1_hiden'))
    model.add(Dropout(0.2, name='Dropout_1'))
    model.add(Dense(64, activation='relu', name='AE2_input'))
    model.add(Dense(32, activation='relu', name='AE2_hiden'))
    model.add(Dense(64, activation='relu', name='AE2_output'))
    model.add(Dropout(0.2, name='Dropout_2'))
    
    model.add(Dense(classes, activation='softmax', name='Softmax_classifier'))
    
    # adam = optimizers.Adam(lr=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['categorical_accuracy'])
    
    model.summary()

    return model
    

# def weight_scalling_factor(clients_trn_data, client_name):
#     print('@@@@@@@ weight scalling factor @@@@@@@@@@')
#     client_names = list(clients_trn_data.keys())
#     print('Client naems: ', client_names)
#     #get the bs
#     bs = list(clients_trn_data[client_name])[0][0].shape[0]
#     print('Bs: ', bs)
#     #first calculate the total training data points across clinets
#     global_count = sum([tf.data.experimental.cardinality(clients_trn_data[client_name]).numpy() for client_name in client_names])*bs
#     print('Global count: ', global_count)
#     # get the total number of data points held by a client
#     local_count = tf.data.experimental.cardinality(clients_trn_data[client_name]).numpy()*bs
#     print('Local count: ', local_count)
#     print(local_count/global_count)

#     return local_count/global_count


def scale_model_weights(weight, scalar):
    '''function for scaling a models weights'''
    print('@@@@@@@@@@@@@@ scale model weights @@@@@@@@')
    weight_final = []
    steps = len(weight)
    print('Steps: ', steps)
    for i in range(steps):
        weight_final.append(scalar * weight[i])
        print('Weight final:', weight_final)
        print('-----------------------------------')
    return weight_final



def sum_scaled_weights(scaled_weight_list):
    '''Return the sum of the listed scaled weights. The is equivalent to scaled avg of the weights'''
    print('@@@@@@@@@@@ sum scaled weights @@@@@@@@@@@@')
    avg_grad = list()
    #get the average grad accross all client gradients
    for grad_list_tuple in zip(*scaled_weight_list):
        layer_mean = tf.math.reduce_sum(grad_list_tuple, axis=0) ###將每個陣列的相對位置相加(因為axis=0)
        avg_grad.append(layer_mean)
        print('layer_mean', layer_mean)
        print('############################################')
        
    return avg_grad

def plotConfusionMatrix(y_test, predict, classes, normalize=False, title='Confusion matrix', fileName='confusion_matrix', cmap=plt.cm.Blues):

    cm = confusion_matrix(y_test, predict)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    
    plt.figure()
    plt.imshow(cm, cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
            ha="center", va="center",
            color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.rcParams.update({'font.size': 8})
    plt.tight_layout()
    plt.savefig(fileName)
    # plt.show()

def plot_confusion_matrix(y_true, y_pred, classes, normalize=False, title='Confusion matrix', fileName='confusion_matrix', cmap=plt.cm.Blues):
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    cm = confusion_matrix(y_true, y_pred)

    # mylist = ['0','1','2']
    # matrics = pd.DataFrame(cm, columns=mylist, index=mylist)
    matrics = pd.DataFrame(cm)
    matrics.to_csv(fileName + '_matrics.csv')

    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    fig, ax = plt.subplots(figsize = (15,15))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    fig.savefig(fileName)
    return ax

def historyDisplay(trainHistory, eval_train, eval_val, trainMetrics, validationMetrics, title, yLabelName, xLabelName, fileName):
    plt.subplots()
    plt.plot(trainHistory.history[trainMetrics])
    plt.plot(trainHistory.history[validationMetrics])
    plt.title(title)
    plt.ylim(0,1.05)
    plt.ylabel(yLabelName)
    plt.xlabel(xLabelName)
    plt.legend(['training', 'validation'], loc='best')
    plt.text(20, 0.4, 'Traning ' + trainMetrics + ':' + "{:.3%}".format(eval_train), fontsize=12)
    plt.text(20, 0.3, 'Traning ' + validationMetrics + ':' + "{:.3%}".format(eval_val), fontsize=12)
    plt.tight_layout()
    plt.savefig(fileName)
    # plt.show()    

def evaluation_model(X_test1, Y_test1, model, labelEncoder, modelFileName, test_mode, name, comm_round): #name可以是client名或global

    if test_mode == 'Train':
        loss, accuracy = model.evaluate(X_test1, Y_test1, verbose=0)

    elif test_mode == 'Val':
        loss, accuracy = model.evaluate(X_test1, Y_test1, verbose=0)

    elif test_mode == 'Test':
        loss, accuracy = model.evaluate(X_test1, Y_test1, verbose=0)
        print("\nTesting Loss: %f, Accuracy: %f" % (loss, accuracy))
        predict = model.predict(X_test1)

        classes = predict.argmax(axis=-1)
        y_test_l = Y_test1.argmax(axis=-1)

        if name == 'Global':
            plot_confusion_matrix(y_test_l, classes, classes=labelEncoder.classes_, title= comm_round + '回_' + name + '_' + modelFileName + '_Confusion matrix', fileName= comm_round + '回_' + name + '_' + modelFileName + '_Confusion matrix')

            report = classification_report(y_test_l, classes, target_names=labelEncoder.classes_, digits=4)
            print(report)

            evaluation_self(y_test_l, classes, labelEncoder, modelFileName, name, comm_round, loss)

        else:
            plot_confusion_matrix(y_test_l, classes, classes=labelEncoder.classes_, title=  name + '_' + comm_round + '回_' + modelFileName + '_Confusion matrix', fileName= name + '_' + comm_round + '回_' +  modelFileName + '_Confusion matrix')

            report = classification_report(y_test_l, classes, target_names=labelEncoder.classes_, digits=4)
            print(report)

            evaluation_self(y_test_l, classes, labelEncoder, modelFileName, name, comm_round, loss)
            
       
    return accuracy, loss

def evaluation_self(y_test_l, classes, labelEncoder, modelFileName, name, comm_round, loss):
    
    y = label_binarize(y_test_l, classes=[0,1,2,3,4,5,6,7,8])
    # print('-------------------------------------------')
    # print(y)
    classes = label_binarize(classes, classes=[0,1,2,3,4,5,6,7,8])
    n_classes = 9

    # print(y)

    precision = dict()
    recall = dict()
    f1 = dict()
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    acc = dict()
    all_AUC = 0.0
    confusion_metrics = pd.DataFrame(columns=['Lable','AUC', 'Precision', 'Recall', 'F1-score'])
    for i in range(n_classes):
        # print(labelEncoder.inverse_transform([i]))
        lable = labelEncoder.inverse_transform([i])

        precision[i] = metrics.precision_score(y[:, i], classes[:, i])
        
        recall[i] = metrics.recall_score(y[:, i], classes[:, i])
        
        f1[i] = metrics.f1_score(y[:, i], classes[:, i])
        
        fpr[i], tpr[i], _ = metrics.roc_curve(y[:, i], classes[:, i])
        roc_auc[i] = metrics.auc(fpr[i], tpr[i])
        all_AUC = all_AUC + roc_auc[i]

        # acc[i] = metrics.accuracy_score(y[:, i], classes[:, i])
        
        precision_array = np.array([precision[i]])
        recall_array = np.array([recall[i]])
        f1_array = np.array([f1[i]])
        roc_auc_array= np.array([roc_auc[i]])
        # acc_array = np.array([acc[i]])
        
        a = np.concatenate((lable, np.round(roc_auc_array,4), np.round(precision_array,4), np.round(recall_array,4), np.round(f1_array,4)), axis=None)
        # print(a.shape)
        a = np.reshape(a ,(1,5))
        s = pd.DataFrame(a, columns=['Lable','AUC', 'Precision', 'Recall', 'F1-score'])
        
        confusion_metrics = confusion_metrics.append(s, ignore_index=True)

    all_AUC = all_AUC/n_classes
    all_Acc = metrics.accuracy_score(y, classes)
    all_Precision = metrics.precision_score(y, classes, average='macro')
    all_Recall = metrics.recall_score(y, classes, average='macro')
    all_F1 = metrics.f1_score(y, classes, average='macro')
    all_Loss = loss


    confusion_metrics['All_AUC'] = np.round(all_AUC,4)
    confusion_metrics['All_Acc'] = np.round(all_Acc,4)
    confusion_metrics['All_Precision'] = np.round(all_Precision,4)
    confusion_metrics['All_Recall'] = np.round(all_Recall,4)
    confusion_metrics['All_F1'] = np.round(all_F1,4)
    confusion_metrics['all_Loss'] = np.round(all_Loss,4)

    print(confusion_metrics)
    if name == 'Global':
        confusion_metrics.to_csv(str(comm_round) + '回_' + name + '_' + modelFileName + '_Caculate metrics.csv', index=False)
    else:
        confusion_metrics.to_csv( name + '_' + str(comm_round) + '回_' + modelFileName + '_Caculate metrics.csv', index=False)

def check_exists(client):
    if os.path.exists('last_client_' + str(client+1) + '_weights.h5'):
        fileExist = os.path.exists('last_client_' + str(client+1) + '_weights.h5')
    else:
        fileExist = os.path.exists('client_' + str(client+1) + '_weights.h5')
    return fileExist