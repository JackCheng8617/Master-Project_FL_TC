import itertools
import logging
import tensorflow as tf
import keras.backend as K
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
import numpy as np
import pandas as pd
import six
from tensorflow.keras import optimizers, regularizers
from keras.callbacks import *
from keras.layers import (LSTM, Activation, Add, Conv1D, Conv2D, Dense,
                          Dropout, Embedding, Flatten, GlobalMaxPooling1D,
                          GlobalMaxPooling2D, Input, MaxPooling1D,
                          MaxPooling2D, Reshape, SimpleRNN, UpSampling1D)
                          
from tensorflow.keras.layers import BatchNormalization
from keras.models import Model, Sequential, load_model
from keras.utils.vis_utils import plot_model
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from tensorflow.python import debug as tf_debug
from sklearn.metrics import classification_report
import time
import datetime
from sklearn import feature_extraction
from sklearn.compose import ColumnTransformer
from keras import regularizers
from sklearn.utils.multiclass import unique_labels
from sklearn.preprocessing import label_binarize
from sklearn import metrics
from tensorflow.keras.layers import LeakyReLU

import os

#FORMAT = '[%(levelname)s]: %(asctime)-15s %(message)s'
#logging.basicConfig(format=FORMAT, level=logging.INFO)

def readTrain():

    trainFilePath = 'Datasets/141_train_1_(new).csv'
    # trainFilePath = '141_train_FilterIP_3_seperate(withoutHeader)_plus.csv'
    # trainFilePath = 'train.csv'
    columnName = [str(i) for i in range(10)] + ['type']
    train = pd.read_csv(trainFilePath, names=columnName)
    # train.drop(['0'], axis=1, inplace=True)

    return train

def oneHotLabel(train):

    return pd.get_dummies(train)

def buildTrain(train, labelEncoder):
    
    typeLabel = list()
    featuresName = list(train.columns[:-(len(labelEncoder.classes_))])
    [typeLabel.append('type_' + str(t)) for t in labelEncoder.classes_]
    X_train = np.array(train[:train.shape[0]][featuresName])
    Y_train = np.array(train[:train.shape[0]][typeLabel])
    
    return X_train, Y_train

def readTesting():

    # trainFilePath = '141_7_test_withOtherFeature_V2(seperate)_withoutHeader.csv'
    trainFilePath = 'Datasets/Cut_141_7_test_withOtherFeature_V2(seperate)_withoutHeader.csv' #important
    # trainFilePath = 'test.csv'
    columnName = [str(i) for i in range(int(10))] + ['type']
    test = pd.read_csv(trainFilePath, names=columnName)
    print(test.head())
    
    # test.drop(['0'], axis=1, inplace=True)

    return test

def buildTest(train, labelEncoder):

    typeLabel = list()
    featuresName = list(train.columns[:-(len(labelEncoder.classes_))])
    print(labelEncoder.classes_)
    [typeLabel.append('type_' + str(t)) for t in labelEncoder.classes_]
    print(labelEncoder.classes_)
    X_train = np.array(train[:train.shape[0]][featuresName])
    Y_train = np.array(train[:train.shape[0]][typeLabel])

    return X_train, Y_train


def shuffle(X,Y):
    
    np.random.seed(10)
    randomList = np.arange(X.shape[0])
    np.random.shuffle(randomList)
    
    return X[randomList], Y[randomList]

def splitData(X,rate, label):
    filter = (X['type'] == label)
    X = X[filter]

    Y_train = X[int(X.shape[0]*rate):]
    Y_val = X[:int(X.shape[0]*rate)]
    
    return Y_train, Y_val

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

def plot_confusion_matrix(y_true, y_pred, classes,
    normalize=False, title='Confusion matrix', fileName='confusion_matrix', cmap=plt.cm.Blues):
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    cm = confusion_matrix(y_true, y_pred)

    # mylist = ['0','1','2']
    # matrics = pd.DataFrame(cm, columns=mylist, index=mylist)
    # matrics = pd.DataFrame(cm)
    # matrics.to_csv('matrics.csv')

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

def buildMLPModel():
    model = Sequential()
    model.add(Dense(256, input_shape=(int(X_train.shape[1]),), name='Input'))#3073
    model.add(Dense(256, activation='relu', name='Hiden_1'))
    model.add(Dropout(0.2, name='Dropout_1'))
    model.add(Dense(256, activation='relu', name='Hiden_2'))
    model.add(Dropout(0.2, name='Dropout_2'))
    model.add(Dense(256, activation='relu', name='Hiden_3'))
    model.add(Dropout(0.2, name='Dropout_3'))
    model.add(Dense(25, activation='softmax', name='Softmax_classifier'))
    # model.add(Dense(27, activation='softmax', name='Softmax_classifier'))
    adam = optimizers.Adam(lr=0.001)
    model.compile(loss='categorical_crossentropy', 
                optimizer=adam, metrics=['categorical_accuracy'])
    model.summary()
    return model

def buildCNNModel():
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=2, padding='Same', 
                    activation='relu', input_shape=(int(X_train.shape[1]), 1),
                    name='Conv_1'))
    model.add(MaxPooling1D(2, name='Maxpool_1'))
    model.add(Conv1D(filters=64, kernel_size=2, padding='Same',
                    activation='relu', name='Conv_2'))
    model.add(MaxPooling1D(2, name='Maxpool_2'))

    model.add(Flatten(name='Full_connect'))
    model.add(Dense(64, activation='relu', name='relu_FC'))
    model.add(Dense(25, activation='softmax', name='Softmax_classifier'))
    
    adam = optimizers.Adam(lr=0.01)#lr=0.01
    model.compile(loss='categorical_crossentropy', 
                optimizer=adam, metrics=['categorical_accuracy'])
    model.summary()
    
    return model

def buildLSTMModel():
    model = Sequential()
    model.add(LSTM(64, dropout=0.2, recurrent_dropout=0.2, input_shape=(int(301+4-2), 1)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.25, name='Dropout_1'))
    model.add(Dense(7, activation='softmax', name='Softmax_classifier'))
    rmSprop = optimizers.RMSprop(lr=0.001)
    model.compile(loss='categorical_crossentropy', 
                optimizer=rmSprop, metrics=['categorical_accuracy'])
    model.summary()
    return model

def buildSAEModel():
    model = Sequential()
    model.add(Dense(128, activation='relu', input_shape=(int(X_train.shape[1]),), name='AE1_input'))
    model.add(Dense(64, activation='relu', name='AE1_hiden'))
    model.add(Dropout(0.2, name='Dropout_1'))
    model.add(Dense(64, activation='relu', name='AE2_input'))
    model.add(Dense(32, activation='relu', name='AE2_hiden'))
    model.add(Dense(64, activation='relu', name='AE2_output'))
    model.add(Dropout(0.2, name='Dropout_2'))
    
    model.add(Dense(25, activation='softmax', name='Softmax_classifier'))
    
    adam = optimizers.Adam(lr=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['categorical_accuracy'])
    
    model.summary()

    return model

def filter_report(now_list, GAN_epoch, pre_list, final_list, FL_epoch): #classify

    pre_list = now_list
    now_list = list()

    print('@@@@@@@@@ Read Testing Report @@@@@@@@@@@')
    #####第一次是要讀client所存的report
    #####第二次就是內部自己的report
    #####因此名稱需要明確
    ##### client : FL_次數_Report(第一次)
    ##### CTCGAN : FL_次數_GAN_次數_Report
    if GAN_epoch == 0:
        report = pd.read_csv('FL_'+ str(FL_epoch) + "_Report.csv", index_col=0)
    else:
        report = pd.read_csv('FL_' + str(FL_epoch) + '_GAN_' + str(GAN_epoch) + "_Report.csv", index_col=0)
    
    print(report)
    print('Labels : ', len(report))
    print('------------- Finished ----------------')
    print('')
    print('@@@@@@ Filter not Achieve Label @@@@@@@@')

    #filter 掉不重要的指標，以及篩選低於指定Recall數值的類別
    report = report.drop(['precision', 'f1-score', 'support'], axis=1)
    filter = (report['recall'] > 0.0000) & (report['recall'] <= 0.9000)
    report = report[filter]
    print(report)

    #################################################################
    # 從這裡去紀錄每次需要生成的類別，記錄他們的recall效能，並且如果有高於原本效能的就也記錄，等到最後所有類別都有達標過，都有專屬的生成資料後，
    # 就開始去做最後的總訓練，並將其中的效能不停重複訓練到紀錄的效能，達到才可以停下

    # report.to_csv(str(round) +'_GAN_report.csv') #需要生成的存成檔案
    print('@@@@@@@@@@ Transpose @@@@@@@@@@@@@') #將目前的
    report = pd.DataFrame(report).transpose()
    report = report.to_dict()
    print('Before, Not achieve Labels : ', len(report))
    print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
    print('')
    print('@@@@@@@@@@ Get Items Key @@@@@@@@@@@@')
    for total_key, value in report.items():
        now_list.append(total_key)
    print('No Saturation Labels: ', now_list)
    print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
    print('')
    print('@@@@@@@@@@@@ filter Saturation Label @@@@@@@@@@@@@')

    for i in pre_list: 
        if i not in now_list: #將比對list_1裡面的元素，是否有無在list_2中，沒有的就將其用final_list存下
            # print(i)
            final_list.append(i)

    for i in final_list: #比對list_2裡面的元素，是否與final_list重複，重複的話就刪除該元素
        if i in now_list:
            del report[i]
            now_list.remove(i)
    # print(report)
    print('Filtered Saturation Labels: ', now_list)
    print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')

    print('@@@@@@@@@@ Transpose @@@@@@@@@@@@@') #將目前的
    print('Dict report:')
    print(report)
    report = pd.DataFrame(report).transpose()
    print('DataFrame report:')
    print(report)

    ##### FL_次數_GAN_次數_GAN_report
    report.to_csv('FL_' + str(FL_epoch) + '_GAN_' + str(GAN_epoch) + '_GAN_report.csv') 
    print('After, Not achieve Labels : ', len(report))
    print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')

    time.sleep(5)
    # return report, now_list
    return now_list, pre_list, final_list

def record_best(best_reord, GAN_epoch):
    
    print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
    ##### client : FL_次數_Report(第一次)
    ##### CTCGAN : FL_次數_GAN_次數_Report
    if GAN_epoch == 0:
        sec_line = pd.read_csv('FL_'+ str(FL_epoch) + "_Report.csv", index_col=0)
    else:
        sec_line = pd.read_csv('FL_' + str(FL_epoch) + '_GAN_' + str(GAN_epoch) + "_Report.csv", index_col=0)
    # print(sec_line)
    sec_line.drop(['precision', 'f1-score', 'support'], axis=1, inplace=True)
    # print(sec_line)
    sec_line = pd.DataFrame(sec_line).transpose()
    sec_line = sec_line.to_dict()
    # print(sec_line)

    recall_sec = list()
    print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
    for total_key, value in sec_line.items():
        for key in value:
            # print(key + ': ' + str(value[key]))
            recall_sec.append(value[key])
    i = 0
    for total_key, value in best_reord.items():
        # print(total_key)
        for key in value:
            # print(key + ': ' + str(value[key]))
            if value[key] < recall_sec[i]:
                value[key] = recall_sec[i] 
            # print('-----------------------------')
            # print(key + ': ' + str(value[key]))
            # print('####')
        i = i + 1
    
    return best_record

def check_Train_time(now_list, final_list, GAN_epoch, stop_point):
    if GAN_epoch == stop_point:
        print('Now rounds: ', GAN_epoch)
        if now_list:
            for i in range(len(now_list)):
                final_list.append(now_list[i])
            print("It's already train for " + str(GAN_epoch) + " times! Now Breaking!")
        now_list.clear()
        time.sleep(3)
    return final_list, now_list

def Compare(best_recall, report):
    Is_best = True
    i = 0
    for total_key, value in report.items():
        print('@@@@@@@@@@ Compare @@@@@@@@@@@@@')
        print('Label: ', total_key)
        if total_key in final_list:
            for key in value:
                if (round(value[key],4)) < (round(best_recall[i], 4)):
                    print('The Recall: ', value[key])
                    print('--------------------------')
                    print('Best_record: ', best_recall[i])
                    Is_best = False
        i = i + 1
        print('@@@@@@@@@@@@@@@@@@@@@@@@@@@')
    return Is_best

def evaliation_self(y_test_l, classes, labelEncoder, FL_epoch, GAN_epoch, loss, last_training, final_GAN_epoch=1, during_best=1):
    #######################################################
    y = label_binarize(y_test_l, classes=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24])
    # print('-------------------------------------------')
    # print(y)
    classes = label_binarize(classes, classes=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24])
    n_classes = 25

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

    # print(confusion_metrics)
    ##改
    if last_training == 3: ## 3=有進入Is_Best循環訓練的過程，如果達到stop point又或是在過程中達到Is_Best都會將最後存成這個格式
        confusion_metrics.to_csv('FL_' + str(FL_epoch) + '_Final_During_Best_' + str(during_best) + '_caculate_metrics.csv', index=False)
    elif last_training == 2:## 2 = 在Is_Best中循環訓練時存的檔名
        confusion_metrics.to_csv('FL_' + str(FL_epoch) + '_Final_' + str(final_GAN_epoch) + '_caculate_metrics.csv', index=False)
    elif last_training == 1:## 1 = 還在GAN中間訓練時的存的檔名
        confusion_metrics.to_csv('FL_' + str(FL_epoch) + '_GAN_' + str(GAN_epoch) + '_caculate_metrics.csv', index=False)
    elif last_training == 0:##準備進入Is_Best中，但其實早已達標無須進入循環，因此直接存檔的檔名
        confusion_metrics.to_csv('FL_' + str(FL_epoch) + '_Final_caculate_metrics.csv', index=False)
    ###########################################
    
##改
def during_filter_report(final_GAN_epoch, Ftp_data, Snapchat, SoundCloud, eBay, report, during_best):
    
    if((report['recall'][3] >= Ftp_data) and (report['recall'][14] >= Snapchat) and (report['recall'][15] >= SoundCloud) and (report['recall'][24] >= eBay)):
        during_best = final_GAN_epoch
        Ftp_data = report['recall'][3]
        Snapchat = report['recall'][14]
        SoundCloud = report['recall'][15]
        eBay = report['recall'][24]
    
    print('Now Is_Best is: ', during_best)
    
    return during_best, Ftp_data, Snapchat, SoundCloud, eBay

"""combine with classify"""

# seed_value= 42
# import numpy as np
# np.random.seed(seed_value)
# import random
# random.seed(seed_value)


### Control GPU used memory
gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

fulltrain = pd.read_csv('fulltrain.csv')
fulltest = pd.read_csv('fulltest.csv')

train_X_train = pd.DataFrame()
train_X_val = pd.DataFrame()

for label in fulltrain.type.unique():
    train_train, train_val = splitData(fulltrain, 0.2, label)
    print('Label: ', label)
    print('Train: ', train_train.shape)
    print('Vali: ', train_val.shape)
    train_X_train = train_X_train.append(train_train, ignore_index=True)
    train_X_val = train_X_val.append(train_val, ignore_index=True)

# del fulltrain
# gc.collect()

print(train_X_train)
print(train_X_val)

print('-----------X_train----------------')
oneHotTrain = oneHotLabel(train_X_train)
labelEncoder = preprocessing.LabelEncoder()
labelEncoder.fit(fulltrain.type.unique())
X_train, Y_train = buildTrain(oneHotTrain, labelEncoder)
print('-----------ok---------------------')
print('')
print('-----------X_val------------------------')
oneHotTrain = oneHotLabel(train_X_val)
X_val, Y_val = buildTrain(oneHotTrain, labelEncoder)
print('------------ok------------------------')

X_train, Y_train = shuffle(X_train, Y_train)
X_val, Y_val = shuffle(X_val, Y_val)


del train_X_train
# gc.collect()
del train_X_val
# gc.collect()

logging.info((X_train.shape, Y_train.shape, X_val.shape, Y_val.shape))

parameters_Frame = pd.read_csv('parameters_Frame.csv')
print(parameters_Frame)
modelFileName = parameters_Frame['ModelFileName'][0]
feature_size = parameters_Frame['feature_size'][0]
local_epoch = parameters_Frame['local_epoch'][0]
local_batch_size = parameters_Frame['local_batch_size'][0]
lr = parameters_Frame['lr'][0]
labels = parameters_Frame['labels'][0]
GAN_train_epoch = parameters_Frame['GAN_train_epoch'][0]
stop_point = parameters_Frame['stop_point'][0]

oneHotTest = oneHotLabel(fulltest)
X_test1, Y_test1 = buildTest(oneHotTest, labelEncoder)
# X_test1 = scaler.transform(X_test1)
logging.info((X_test1.shape, Y_test1.shape))

if modelFileName == 'CNNModel':
    X_test1 = X_test1.reshape(X_test1.shape[0], X_test1.shape[1], 1)

FL_epoch = 0
GAN_epoch = 0

print('@@@ FL_count @@@@@@@@')
if os.path.exists('FL_count.txt'):
    f = open('FL_count.txt')
    FL_epoch = f.read()
    FL_epoch = int(FL_epoch)
    print(FL_epoch)
    f.close
print('@@@@@@@@@@@@@@@@@@@@')
print('@@@ GAN_count @@@@@@@@')

path = 'GAN_count.txt'
f = open(path, 'w')
f.write(str(GAN_epoch))
f.close()

# print('############ Check Time ##############')
# model_load = load_model('FL_'+ str(FL_epoch) + '_' + modelFileName + '.h5')
# adam = optimizers.Adam(lr=lr)
# model_load.compile(loss='categorical_crossentropy', 
#         optimizer=adam, metrics=['categorical_accuracy'])

# loss, accuracy = model_load.evaluate(X_test1, Y_test1)
# print("\nTesting Loss: %f, Accuracy: %f" % (loss, accuracy))

# predict = model_load.predict(X_test1)

# classes = predict.argmax(axis=-1)
# y_test_l = Y_test1.argmax(axis=-1)

# #### FL_次數_Final_...
# plot_confusion_matrix(y_test_l, classes, classes=labelEncoder.classes_, title='MLP Confusion matrix', fileName='FL_' + str(FL_epoch) + '_GAN_Final_' + modelFileName + '_confusion_matrix')

# report = classification_report(y_test_l, classes, target_names=labelEncoder.classes_, digits=4)
# print(report)
# print('########################################')
# time.sleep(10)

report = pd.read_csv('FL_'+ str(FL_epoch) + "_Report.csv", index_col=0)

#########################################################################
# 2022.03.17修改
print('@@@@@@@@@@@@ Recording Best @@@@@@@@@@@@@@@')
best_record = report.drop(['precision', 'f1-score', 'support'], axis=1)
best_record = pd.DataFrame(best_record).transpose()
best_record = best_record.to_dict()
print('Best Record:')
print(best_record)
print('-----------------------------------------------------------')
print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
######################################################################### 留
final_list = list()
pre_list = list()
now_list = list()
now_list, pre_list, final_list = filter_report(now_list, GAN_epoch, pre_list, final_list, FL_epoch)
# now_list, final_list = filter_Label(pre_list, now_list, final_list) #這段不一定要
print('Now Round: ', GAN_epoch)
print('Need to GAN: ', now_list)
print("Don't need to GAN: ", final_list)
final_temp_data_test = pd.DataFrame()
############################################################
#要把飽和的類別的生成資料跟著一起訓練  搞定
#卡點，可設置說，最多只跑到幾round就必須要停止。
#當都飽和了，最後需要將存下來的生成資料去做總訓練，讓各類別都有最好效能。(可能必須要寫出來)
#可能在過程中，需要生成的各類別，都會去紀錄當前這些類別的recall效能，並在過程中，去比較每次再訓練後的效能，
#若超過就去替換，最後，當訓還跑完後，需要有最後一步驟就是不停的用這些生成資料重複訓練，直到達到紀錄的效能，
while now_list:
    print('Strating using GAN')
    time.sleep(3)
    os.system('python FL_combine_test.py')
    print('@@@@@@@@@@@@@@ Retraining @@@@@@@@@@@@@@@@@@@@@')
    print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')

    print("@@@@@@@@@@@@@@@@@After Append Sample@@@@@@@@@@@@@@@@@@@@@")
    ##### FL_次數_GAN_次數_temp_data_test
    sample = pd.read_csv('FL_'+ str(FL_epoch) + '_GAN_' + str(GAN_epoch) + '_temp_data_test.csv')
    fulltrain_append = fulltrain.append(sample, ignore_index=True)
    print('New Sample Data:')
    print(sample)
    fulltrain_append = fulltrain_append.append(final_temp_data_test, ignore_index=True)
    print('With Old Sample Data:')
    print(final_temp_data_test)
    
    print("@@@@@@@@@@@@@@@@@After Sorting@@@@@@@@@@@@@@@@@@@@@")
    fulltrain_append = fulltrain_append.sort_values(by='type')
    print(fulltrain_append)
    time.sleep(3)

    print("")

    train_X_train = pd.DataFrame()
    train_X_val = pd.DataFrame()

    for label in fulltrain.type.unique():
        train_train, train_val = splitData(fulltrain_append, 0.2, label)
        print('Label: ', label)
        print('Train: ', train_train.shape)
        print('Vali: ', train_val.shape)
        train_X_train = train_X_train.append(train_train, ignore_index=True)
        train_X_val = train_X_val.append(train_val, ignore_index=True)

    # del fulltrain
    # gc.collect()

    print(train_X_train)
    print(train_X_val)

    print('-----------X_train----------------')
    oneHotTrain = oneHotLabel(train_X_train)
    # labelEncoder = preprocessing.LabelEncoder()
    # labelEncoder.fit(fulltrain.type.unique())
    X_train, Y_train = buildTrain(oneHotTrain, labelEncoder)
    print('-----------ok---------------------')
    print('')
    print('-----------X_val------------------------')
    oneHotTrain = oneHotLabel(train_X_val)
    X_val, Y_val = buildTrain(oneHotTrain, labelEncoder)
    print('------------ok------------------------')

    X_train, Y_train = shuffle(X_train, Y_train)
    X_val, Y_val = shuffle(X_val, Y_val)

    del train_X_train
    # gc.collect()
    del train_X_val
    # gc.collect()

    logging.info((X_train.shape, Y_train.shape, X_val.shape, Y_val.shape))
    
    ################################################################
    ##### 0: FL_次數_modelFileName
    ##### 1+:FL_次數_GAN_次數_modelFileName
    
    if GAN_epoch == 0:
        model = load_model('FL_' + str(FL_epoch) + '_' + modelFileName + '.h5')
        adam = optimizers.Adam(lr=lr)
        model.compile(loss='categorical_crossentropy', 
                optimizer=adam, metrics=['categorical_accuracy'])
    else:
        model = load_model('FL_' + str(FL_epoch) + '_GAN_' + str(GAN_epoch) + '_' + modelFileName + '.h5')
        adam = optimizers.Adam(lr=lr)
        model.compile(loss='categorical_crossentropy', 
                optimizer=adam, metrics=['categorical_accuracy'])

    GAN_epoch = GAN_epoch + 1 #要訓練下一個round的模型，因此是+1

    del final_temp_data_test
    final_temp_data_test = pd.DataFrame()
    path = 'GAN_count.txt'
    f = open(path, 'w')
    f.write(str(GAN_epoch))
    f.close()
    #################################################################

    earlyStopping = EarlyStopping(monitor="loss", patience=10, verbose=2, mode="auto")
    ##### 1+:FL_次數_GAN_次數_modelFileName
    modelCheckPoint = ModelCheckpoint(filepath='FL_' + str(FL_epoch) + '_GAN_' + str(GAN_epoch) + '_' + modelFileName + '.h5', monitor='val_loss', mode='min', verbose=2, save_best_only=True)
    reduceLROnPlateau = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, verbose=2)
    callbacks=[modelCheckPoint]

    History = model.fit(X_train, Y_train, epochs=GAN_train_epoch,
            batch_size=local_batch_size, validation_data=(X_val, Y_val),
            shuffle=True, verbose=2, callbacks=callbacks)

    loss, accuracy = model.evaluate(X_train, Y_train)

    W, b = model.layers[0].get_weights()

    val_loss, val_accuracy = model.evaluate(X_val, Y_val)
    print("\nTraining Loss: %f, Accuracy: %f" % (loss, accuracy))
    #### FL_次數_GAN_次數_....
    historyDisplay(History, accuracy, val_accuracy, 'categorical_accuracy', 'val_categorical_accuracy', 'FL_' + str(FL_epoch) + '_GAN_' + str(GAN_epoch) + '_' + modelFileName + 'ModelAcc', 'accuracy', 'epoch', 'FL_' + str(FL_epoch) + '_GAN_' + str(GAN_epoch) + '_' + modelFileName + '_acc_history')
    historyDisplay(History, loss, val_loss, 'loss', 'val_loss', 'FL_' + str(FL_epoch) + '_GAN_' + str(GAN_epoch) + '_' + modelFileName + 'ModelLoss', 'loss', 'epoch', 'FL_' + str(FL_epoch) + '_GAN_' + str(GAN_epoch) + '_' + modelFileName + '_loss_history')

    hist_df = pd.DataFrame(History.history) 
    hist_df.to_csv('FL_' + str(FL_epoch) + '_GAN_' + str(GAN_epoch) + '_' + modelFileName + '_Training_Validation_Loss_and_Accuracy.csv')

    logging.info('Testing Start')
    start = datetime.datetime.now()
    ##### 1+:FL_次數_GAN_次數_modelFileName
    model_load = load_model('FL_' + str(FL_epoch) + '_GAN_' + str(GAN_epoch) + '_' + modelFileName + '.h5')
    adam = optimizers.Adam(lr=lr)
    model_load.compile(loss='categorical_crossentropy', 
            optimizer=adam, metrics=['categorical_accuracy'])
    
    loss, accuracy = model_load.evaluate(X_test1, Y_test1)
    end = datetime.datetime.now()
    print(" Testing time: ",(end-start).total_seconds())
    print("\nTesting Loss: %f, Accuracy: %f" % (loss, accuracy))

    predict = model_load.predict(X_test1)

    classes = predict.argmax(axis=-1)
    y_test_l = Y_test1.argmax(axis=-1)

    ##### FL_次數_GAN_次數_....
    plot_confusion_matrix(y_test_l, classes, classes=labelEncoder.classes_, title='MLP Confusion matrix', fileName='FL_' + str(FL_epoch) + '_GAN_' + str(GAN_epoch) + '_' + modelFileName + '_confusion_matrix')

    report = classification_report(y_test_l, classes, target_names=labelEncoder.classes_, digits=4, output_dict=True)
    print(report)
    
    

    del report['accuracy'], report['macro avg'], report['weighted avg'] #把最後的三個不重要的index丟掉
    del model
    del model_load
    del fulltrain_append
    tf.keras.backend.clear_session()
    report = pd.DataFrame(report).transpose()
    # report = pd.DataFrame(report)
    print(report.columns)
    
    ##### 1+:FL_次數_GAN_次數_Report

    report.to_csv('FL_' + str(FL_epoch) + '_GAN_' + str(GAN_epoch) + "_Report.csv") ###filter_Report_名稱須改GAN_次數_Report
    
    ######## 2022/04/01 修改 #####
    evaliation_self(y_test_l, classes, labelEncoder, FL_epoch=FL_epoch, GAN_epoch=GAN_epoch, loss=loss, last_training=1)
    
    print('@@@@@@@@@@@@@ Recording Best @@@@@@@@@@@@@@@@')
    best_record = record_best(best_record, GAN_epoch)
    print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')

    # pre_list = now_list #之前的
    # now_list = list()
    now_list, pre_list, final_list = filter_report(now_list, GAN_epoch, pre_list, final_list, FL_epoch)

    final_list, now_list = check_Train_time(now_list, final_list, GAN_epoch, stop_point)

    print('Now Round: ', GAN_epoch)
    print('Need to GAN: ', now_list)
    print("Don't need to GAN: ", final_list)
    
    print('@@@@@@@@@@@@@@@@ Gather effective Data @@@@@@@@@@@@@@@@@@@@@@@')

    ##### FL_次數_GAN_次數_temp_data_test
    
    if final_list: #判斷有沒有類別飽和
        pre_temp_data = pd.read_csv('FL_' + str(FL_epoch) + '_GAN_' + str(GAN_epoch-1) + '_temp_data_test.csv') #將前一round的類別讀進來
        for i in final_list: #根據飽和的類別，進去前一round裡面抓資料
            filter = (pre_temp_data['type'] == i) #從前round的生成資料中篩選該類別的收籌資料
            if os.path.exists('FL_' + str(FL_epoch) + '_final_temp_data_test.csv'): #判斷有無final_temp的csv檔

                #### FL_次數_final_temp_data_test
                final_temp_data_test = pd.read_csv('FL_' + str(FL_epoch) + '_final_temp_data_test.csv') #有的話，將該檔案讀進來

                try: #因為有可能該類別中視前幾輪的，因此前一論可能抓不到該類別的資料
                    final_temp_data_test = final_temp_data_test.append(pre_temp_data[filter], ignore_index=True)

                    #### FL_次數_final_temp_data_test
                    final_temp_data_test.to_csv('FL_' + str(FL_epoch) + '_final_temp_data_test.csv', index=False) #將類別append在final_temp的資料後面，並再次存取
                except Exception as e:
                    print('There is no' + str(i) + ' sample.')
                    print(e)
                    time.sleep(3)
            else:
                final_temp_data_test = pre_temp_data[filter] 

                #### FL_次數_final_temp_data_test
                final_temp_data_test.to_csv('FL_' + str(FL_epoch) + '_final_temp_data_test.csv', index=False)
    print('Final_temp_data_test:')
    print(final_temp_data_test)
    print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
    time.sleep(3)

##### 若都已達標，就沒有final_list，也就不用做CTC部分
if final_list:
    print('The CTC-GAN Finished!')

    ##### FL_次數_GAN_次數_Final Report
    print('FL_' + str(FL_epoch) + '_GAN_' + str(GAN_epoch) + ' Final Report:')
    ##### FL_次數_GAN_次數_Report
    report = pd.read_csv('FL_' + str(FL_epoch) + '_GAN_' + str(GAN_epoch) + '_Report.csv', index_col=0)
    report.drop(['precision', 'f1-score', 'support'], axis=1, inplace=True)
    print(report)
    print('---------------------------------------------')
    print('Now Try to train the Best')
    print('The Best Record')
    print(best_record)
    print('---------------------------------------------')

    report = pd.DataFrame(report).transpose()
    report = report.to_dict()
    print(report)

    best_recall = list()

    for total_key, value in best_record.items():
        for key in value:
            print(key + ': ' + str(value[key]))
            best_recall.append(value[key])

    Is_best = Compare(best_recall, report)

    print("@@@@@@@@@@@@@@@@@After Append Sample@@@@@@@@@@@@@@@@@@@@@")

    #### FL_次數_final_temp_data_test
    sample = pd.read_csv('FL_' + str(FL_epoch) + '_final_temp_data_test.csv')
    fulltrain_append = fulltrain.append(sample, ignore_index=True)
    print('New Sample Data:')
    print(sample)

    print("@@@@@@@@@@@@@@@@@After Sorting@@@@@@@@@@@@@@@@@@@@@")
    fulltrain_append = fulltrain_append.sort_values(by='type')
    print(fulltrain_append)
    time.sleep(3)

    print('Is Best? :')
    print(Is_best)

    if Is_best == True:
        model_load = load_model('FL_' + str(FL_epoch) + '_GAN_' + str(GAN_epoch) + '_' + modelFileName + '.h5')
        model_load.save('FL_'+ str(FL_epoch) + '_final_' + modelFileName + '.h5')  
        report = pd.read_csv('FL_' + str(FL_epoch) + '_GAN_' + str(GAN_epoch) + '_Report.csv', index_col=0)
        report.to_csv('FL_' + str(FL_epoch) + "_Final_Report.csv")
        evaliation_self(y_test_l, classes, labelEncoder, FL_epoch=FL_epoch, GAN_epoch=GAN_epoch, loss=loss, last_training=0) 
    else:

        final_GAN_epoch = 1 

        while Is_best == False:

            print('Now Final Train rounds: ', final_GAN_epoch)
            time.sleep(3)

            train_X_train = pd.DataFrame()
            train_X_val = pd.DataFrame()

            for label in fulltrain.type.unique():
                train_train, train_val = splitData(fulltrain_append, 0.2, label)
                print('Label: ', label)
                print('Train: ', train_train.shape)
                print('Vali: ', train_val.shape)
                train_X_train = train_X_train.append(train_train, ignore_index=True)
                train_X_val = train_X_val.append(train_val, ignore_index=True)

            print(train_X_train)
            print(train_X_val)

            print('-----------X_train----------------')
            oneHotTrain = oneHotLabel(train_X_train)
            # labelEncoder = preprocessing.LabelEncoder()
            # labelEncoder.fit(train.type.unique())
            X_train, Y_train = buildTrain(oneHotTrain, labelEncoder)
            print('-----------ok---------------------')
            print('')
            print('-----------X_val------------------------')
            oneHotTrain = oneHotLabel(train_X_val)
            X_val, Y_val = buildTrain(oneHotTrain, labelEncoder)
            print('------------ok------------------------')

            X_train, Y_train = shuffle(X_train, Y_train)
            X_val, Y_val = shuffle(X_val, Y_val)

            del train_X_train
            del train_X_val

            logging.info((X_train.shape, Y_train.shape, X_val.shape, Y_val.shape))

            ################################################################
            ##### FL_次數_GAN_次數_modelFileName
            model = load_model('FL_' + str(FL_epoch) + '_GAN_' + str(GAN_epoch) + '_' + modelFileName + '.h5')
            adam = optimizers.Adam(lr=lr)
            model.compile(loss='categorical_crossentropy', 
                        optimizer=adam, metrics=['categorical_accuracy'])
            #################################################################

            earlyStopping = EarlyStopping(monitor="loss", patience=10, verbose=2, mode="auto")

            #### FL_次數_modelFileName
            modelCheckPoint = ModelCheckpoint(filepath='FL_'+ str(FL_epoch) + '_final_' + str(final_GAN_epoch) + '_' + modelFileName + '.h5', monitor='val_loss', mode='min', verbose=2, save_best_only=True)
            reduceLROnPlateau = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, verbose=2)
            callbacks=[modelCheckPoint]

            History = model.fit(X_train, Y_train, epochs=GAN_train_epoch,
                    batch_size=local_batch_size, validation_data=(X_val, Y_val),
                    shuffle=True, verbose=2, callbacks=callbacks)

            loss, accuracy = model.evaluate(X_train, Y_train)

            val_loss, val_accuracy = model.evaluate(X_val, Y_val)
            print("\nTraining Loss: %f, Accuracy: %f" % (loss, accuracy))
            ##### FL_次數_final_....
            historyDisplay(History, accuracy, val_accuracy, 'categorical_accuracy', 'val_categorical_accuracy', 'FL_' + str(FL_epoch) + '_GAN_Final_' + str(final_GAN_epoch) + '_' + modelFileName + 'ModelAcc', 'accuracy', 'epoch', 'FL_' + str(FL_epoch) + '_GAN_Final_' + str(final_GAN_epoch) + '_' +  modelFileName + '_acc_history')
            historyDisplay(History, loss, val_loss, 'loss', 'val_loss', 'FL_' + str(FL_epoch) + '_GAN_Final_' + str(final_GAN_epoch) + '_' + modelFileName + 'ModelLoss', 'loss', 'epoch', 'FL_' + str(FL_epoch) + '_GAN_Final_' + str(final_GAN_epoch) + '_' +  modelFileName + '_loss_history')

            hist_df = pd.DataFrame(History.history) 
            hist_df.to_csv('FL_'+ str(FL_epoch) + '_final_' + str(final_GAN_epoch) + '_' + modelFileName + '_Training_Validation_Loss_and_Accuracy.csv')

            logging.info('Testing Start')
            start = datetime.datetime.now()

            #### FL_次數_Final_modelFileName
            model_load = load_model('FL_'+ str(FL_epoch) + '_final_' + str(final_GAN_epoch) + '_' + modelFileName + '.h5')
            adam = optimizers.Adam(lr=lr)
            model_load.compile(loss='categorical_crossentropy', 
                    optimizer=adam, metrics=['categorical_accuracy'])

            
            
            loss, accuracy = model_load.evaluate(X_test1, Y_test1)
            end = datetime.datetime.now()
            print(" Testing time: ",(end-start).total_seconds())
            print("\nTesting Loss: %f, Accuracy: %f" % (loss, accuracy))

            predict = model_load.predict(X_test1)

            classes = predict.argmax(axis=-1)
            y_test_l = Y_test1.argmax(axis=-1)

            #### FL_次數_Final_...
            plot_confusion_matrix(y_test_l, classes, classes=labelEncoder.classes_, title='MLP Confusion matrix', fileName='FL_' + str(FL_epoch) + '_GAN_Final_' + str(final_GAN_epoch) + '_' + modelFileName + '_confusion_matrix')

            report = classification_report(y_test_l, classes, target_names=labelEncoder.classes_, digits=4, output_dict=True)
            print(report)

            ##### 2022/04/01 修改 #####
            evaliation_self(y_test_l, classes, labelEncoder, FL_epoch=FL_epoch, GAN_epoch=GAN_epoch, loss=loss, last_training=2, final_GAN_epoch=final_GAN_epoch)

            del report['accuracy'], report['macro avg'], report['weighted avg'] #把最後的三個不重要的index丟掉
            del model
            del model_load
            # del fulltrain_append
            tf.keras.backend.clear_session()
            report = pd.DataFrame(report).transpose()
            # report = pd.DataFrame(report)
            print(report.columns)

            #### FL_次數_Final_Report
            report.to_csv('FL_' + str(FL_epoch) + "_Final_"  + str(final_GAN_epoch) + "_Report.csv")

            ##### Check Is Best ####
            #### FL_次數_Final_Report
            report = pd.read_csv('FL_' + str(FL_epoch) + "_Final_"  + str(final_GAN_epoch) + "_Report.csv", index_col=0)
            report.drop(['precision', 'f1-score', 'support'], axis=1, inplace=True)
            print(report)
            print('---------------------------------------------')
            
            if final_GAN_epoch == 1:
                Ftp_data = report['recall'][3]
                Snapchat = report['recall'][14]
                SoundCloud = report['recall'][15]
                eBay = report['recall'][24]
                during_best = 1
            
            during_best, Ftp_data, Snapchat, SoundCloud, eBay = during_filter_report(final_GAN_epoch, Ftp_data, Snapchat, SoundCloud, eBay, report, during_best)

            report = pd.DataFrame(report).transpose()
            report = report.to_dict()
            print(report)

            Is_best = Compare(best_recall, report)
            print('Is Best? :')
            print(Is_best)
            
            
            if Is_best == False:
                if final_GAN_epoch == stop_point:
                    print('Final Training Rounds: ', final_GAN_epoch)
                    print('During Best:', during_best)
                    time.sleep(3)
                    start = datetime.datetime.now()
                    model_load = load_model('FL_'+ str(FL_epoch) + '_final_' + str(during_best) + '_' + modelFileName + '.h5')
                    model_load.save('FL_'+ str(FL_epoch) + '_final_' + modelFileName + '.h5')  
                    adam = optimizers.Adam(lr=lr)
                    model_load.compile(loss='categorical_crossentropy', 
                            optimizer=adam, metrics=['categorical_accuracy'])

                    
                    loss, accuracy = model_load.evaluate(X_test1, Y_test1)
                    end = datetime.datetime.now()
                    print(" Testing time: ",(end-start).total_seconds())
                    print("\nTesting Loss: %f, Accuracy: %f" % (loss, accuracy))

                    predict = model_load.predict(X_test1)

                    classes = predict.argmax(axis=-1)
                    y_test_l = Y_test1.argmax(axis=-1)

                    #### FL_次數_Final_...
                    ##改
                    plot_confusion_matrix(y_test_l, classes, classes=labelEncoder.classes_, title='MLP Confusion matrix', fileName='FL_' + str(FL_epoch) + '_GAN_Final_' + modelFileName + '_confusion_matrix')

                    report = classification_report(y_test_l, classes, target_names=labelEncoder.classes_, digits=4, output_dict=True)
                    print(report)

                    ##### 2022/04/01 修改 #####
                    ##改
                    evaliation_self(y_test_l, classes, labelEncoder, FL_epoch=FL_epoch, GAN_epoch=GAN_epoch, loss=loss, last_training=3, final_GAN_epoch=final_GAN_epoch, during_best=during_best)

                    del report['accuracy'], report['macro avg'], report['weighted avg'] #把最後的三個不重要的index丟掉
                    
                    del model_load
                    # del fulltrain_append
                    tf.keras.backend.clear_session()
                    report = pd.DataFrame(report).transpose()
                    # report = pd.DataFrame(report)
                    print(report.columns)

                    #### FL_次數_Final_Report
                    ##改
                    report.to_csv('FL_' + str(FL_epoch) + "_Final_Report.csv")
                    time.sleep(3)
                    break
                final_GAN_epoch = final_GAN_epoch + 1
            elif Is_best == True:
                print('Final Training Rounds: ', final_GAN_epoch)
                #print('During Best:', during_best)
                time.sleep(3)
                start = datetime.datetime.now()
                model_load = load_model('FL_'+ str(FL_epoch) + '_final_' + str(final_GAN_epoch) + '_' + modelFileName + '.h5')
                model_load.save('FL_'+ str(FL_epoch) + '_final_' + modelFileName + '.h5')  
                adam = optimizers.Adam(lr=lr)
                model_load.compile(loss='categorical_crossentropy', 
                        optimizer=adam, metrics=['categorical_accuracy'])

                
                loss, accuracy = model_load.evaluate(X_test1, Y_test1)
                end = datetime.datetime.now()
                print(" Testing time: ",(end-start).total_seconds())
                print("\nTesting Loss: %f, Accuracy: %f" % (loss, accuracy))

                predict = model_load.predict(X_test1)

                classes = predict.argmax(axis=-1)
                y_test_l = Y_test1.argmax(axis=-1)

                #### FL_次數_Final_...
                ##改
                plot_confusion_matrix(y_test_l, classes, classes=labelEncoder.classes_, title='MLP Confusion matrix', fileName='FL_' + str(FL_epoch) + '_GAN_Final_' + modelFileName + '_confusion_matrix')

                report = classification_report(y_test_l, classes, target_names=labelEncoder.classes_, digits=4, output_dict=True)
                print(report)

                ##### 2022/04/01 修改 #####
                ##改
                evaliation_self(y_test_l, classes, labelEncoder, FL_epoch=FL_epoch, GAN_epoch=GAN_epoch, loss=loss, last_training=0, final_GAN_epoch=final_GAN_epoch, during_best=during_best)

                del report['accuracy'], report['macro avg'], report['weighted avg'] #把最後的三個不重要的index丟掉
                
                del model_load
                # del fulltrain_append
                tf.keras.backend.clear_session()
                report = pd.DataFrame(report).transpose()
                # report = pd.DataFrame(report)
                print(report.columns)

                #### FL_次數_Final_Report
                ##改
                report.to_csv('FL_' + str(FL_epoch) + "_Final_Report.csv")
        

    print(pd.DataFrame(best_record).transpose())
    print('-----------------------------')
    print(pd.DataFrame(report))
else:
    print('All labels achieved!')
    start = datetime.datetime.now()
    model_load = load_model('FL_'+ str(FL_epoch) + '_' + modelFileName + '.h5')
    model_load.save('FL_'+ str(FL_epoch) + '_final_' + modelFileName + '.h5')
    adam = optimizers.Adam(lr=lr)
    model_load.compile(loss='categorical_crossentropy', 
            optimizer=adam, metrics=['categorical_accuracy'])

    
    loss, accuracy = model_load.evaluate(X_test1, Y_test1)
    end = datetime.datetime.now()
    print(" Testing time: ",(end-start).total_seconds())
    print("\nTesting Loss: %f, Accuracy: %f" % (loss, accuracy))

    predict = model_load.predict(X_test1)

    classes = predict.argmax(axis=-1)
    y_test_l = Y_test1.argmax(axis=-1)

    #### FL_次數_Final_...
    ##改
    plot_confusion_matrix(y_test_l, classes, classes=labelEncoder.classes_, title='MLP Confusion matrix', fileName='FL_' + str(FL_epoch) + '_GAN_Final_' + modelFileName + '_confusion_matrix')

    report = classification_report(y_test_l, classes, target_names=labelEncoder.classes_, digits=4, output_dict=True)
    print(report)

    ##### 2022/04/01 修改 #####
    ##改
    evaliation_self(y_test_l, classes, labelEncoder, FL_epoch=FL_epoch, GAN_epoch=GAN_epoch, loss=loss, last_training=0)

    del report['accuracy'], report['macro avg'], report['weighted avg'] #把最後的三個不重要的index丟掉
    
    del model_load
    # del fulltrain_append
    tf.keras.backend.clear_session()
    report = pd.DataFrame(report).transpose()
    # report = pd.DataFrame(report)
    print(report.columns)

    #### FL_次數_Final_Report
    ##改
    report.to_csv('FL_' + str(FL_epoch) + "_Final_Report.csv")

    ##由於類別都飽和不再做GAN，而回去client程式後，需讀取當前生成資料，因此，須將前輪的生成資料讀進並存成當前輪數的生成資料。
    print("Load pre round's final temp data")
    pre_round_final_temp_data = pd.read_csv('FL_' + str(FL_epoch-1) + '_final_temp_data_test.csv')
    print('--------------------------')
    print("Save pre round's final temp data to be now final temp data")
    pre_round_final_temp_data.to_csv('FL_' + str(FL_epoch) + '_final_temp_data_test.csv', index=False)
# print('############ Check Time ##############')
# model_load = load_model('FL_'+ str(FL_epoch) + '_final_' + modelFileName + '.h5')
# adam = optimizers.Adam(lr=lr)
# model_load.compile(loss='categorical_crossentropy', 
#         optimizer=adam, metrics=['categorical_accuracy'])

# loss, accuracy = model_load.evaluate(X_test1, Y_test1)
# print("\nTesting Loss: %f, Accuracy: %f" % (loss, accuracy))

# predict = model_load.predict(X_test1)

# classes = predict.argmax(axis=-1)
# y_test_l = Y_test1.argmax(axis=-1)

# #### FL_次數_Final_...
# plot_confusion_matrix(y_test_l, classes, classes=labelEncoder.classes_, title='MLP Confusion matrix', fileName='FL_' + str(FL_epoch) + '_GAN_Final_' + modelFileName + '_confusion_matrix')

# report = classification_report(y_test_l, classes, target_names=labelEncoder.classes_, digits=4)
# print(report)
# print('########################################')

