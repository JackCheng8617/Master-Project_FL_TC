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
    #trainFilePath = 'Datasets/141_7_train_withOtherFeature_V2(seperate)_withoutHeader.csv'
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

def readTesting():

    trainFilePath = 'Datasets/Cut_141_7_test_withOtherFeature_V2(seperate)_withoutHeader.csv'

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

def historyDisplay(trainHistory, eval_train, eval_val, trainMetrics, validationMetrics, title, yLabelName, xLabelName, fileName, round):
    
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
    plt.savefig(str(round+1) + '_' + fileName)
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
    fig.savefig(str(round+1) +'_' +fileName)
    return ax

def buildMLPModel():
    # model = Sequential()
    # model.add(Dense(256, input_shape=(int(feature_size),), name='Input'))
    # model.add(Dense(256, name='Hiden_1'))
    # model.add(LeakyReLU(alpha=0.2))
    # model.add(Dropout(0.2, name='Dropout_1'))
    # model.add(Dense(256, name='Hiden_2'))
    # model.add(LeakyReLU(alpha=0.2))
    # model.add(Dropout(0.2, name='Dropout_2'))
    # model.add(Dense(256,  name='Hiden_3'))
    # model.add(LeakyReLU(alpha=0.2))
    # model.add(Dropout(0.2, name='Dropout_3'))
    # model.add(Dense(25, activation='softmax', name='Softmax_classifier'))
    # adam = optimizers.Adam(lr=0.01)
    # model.compile(loss='categorical_crossentropy', 
    #             optimizer=adam, metrics=['categorical_accuracy'])
    # model.summary()
    
    # return model
    model = Sequential()
    model.add(Dense(256, input_shape=(int(feature_size),), name='Input'))
    model.add(Dense(256, activation='relu', name='Hiden_1'))
    model.add(Dropout(0.2, name='Dropout_1'))
    model.add(Dense(256, activation='relu', name='Hiden_2'))
    model.add(Dropout(0.2, name='Dropout_2'))
    model.add(Dense(256, activation='relu', name='Hiden_3'))
    model.add(Dropout(0.2, name='Dropout_3'))
    model.add(Dense(labels, activation='softmax', name='Softmax_classifier'))
    adam = optimizers.Adam(lr=lr)
    model.compile(loss='categorical_crossentropy', 
                optimizer=adam, metrics=['categorical_accuracy'])
    model.summary()
    return model

def buildCNNModel():
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=2, padding='Same', 
                    activation='relu', input_shape=(int(feature_size), 1),
                    name='Conv_1'))
    model.add(MaxPooling1D(2, name='Maxpool_1'))
    model.add(Conv1D(filters=64, kernel_size=2, padding='Same',
                    activation='relu', name='Conv_2'))
    model.add(MaxPooling1D(2, name='Maxpool_2'))

    model.add(Flatten(name='Full_connect'))
    model.add(Dense(64, activation='relu', name='relu_FC'))
    model.add(Dense(labels, activation='softmax', name='Softmax_classifier'))
    
    adam = optimizers.Adam(lr=lr)#lr=0.01
    model.compile(loss='categorical_crossentropy', 
                optimizer=adam, metrics=['categorical_accuracy'])
    model.summary()
    
    return model

def buildLSTMModel():
    model = Sequential()
    model.add(LSTM(64, dropout=0.2, recurrent_dropout=0.2, input_shape=(int(feature_size), 1)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.25, name='Dropout_1'))
    model.add(Dense(labels, activation='softmax', name='Softmax_classifier'))
    rmSprop = optimizers.RMSprop(lr=lr)
    model.compile(loss='categorical_crossentropy', 
                optimizer=rmSprop, metrics=['categorical_accuracy'])
    model.summary()
    return model

def buildSAEModel():
    model = Sequential()
    model.add(Dense(128, activation='relu', input_shape=(int(feature_size),), name='AE1_input'))
    model.add(Dense(64, activation='relu', name='AE1_hiden'))
    model.add(Dropout(0.2, name='Dropout_1'))
    model.add(Dense(64, activation='relu', name='AE2_input'))
    model.add(Dense(32, activation='relu', name='AE2_hiden'))
    model.add(Dense(64, activation='relu', name='AE2_output'))
    model.add(Dropout(0.2, name='Dropout_2'))
    
    model.add(Dense(labels, activation='softmax', name='Softmax_classifier'))
    
    adam = optimizers.Adam(lr=lr)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['categorical_accuracy'])
    
    model.summary()

    return model

def check_exists():
    last = 0
    fileExist = False
    if os.path.exists('global_weights.h5'):
        fileExist = True
        last = 0
    elif os.path.exists('last_global_weights.h5'):
        fileExist = True
        last = 1
    return fileExist, last

def check_parameters():
    fileExist = False
    if os.path.exists('parameters_Frame.csv'):
        fileExist = True
    return fileExist

def evaliation_self(y_test_l, classes, labelEncoder, rounds, loss):
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
    confusion_metrics.to_csv(str(rounds+1)+'_caculate_metrics.csv', index=False)
    ###########################################

####2022/04/18 change####
def filter_report(rounds, CTC_GAN, do_GAN_first, do_GAN_second, remainder_num):
    saturation = False
    # if (CTC_GAN and ((round == do_GAN_first) or (round == do_GAN_second))):
    saturation = False
    global saturation_count
    # if (CTC_GAN and ((round == do_GAN_first) or (round == do_GAN_second))):
    if (CTC_GAN and (((rounds%do_GAN_first)==remainder_num) or ((rounds%do_GAN_second)==remainder_num))):
        report = pd.read_csv('FL_'+ str(rounds) + "_Final_Report.csv", index_col=0)
    else:
        report = pd.read_csv('FL_'+ str(rounds) + "_Report.csv", index_col=0)

    report = report.drop(['precision', 'f1-score', 'support'], axis=1)
    
    if CTC_GAN:
        print('With CTC-GAN')

        if((report['recall'][3] >= 0.8000) and (report['recall'][14] >= 0.8000) and (report['recall'][15] >= 0.8000) and (report['recall'][24] >= 0.8000)):
            saturation_count = saturation_count + 1
            print('飽和次數: ', saturation_count)

            if saturation_count == 3:
                
                saturation = True
                print('飽和次數已達標，將回傳 saturation 為: ', saturation)
            time.sleep(5)

        else:
            print('飽和中斷')
            saturation_count = 0
            saturation = False
    
    else:
        print('Without CTC-GAN')

        if((report['recall'][3] >= 0.8000) and (report['recall'][14] >= 0.8000) and (report['recall'][15] >= 0.8000) and (report['recall'][24] >= 0.8000)):
            saturation_count = saturation_count + 1
            print('飽和次數: ', saturation_count)

            if saturation_count == 1:
                
                saturation = True
                print('飽和次數已達標，將回傳 saturation 為: ', saturation)
            time.sleep(5)

        else:
            print('飽和中斷')
            saturation_count = 0
            saturation = False
            
    return saturation

# seed_value= 42
# import numpy as np
# np.random.seed(seed_value)
# import random
# random.seed(seed_value)


### Control GPU used memory
gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

train = readTrain()
print('------------Local Data----------------')
logging.info((
            'src_ipUnique ', len(train['0'].unique()),
            'dst_ipUnique ', len(train['1'].unique()),
            'server_portUnique ', len(train['2'].unique()),
            'proto_Unique ', len(train['3'].unique()),
            'total ', len(train['0'].unique())+
                    len(train['1'].unique())+
                    len(train['2'].unique())+
                    len(train['3'].unique())))
print('-------------------------------------')

parameters_File = False

while parameters_File == False:
    parameters_File = check_parameters() #判斷資料是否存在以及，如果last=1是否是最後一次的global round
    print(parameters_File)
    if parameters_File == False:
        print('Wating parameters file!')
        time.sleep(3)
    elif parameters_File == True:
        parameters_Frame = pd.read_csv('parameters_Frame.csv')
        print(parameters_Frame)
        hash_vector_size1 = parameters_Frame['src_ipUnique'][0]
        hash_vector_size2 = parameters_Frame['dst_ipUnique'][0]
        hash_vector_size3 = parameters_Frame['server_portUnique'][0]
        hash_vector_size4 = parameters_Frame['proto_Unique'][0]
        modelFileName = parameters_Frame['ModelFileName'][0]
        feature_size = parameters_Frame['feature_size'][0]
        local_epoch = parameters_Frame['local_epoch'][0]
        local_batch_size = parameters_Frame['local_batch_size'][0]
        lr = parameters_Frame['lr'][0]
        labels = parameters_Frame['labels'][0]
        CTC_GAN = parameters_Frame['CTC-GAN_client_1'][0]
        do_GAN_first = parameters_Frame['do_GAN_first'][0]
        do_GAN_second = parameters_Frame['do_GAN_second'][0]
        remainder_num = parameters_Frame['remainder_num'][0]



        

print('------------Global Data----------------')
logging.info(('src_ipUnique ', hash_vector_size1,
                    'dst_ipUnique ', hash_vector_size2,
                    'server_portUnique ', hash_vector_size3,
                    'proto_Unique ', hash_vector_size4,
                    'total ', hash_vector_size1+
                            hash_vector_size2+
                            hash_vector_size3+
                            hash_vector_size4
                            ))
print('-------------------------------------')       
    
ct = ColumnTransformer([('src_ip', feature_extraction.FeatureHasher(n_features=hash_vector_size1,
                            input_type='string'), '0'),
                        ('dst_ip', feature_extraction.FeatureHasher(n_features=hash_vector_size2,
                            input_type='string'), '1'),
                        ('server_port', feature_extraction.FeatureHasher(n_features=hash_vector_size3,
                            input_type='string'), '2'),
                        ('proto', feature_extraction.FeatureHasher(n_features=hash_vector_size4,
                            input_type='string'), '3')])

ct.fit(train[['0', '1', '2', '3']].astype('str'))

hashTrain = ct.transform(train[['0', '1', '2', '3']].astype('str')).toarray()
Input_name = [str(i) for i in range(hashTrain.shape[1])]
allTrain = pd.DataFrame(hashTrain, columns = Input_name)
print(allTrain)
fulltrain = pd.concat([allTrain, train[['4','5','6','7','8','9']]], axis=1, ignore_index=True)
print(0)
print(fulltrain)
print(type(fulltrain))

print('@@@@@@@@@@ Do Pre-Processing on Scaling of Test @@@@@@@@@@@@@@@@@@@@@@')
test = readTesting()

hashTest = ct.transform(test[['0', '1', '2', '3']].astype('str')).toarray()
Input_name = [str(i) for i in range(hashTest.shape[1])]
allTest = pd.DataFrame(hashTest, columns = Input_name)
fulltest = pd.concat([allTest, test[['4', '5', '6', '7', '8', '9']]], axis=1, ignore_index=True)

X_min_max_scaler = preprocessing.MinMaxScaler()
scaler = X_min_max_scaler.fit(fulltest)
fulltest = scaler.transform(fulltest)
fulltest = pd.DataFrame(fulltest)

fulltest = pd.concat([fulltest, test[['type']]], axis=1, ignore_index=True)
fulltest.columns = [str(i) for i in range(hashTest.shape[1]+6)] + ['type']

print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')

fulltrain = scaler.transform(fulltrain)
fulltrain = pd.DataFrame(fulltrain)

fulltrain = pd.concat([fulltrain, train[['type']]], axis=1, ignore_index=True)
fulltrain.columns = [str(i) for i in range(hashTrain.shape[1]+6)] + ['type']

print("@@@@@@@@@@@@@@@@@After Sorting@@@@@@@@@@@@@@@@@@@@@")
fulltrain = fulltrain.sort_values(by='type')
print(fulltrain)


print('@@@@@@ 預處裡檔案 @@@@@@@@@@@@')
if not os.path.exists('fulltrain.csv'):
    fulltrain.to_csv('fulltrain.csv', index=False)
if not os.path.exists('fulltest.csv'):
    fulltest.to_csv('fulltest.csv', index=False)
print('@@@@@@@@@@@@@@@@@@@@')

print("")

train_X_train = pd.DataFrame()
train_X_val = pd.DataFrame()

for label in train.type.unique():
    train_train, train_val = splitData(fulltrain, 0.2, label)
    print('Label: ', label)
    print('Train: ', train_train.shape)
    print('Vali: ', train_val.shape)
    train_X_train = train_X_train.append(train_train, ignore_index=True)
    train_X_val = train_X_val.append(train_val, ignore_index=True)

print(train_X_train)
print(train_X_val)

print('-----------X_train----------------')
oneHotTrain = oneHotLabel(train_X_train)
labelEncoder = preprocessing.LabelEncoder()
labelEncoder.fit(train.type.unique())
X_train, Y_train = buildTrain(oneHotTrain, labelEncoder)
print('-----------ok---------------------')
print('')
print('-----------X_val------------------------')
oneHotTrain = oneHotLabel(train_X_val)
X_val, Y_val = buildTrain(oneHotTrain, labelEncoder)
print('------------ok------------------------')

X_train, Y_train = shuffle(X_train, Y_train)
X_val, Y_val = shuffle(X_val, Y_val)

logging.info((X_train.shape, Y_train.shape, X_val.shape, Y_val.shape))

print('@@@@@@@@@@@@@@@@ Do Pro-Processing on Onehot @@@@@@@@@@@@@@@@@@@@@@@@@')
oneHotTest = oneHotLabel(fulltest)
X_test1, Y_test1 = buildTest(oneHotTest, labelEncoder)

if modelFileName == 'CNNModel':
    X_test1 = X_test1.reshape(X_test1.shape[0], X_test1.shape[1], 1)

logging.info((X_test1.shape, Y_test1.shape)) #如果CNN就要另外加一段
print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')


local_file = False
####2022/04/18 change####
staturaion = False

round = 0

saturation_count = 0
# CTC_GAN = False ### For testing one client(clinet 2) has CTC-GAN

while local_file != True:
    local_file, last = check_exists() #判斷資料是否存在以及，如果last=1是否是最後一次的global round
    print(local_file)
    print(last)
    if local_file == False:
        print('Clietn 1 Wait!') #不同client須改
        print(round+1)
        time.sleep(3)
    else:
        if last == 0:
             #choose model
            if modelFileName == 'MLPModel':
                X_train = X_train
                Y_train = Y_train
                X_val = X_val
                Y_val = Y_val
                model = buildMLPModel()
                logging.info(modelFileName)
                

            elif modelFileName == 'SAEModel':
                X_train = X_train
                Y_train = Y_train
                X_val = X_val
                Y_val = Y_val
                model = buildSAEModel()
                logging.info(modelFileName)
                

            elif modelFileName == 'CNNModel':
                X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
                Y_train = Y_train
                X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
                Y_val = Y_val
                model = buildCNNModel()
                logging.info(modelFileName)
                

            elif modelFileName == 'LSTMModel':
                X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
                Y_train = Y_train
                X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
                Y_val = Y_val
                model = buildLSTMModel()
                logging.info(modelFileName)

            open_file = False
            while open_file == False:
                try:
                    model.load_weights('global_weights.h5')
                    open_file = True
                except Exception as e:
                    open_file = False
                    print(e)
                    time.sleep(3)
                    print('------------')

            ####2022/04/18 change####
            if os.path.exists('last_client_1_weights.h5'):
                '''
                try:
                    os.system('python client_socket_send.py')
                    print('Client Weights Send Successfully!')
                except Exception as e:
                    print(e)
                    time.sleep(3)
                
                model.load_weights('last_client_1_weights.h5')
                adam = optimizers.Adam(lr=lr)
                model.compile(loss='categorical_crossentropy', 
                optimizer=adam, metrics=['categorical_accuracy'])
                loss, accuracy = model.evaluate(X_test1, Y_test1)
                print("\nTesting Loss: %f, Accuracy: %f" %  (loss, accuracy))

                predict = model.predict(X_test1)
                classes = predict.argmax(axis=-1)
                y_test_l = Y_test1.argmax(axis=-1)

                evaliation_self(y_test_l, classes, labelEncoder, rounds=round, loss=loss)

                plot_confusion_matrix(y_test_l, classes, classes=labelEncoder.classes_, title='Confusion matrix', fileName=modelFileName + '_confusion_matrix')
                ####要有output_dict=True才能轉成Dataframe
                report = classification_report(y_test_l, classes, target_names=labelEncoder.classes_, digits=4, output_dict=True)
                print(report)
                print(type(report))

                ####2022/04/18 change####
                del report['accuracy'], report['macro avg'], report['weighted avg'] #把最後的三個不重要的index丟掉
                tf.keras.backend.clear_session()

                report = pd.DataFrame(report).transpose()
                # report = pd.DataFrame(report)
                print(report.columns)
                print(report)
                time.sleep(5)
                ##### client : FL_次數_Report(第一次)
                ##### CTCGAN : FL_次數_GAN_次數_Report

                report.to_csv('FL_'+ str(round+1) + "_Report.csv")
                '''
                print('Waiting for other client saturation!')
                local_file = False
                time.sleep(3)
                try:
                    os.remove('global_weights.h5')
                    print('The pre-global weights file')
                except Exception as e:
                    print(e)
                    time.sleep(3)
                
            ####2022/04/18 change####
            else:
                start = datetime.datetime.now()

                earlyStopping = EarlyStopping(monitor="loss", patience=10, verbose=2, mode="auto")
                modelCheckPoint = ModelCheckpoint(filepath='FL_'+ str(round+1) + '_' + modelFileName + '.h5', monitor='val_loss', mode='min', verbose=2, save_best_only=True)
                reduceLROnPlateau = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, verbose=2)
                callbacks=[modelCheckPoint]

                History = model.fit(X_train, Y_train, epochs=local_epoch,
                        batch_size=local_batch_size, validation_data=(X_val, Y_val),
                        shuffle=True, verbose=2, callbacks=callbacks)

                loss, accuracy = model.evaluate(X_train, Y_train)
                val_loss, val_accuracy = model.evaluate(X_val, Y_val)
                print("\nTraining Loss: %f, Accuracy: %f" %  (loss, accuracy))
                historyDisplay(History, accuracy, val_accuracy, 'categorical_accuracy', 'val_categorical_accuracy', modelFileName + 'ModelAcc', 'accuracy', 'epoch', modelFileName + '_acc_history', round)
                historyDisplay(History, loss, val_loss, 'loss', 'val_loss', modelFileName + 'ModelLoss', 'loss', 'epoch', modelFileName + '_loss_history', round)

                hist_df = pd.DataFrame(History.history) 
                hist_df.to_csv('FL_'+ str(round+1) + '_' + modelFileName + '_Training_Validation_Loss_and_Accuracy.csv')

                local_file = False
                #Test
                # os.system('python client_test_result.py')
                print('@@@@@@@@@@@@@@ Test Start @@@@@@@@@@@@@@@@@@@@@')
                # print(" Testing time: ",(end-start).total_seconds())
                model_load = load_model('FL_'+ str(round+1) + '_' + modelFileName + '.h5')
                adam = optimizers.Adam(lr=lr)
                model_load.compile(loss='categorical_crossentropy', 
                optimizer=adam, metrics=['categorical_accuracy'])
                loss, accuracy = model_load.evaluate(X_test1, Y_test1)
                print("\nTesting Loss: %f, Accuracy: %f" %  (loss, accuracy))

                predict = model_load.predict(X_test1)
                classes = predict.argmax(axis=-1)
                y_test_l = Y_test1.argmax(axis=-1)

                evaliation_self(y_test_l, classes, labelEncoder, rounds=round, loss=loss)

                plot_confusion_matrix(y_test_l, classes, classes=labelEncoder.classes_, title='Confusion matrix', fileName=modelFileName + '_confusion_matrix')
                ####要有output_dict=True才能轉成Dataframe
                report = classification_report(y_test_l, classes, target_names=labelEncoder.classes_, digits=4, output_dict=True)
                print(report)
                print(type(report))

                ####2022/04/18 change####
                del report['accuracy'], report['macro avg'], report['weighted avg'] #把最後的三個不重要的index丟掉
                tf.keras.backend.clear_session()

                report = pd.DataFrame(report).transpose()
                # report = pd.DataFrame(report)
                print(report.columns)
                print(report)
                time.sleep(5)
                ##### client : FL_次數_Report(第一次)
                ##### CTCGAN : FL_次數_GAN_次數_Report

                report.to_csv('FL_'+ str(round+1) + "_Report.csv")

                ############################# CTC-GAN #########################
                ###### 存初始report ############
                # if (CTC_GAN and ((round == do_GAN_first) or (round == do_GAN_second))):
                if (CTC_GAN and ((((round+1)%do_GAN_first)==remainder_num) or (((round+1)%do_GAN_second)==remainder_num))):
                    # if round == 14:

                    path = 'FL_count.txt'
                    f = open(path, 'w')
                    f.write(str(round+1))
                    f.close()

                    os.system('python FL_classify_loop.py')
                    
                    ######################
                    #################################################################
                end = datetime.datetime.now()
                print(" The Running time: ",(end-start).total_seconds())
                train_time = {'Round': [round+1], 'Start_Time': [start], 'End_Time': [end], 'Total_Time': [(end-start).total_seconds()]}
                train_time = pd.DataFrame(data=train_time)
                if not os.path.exists('Running Time.csv'):
                    train_time.to_csv('Running Time.csv', index=False)
                elif os.path.exists('Running Time.csv'):
                    temp = pd.read_csv('Running Time.csv')
                    temp = temp.append(train_time, ignore_index=True)
                    temp.to_csv('Running Time.csv', index=False)

                ### 統一在Classify_loop做測試
                # print('@@@@@@@@@@@@@@ Test Start @@@@@@@@@@@@@@@@@@@@@')
                # if (CTC_GAN and ((round == do_GAN_first) or (round == do_GAN_second))):
                if (CTC_GAN and ((((round+1)%do_GAN_first)==remainder_num) or (((round+1)%do_GAN_second)==remainder_num))):
                    print('Has CTC-GAN!')
                    model_load = load_model('FL_'+ str(round+1) + '_final_' + modelFileName + '.h5')
                    
                else:
                    print('Not Has CTC-GAN!')
                    model_load = load_model('FL_'+ str(round+1) + '_' + modelFileName + '.h5')
                
                ####2022/04/18 change####
                saturation = filter_report(round+1, CTC_GAN, do_GAN_first, do_GAN_second, remainder_num)
                
                if saturation:
                    model_load.save_weights('last_client_1_weights.h5') #不同client須改
                    
                    path = 'Saturation.txt'
                    f = open(path, 'w')
                    f.write(str(round+1))
                    f.close()
                    
                    print('已飽和')
                    time.sleep(3)
                else:
                    model_load.save_weights('client_1_weights.h5') #不同client須改
                    print('未飽和')
                    time.sleep(3)
                ####

                try:
                    os.remove('global_weights.h5')
                    print('The pre-global weights file')
                except Exception as e:
                    print(e)
                    time.sleep(3)

                try:
                    os.system('python client_socket_send.py')
                    print('Client Weights Send Successfully!')
                except Exception as e:
                    print(e)
                    time.sleep(3)

                del model
                del model_load
                tf.keras.backend.clear_session()
                
                print('@@@@@@@@ Shuffle Data @@@@@@@@@@@@@@@@@')
                train_X_train = pd.DataFrame()
                train_X_val = pd.DataFrame()

                fulltrain_append = pd.DataFrame()

                ### 2022/04/15新增
                # if (CTC_GAN and (round >= do_GAN_first)):
                if (CTC_GAN and (((round+1)%do_GAN_first)==remainder_num)):
                    try:
                        sample = pd.read_csv('FL_' + str(round + 1) + '_final_temp_data_test.csv')
                        fulltrain_append = fulltrain.append(sample, ignore_index=True)
                        print('@@@ 加入生成資料 @@@')
                        print(fulltrain)
                        print(sample)
                        print('@@@@@@@@@@@@@@@@@@@')
                    except Exception as e:
                        print(e)
                        time.sleep(3)
                if (CTC_GAN and (((round+1)%do_GAN_second)==remainder_num)):
                    try:
                        sample = pd.read_csv('FL_' + str(round + 1) + '_final_temp_data_test.csv')
                        fulltrain_append = fulltrain.append(sample, ignore_index=True)
                        print('@@@ 加入生成資料 @@@')
                        print(fulltrain)
                        print(sample)
                        print('@@@@@@@@@@@@@@@@@@@')
                    except Exception as e:
                        print(e)
                        time.sleep(3)
                time.sleep(3)
                

                for label in train.type.unique():
                    # if (CTC_GAN and ((round >= do_GAN_first) or (round >= do_GAN_second))):
                    if (CTC_GAN and ((((round+1)%do_GAN_first)==remainder_num) or (((round+1)%do_GAN_second)==remainder_num))):
                        train_train, train_val = splitData(fulltrain_append, 0.2, label)
                    else:
                        train_train, train_val = splitData(fulltrain, 0.2, label)
                    print('Label: ', label)
                    print('Train: ', train_train.shape)
                    print('Vali: ', train_val.shape)
                    train_X_train = train_X_train.append(train_train, ignore_index=True)
                    train_X_val = train_X_val.append(train_val, ignore_index=True)

                print(train_X_train)
                print(train_X_val)

                print('-----------X_train----------------')
                oneHotTrain = oneHotLabel(train_X_train)
                X_train, Y_train = buildTrain(oneHotTrain, labelEncoder)
                print('-----------ok---------------------')
                print('')
                print('-----------X_val------------------------')
                oneHotTrain = oneHotLabel(train_X_val)
                X_val, Y_val = buildTrain(oneHotTrain, labelEncoder)
                print('------------ok------------------------')

                X_train, Y_train = shuffle(X_train, Y_train)
                X_val, Y_val = shuffle(X_val, Y_val)
                print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')

            round = round + 1

        elif last == 1:
             #choose model
            if modelFileName == 'MLPModel':
                X_train = X_train
                Y_train = Y_train
                X_val = X_val
                Y_val = Y_val
                model = buildMLPModel()
                logging.info(modelFileName)
                

            elif modelFileName == 'SAEModel':
                X_train = X_train
                Y_train = Y_train
                X_val = X_val
                Y_val = Y_val
                model = buildSAEModel()
                logging.info(modelFileName)
                

            elif modelFileName == 'CNNModel':
                X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
                Y_train = Y_train
                X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
                Y_val = Y_val
                model = buildCNNModel()
                logging.info(modelFileName)
                

            elif modelFileName == 'LSTMModel':
                X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
                Y_train = Y_train
                X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
                Y_val = Y_val
                model = buildLSTMModel()
                logging.info(modelFileName)

            local_file = True
            open_file = False
            while open_file == False:
                try:
                    model.load_weights('last_global_weights.h5')
                    # m = tf.keras.models.clone_model(model)
                    # m.set_weights(model.get_weights())
                    # adam = optimizers.Adam(lr=0.01)
                    # m.compile(loss='categorical_crossentropy', 
                    #     optimizer=adam, metrics=['categorical_accuracy'])
                    open_file = True
                except Exception as e:
                    open_file = False
                    print(e)
                    time.sleep(3)

            start = datetime.datetime.now()

            earlyStopping = EarlyStopping(monitor="loss", patience=10, verbose=2, mode="auto")
            modelCheckPoint = ModelCheckpoint(filepath='FL_'+ str(round+1) + '_' + modelFileName + '.h5', monitor='val_loss', mode='min', verbose=2, save_best_only=True)
            reduceLROnPlateau = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, verbose=2)
            callbacks=[modelCheckPoint]

            History = model.fit(X_train, Y_train, epochs=local_epoch,
                    batch_size=local_batch_size, validation_data=(X_val, Y_val),
                    shuffle=True, verbose=2, callbacks=callbacks)

            loss, accuracy = model.evaluate(X_train, Y_train)
            val_loss, val_accuracy = model.evaluate(X_val, Y_val)
            print("\nTraining Loss: %f, Accuracy: %f" %  (loss, accuracy))
            historyDisplay(History, accuracy, val_accuracy, 'categorical_accuracy', 'val_categorical_accuracy', modelFileName + 'ModelAcc', 'accuracy', 'epoch', modelFileName + '_acc_history', round)
            historyDisplay(History, loss, val_loss, 'loss', 'val_loss', modelFileName + 'ModelLoss', 'loss', 'epoch', modelFileName + '_loss_history', round)

            hist_df = pd.DataFrame(History.history) 
            hist_df.to_csv('FL_'+ str(round+1) + '_' + modelFileName + '_Training_Validation_Loss_and_Accuracy.csv')
            
            #Test
            # os.system('python client_test_result.py')
            print('@@@@@@@@@@@@@@ Test Start @@@@@@@@@@@@@@@@@@@@@')
            # print(" Testing time: ",(end-start).total_seconds())
            model_load = load_model('FL_'+ str(round+1) + '_' + modelFileName + '.h5')
            adam = optimizers.Adam(lr=lr)
            model_load.compile(loss='categorical_crossentropy', 
            optimizer=adam, metrics=['categorical_accuracy'])
            loss, accuracy = model_load.evaluate(X_test1, Y_test1)
            print("\nTesting Loss: %f, Accuracy: %f" %  (loss, accuracy))

            predict = model_load.predict(X_test1)
            classes = predict.argmax(axis=-1)
            y_test_l = Y_test1.argmax(axis=-1)

            evaliation_self(y_test_l, classes, labelEncoder, rounds=round, loss=loss)

            plot_confusion_matrix(y_test_l, classes, classes=labelEncoder.classes_, title='Confusion matrix', fileName=modelFileName + '_confusion_matrix')
            ####要有output_dict=True才能轉成Dataframe
            report = classification_report(y_test_l, classes, target_names=labelEncoder.classes_, digits=4, output_dict=True)
            print(report)
            print(type(report))
            
            ####2022/04/18 change####
            del report['accuracy'], report['macro avg'], report['weighted avg'] #把最後的三個不重要的index丟掉
            tf.keras.backend.clear_session()

            report = pd.DataFrame(report).transpose()
            # report = pd.DataFrame(report)
            print(report.columns)
            print(report)
            time.sleep(5)
            ##### client : FL_次數_Report(第一次)
            ##### CTCGAN : FL_次數_GAN_次數_Report

            report.to_csv('FL_'+ str(round+1) + "_Report.csv")
            
            end = datetime.datetime.now()
            print(" The Running time: ",(end-start).total_seconds())
            train_time = {'Round': [round+1], 'Start_Time': [start], 'End_Time': [end], 'Total_Time': [(end-start).total_seconds()]}
            train_time = pd.DataFrame(data=train_time)
            if not os.path.exists('Running Time.csv'):
                train_time.to_csv('Running Time.csv', index=False)
            elif os.path.exists('Running Time.csv'):
                temp = pd.read_csv('Running Time.csv')
                temp = temp.append(train_time, ignore_index=True)
                temp.to_csv('Running Time.csv', index=False)

            ####2022/04/18 change####
            model_load.save_weights('after_last_client_1_weights.h5') #不同client須改

            print('@@@@@@@@@@@@@')
            print('Finished!!')
            print('@@@@@@@@@@@@@')
            del model
            del model_load
            tf.keras.backend.clear_session()
            round = round + 1