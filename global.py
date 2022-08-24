import os
from tkinter.tix import Tree
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers import Adam
from fl_mnist_implementation_tutorial_utils_改_Separated import *
print('------------------------------------------------------------')
import logging
import tensorflow as tf
import numpy as np
import pandas as pd
import time

from numba import cuda

#FORMAT = '[%(levelname)s]: %(asctime)-15s %(message)s'
#logging.basicConfig(format=FORMAT, level=logging.INFO)

def readTrain():

    trainFilePath = 'Datasets/141_7_train_withOtherFeature_V2(seperate)_withoutHeader.csv'
    columnName = [str(i) for i in range(10)] + ['type']
    train = pd.read_csv(trainFilePath, names=columnName)
    # train.drop(['0'], axis=1, inplace=True)

    return train

def choose_model(modelFileName, shape, classes, optimizer):
    if modelFileName == 'MLPModel':
        global_model = buildMLPModel(shape, classes, optimizer)
        logging.info(modelFileName)
        return global_model

    elif modelFileName == 'SAEModel':
        global_model = buildSAEModel(shape, classes, optimizer)
        logging.info(modelFileName)
        return global_model

    elif modelFileName == 'CNNModel':
        global_model = buildCNNModel(shape, classes, optimizer)
        logging.info(modelFileName)
        return global_model

    elif modelFileName == 'LSTMModel':
        global_model = buildLSTMModel(shape, classes, optimizer)
        logging.info(modelFileName)
        return global_model

print('')
print('@@@@@@@@@@@@@@  Starting Loading  @@@@@@@@@@@@@@@@')
print('')

### Control GPU used memory
gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

train = readTrain()
logging.info((
            'src_ipUnique ', len(train['0'].unique()),
            'dst_ipUnique ', len(train['1'].unique()),
            'server_portUnique ', len(train['2'].unique()),
            'proto_Unique ', len(train['3'].unique()),
            'total ', len(train['0'].unique())+
                    len(train['1'].unique())+
                    len(train['2'].unique())+
                    len(train['3'].unique())))

hash_vector_size1 = len(train['0'].unique())
hash_vector_size2 = len(train['1'].unique())
hash_vector_size3 = len(train['2'].unique())
hash_vector_size4 = len(train['3'].unique())

# hash_vector_size1 = 2
# hash_vector_size2 = 1000
# hash_vector_size3 = 1000
# hash_vector_size4 = 300

# seed_value= 42
# import numpy as np
# np.random.seed(seed_value)
# import random
# random.seed(seed_value)

statistic_feature = 6

feature_size = hash_vector_size1 + hash_vector_size2 + hash_vector_size3 + hash_vector_size4 + statistic_feature

modelFileName = 'MLPModel'

lr = 0.001
labels = 25 #Important##############################
local_epoch = 100
GAN_train_epoch = 50
CTC_GAN_client_1 = True #True False
CTC_GAN_client_2 = True
CTC_GAN_client_3 = True
do_GAN_first = 1 #### chose round that need to do GAN, 0 ~ 14 -> 1 ~ 15
do_GAN_second = 1 ### Two parameters control which round should do GAN
remainder_num = 0
stop_point = 3 #GAN訓練次數的卡點

clients = 3



if modelFileName == 'CNNModel':
    local_batch_size = 512
else:
    local_batch_size = 4000
    ## Client 目前先改成2000(於本地端)，測試是否能防止出錯

comms_round = 15
optimizers_Name = 'Adam'

if optimizers_Name == 'SGD':
    optimizer = SGD(lr=lr, 
                    decay=lr / comms_round,  
                    momentum=0.9
                ) 
elif optimizers_Name == 'Adam':
    optimizer = Adam(lr=lr, 
                    # decay=lr / comms_round, 
                    # momentum=0.9          ### Adam沒有該超參數
                ) 

parameters = {'src_ipUnique': [hash_vector_size1], 'dst_ipUnique': [hash_vector_size2],
                'server_portUnique': [hash_vector_size3], 'proto_Unique': [hash_vector_size4],
                'ModelFileName': [modelFileName], 'feature_size' : [feature_size],
                'local_epoch': [local_epoch], 'local_batch_size': [local_batch_size],
                'lr': [lr], 'labels': [labels], 'CTC-GAN_client_1': [CTC_GAN_client_1],
                'CTC-GAN_client_2': [CTC_GAN_client_2], 'CTC-GAN_client_3': [CTC_GAN_client_3],'GAN_train_epoch': [GAN_train_epoch],
                'stop_point': [stop_point], 'do_GAN_first': [do_GAN_first], 
                'do_GAN_second': [do_GAN_second], 'remainder_num': [remainder_num]
                }

parameters_Frame = pd.DataFrame(data=parameters)
print(parameters_Frame)
parameters_Frame.to_csv('parameters_Frame.csv', index=False)

try:
    os.system('python global_socket_send.py')
    print('Parameters File Send Successfully!')
    # print('Deleting Parameters File!')
    # os.remove('parameters_Frame.csv')
except Exception as e:
    print(e)
    time.sleep(3)


global_model = choose_model(modelFileName=modelFileName, shape=feature_size, classes=labels, optimizer=optimizer)
logging.info(modelFileName)

scaled_local_weight_list = list()


flag = False
comm_round = 0
local_file = False
open_file = False

####2022/04/18 change####
final_saturation = 0

Total_Start = datetime.datetime.now()

while True:
    start = datetime.datetime.now()
    
    logging.info('Global Round:' + str(comm_round+1))

    if flag == False:
        # start = datetime.datetime.now()
    
        global_model.save_weights('global_weights.h5')
        print('@@@@@@@@@@@@@@')
        print('Global Round: ', comm_round+1)
        print('@@@@@@@@@@@@@@')

        try:
            os.system('python global_socket_send.py')
            print('Global Send Successfully!')
        except Exception as e:
            print(e)
            time.sleep(3)

        del global_model
        tf.keras.backend.clear_session()

        src= './global_weights.h5'
        des= './Global_Model/'+ str(comm_round) + '_global_weights.h5'

        os.replace(src,des)

        flag = True
    if flag == True:

        # start = datetime.datetime.now()
        
        global_model = choose_model(modelFileName=modelFileName, shape=feature_size, classes=labels, optimizer=optimizer)
        logging.info(modelFileName)

        for client in range(clients):
            while local_file != True:
                local_file = check_exists(client)
                if local_file == False:
                    print('Global Wait!')
                    print(comm_round+1)
                    time.sleep(3)
                else:
                    print('Got ' + str(client+1) + ' the file!')
                    
            local_file = False
            #load_weight需要有個完整的模型，就像使用get_weight的方式，不能像load_model()是一個函式

            while open_file == False:  
                try:
                    print('Loading weights!!!')
                    ####2022/04/18 change####
                    if os.path.exists('last_client_' + str(client+1) + '_weights.h5'):
                        global_model.load_weights('last_client_' + str(client+1) + '_weights.h5')
                        final_saturation = final_saturation + 1
                        print('Loading saturation clients ' + str(client+1) + ' weight!!!')
                        print('Saturation Count: ', final_saturation)
                        time.sleep(3)
                    else:
                        global_model.load_weights('client_'+ str(client+1) + '_weights.h5')
                        print('Loading clients ' + str(client+1) + ' weight!!!')
                        time.sleep(3)

                    open_file = True
                except Exception as e:
                    open_file = False
                    print(e)
                    time.sleep(3)

            open_file = False

            local_weight = global_model.get_weights()
            local_weight = scale_model_weights(local_weight, 1/clients)
            scaled_local_weight_list.append(local_weight)
            

        average_weights = sum_scaled_weights(scaled_local_weight_list)
        global_model.set_weights(average_weights)

        ################# 2022.03.09 修正 #################
        scaled_local_weight_list.clear()
        ################# 2022.03.09 修正 #################
        
        # for client in range(clients):
        #     if os.path.exists('last_client_' + str(client+1) + '_weights.h5'):
        #         final_saturation = final_saturation + 1

        ####2022/04/18 change####
        if final_saturation != clients:

            global_model.save_weights('global_weights.h5')

            print('@@@@@@@@@@@@@@')
            print('Global Round: ', comm_round+2) 
            print('@@@@@@@@@@@@@@')
            
            for client in range(clients):
                # if os.path.exists('last_client_' + str(client+1) + '_weights.h5'):
                #     os.remove('last_client_' + str(client+1) + '_weights.h5')
                # else:
                try:
                    os.remove('client_'+ str(client+1) + '_weights.h5')
                except Exception as e:
                    print(e)

            try:
                os.system('python global_socket_send.py')
                print('Global Send Successfully!')

                # os.remove('global_weights.h5')
                src= './global_weights.h5'
                des= './Global_Model/'+ str(comm_round+1) + '_global_weights.h5'

                os.replace(src,des)

            except Exception as e:
                print(e)
                time.sleep(3)
            
            time.sleep(5)
            
            ###如果要做第四章的東西，需要註解調下方的del
            del global_model
            tf.keras.backend.clear_session()
            
            ####2022/04/18 change####
            final_saturation = 0
            
            end = datetime.datetime.now()
            
            path = 'Running Time'+ str(comm_round+1) +'.txt'
            f = open(path, 'w')
            f.write('GR: '+ str(comm_round+2) + ', time is : ' + str((end-start).total_seconds()))
            f.close()

        ####2022/04/18 change####
        elif final_saturation == clients:
            global_model.save_weights('last_global_weights.h5')
            print('@@@@@@@@@@@@@@')
            print('Last Round: ', comm_round+2, ' to send back, only for updating!')
            print('@@@@@@@@@@@@@@')
           
            try:
                os.system('python global_socket_send.py')
                print('Global Send Successfully!')

                src= './last_global_weights.h5'
                des= './Global_Model/last_global_weights.h5'

                os.replace(src,des)

            except Exception as e:
                print(e)
                time.sleep(3)
            
            time.sleep(5)
            
            ###如果要做第四章的東西，需要註解調下方的del
            del global_model
            tf.keras.backend.clear_session()

            for client in range(clients):
                try:
                    if os.path.exists('last_client_' + str(client+1) + '_weights.h5'):
                        os.remove('last_client_' + str(client+1) + '_weights.h5')
                    else:
                        os.remove('client_'+ str(client+1) + '_weights.h5')
                except Exception as e:
                    print(e)
            
            print('----------- Time to Break!-----------')
            time.sleep(3)
            
            end = datetime.datetime.now()
            
            path = 'Running Time'+ str(comm_round+1) +'.txt'
            f = open(path, 'w')
            f.write('GR: '+ str(comm_round+2) + ', time is : ' + str((end-start).total_seconds()))
            f.close()
            
            break
    
    ##For chepter 4, test GR:100, localR:1 to compare with centralized on different model
    # if (comm_round+1) == 100:
    #     global_model.save_weights('last_global_weights.h5')
    #     print('@@@@@@@@@@@@@@')
    #     print('Last Round: ', comm_round, ' to send back, only for updating!')
    #     print('@@@@@@@@@@@@@@')
        
    #     try:
    #         os.system('python global_socket_send.py')
    #         print('Global Send Successfully!')

    #     except Exception as e:
    #         print(e)
    #         time.sleep(3)
        
    #     del global_model
    #     tf.keras.backend.clear_session()

    #     for client in range(clients):
    #         try:
    #             if os.path.exists('last_client_' + str(client+1) + '_weights.h5'):
    #                 os.remove('last_client_' + str(client+1) + '_weights.h5')
    #             else:
    #                 os.remove('client_'+ str(client+1) + '_weights.h5')
    #         except Exception as e:
    #             print(e)
        
    #     print('----------- Time to Break!-----------')
    #     time.sleep(3)
    #     break

    comm_round = comm_round + 1

####2022/04/18 change####
Total_End = datetime.datetime.now()
if not os.path.exists('Running Time.txt'):
    path = 'Running Time.txt'
    f = open(path, 'w')
    f.write('GR: '+ str(comm_round+2) + ', time is : ' + str((Total_End-Total_Start).total_seconds()))
    f.close()
