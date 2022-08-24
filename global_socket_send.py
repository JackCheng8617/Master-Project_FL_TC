import socket
import os
import sys
import struct
import time

def check_exists():
    last = 0
    if os.path.exists('last_global_weights.h5'):
        last = 1
    elif os.path.exists('global_weights.h5'):
        last = 0
    elif os.path.exists('parameters_Frame.csv'):
        last = 2
    return last

def SendFile():
    fileinfo_size = struct.calcsize('128sl')
    
    fhead = struct.pack('128sl', os.path.basename(filepath).encode('utf-8'),
                        os.stat(filepath).st_size)
    s.send(fhead)
    print ('client filepath: {0}'.format(filepath))

    fp = open(filepath, 'rb')
    while 1:
        data = fp.read(1024)
        if not data:
            print ('{0} File Upload...'.format(filepath))
            break
        s.send(data)

host = ['120.108.205.26', '120.108.205.24', '120.108.205.11']  # 远程socket服务器ip
port = [50002, 50003, 50004]
# host = ['120.108.205.9']  # 远程socket服务器ip
# port = [50004]

for i in range(len(port)):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # 实例化socket
    s.connect((host[i], port[i]))  # 连接socket服务器

    #傳檔案
    if check_exists() == 0:
        filepath = 'global_weights.h5'
    elif check_exists() == 1:
        filepath = 'last_global_weights.h5'
    elif check_exists() == 2:
        filepath = 'parameters_Frame.csv'
        print('Send the parameters!!!')

    if os.path.isfile(filepath):
        SendFile()
        
    data = s.recv(1024).strip() #接收Client回傳的字串，判斷是否要重傳
    print('Send to Client ' + str(i+1))
    print(str(data.decode()))

    print('@@@@@')
    print('Send Finished!')
    print('@@@@@')
    s.close()
