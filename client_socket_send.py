import socket
import os
import sys
import struct
import time

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

host = '120.108.205.9'  # 远程socket服务器ip
port = 50001
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # 实例化socket
s.connect((host, port))  # 连接socket服务器

#傳檔案
####2022/04/18 change####
if os.path.exists('last_client_1_weights.h5'):
    filepath = 'last_client_1_weights.h5' #不同client須改
else:
    filepath = 'client_1_weights.h5' #不同client須改


if os.path.isfile(filepath):
    SendFile()
    
data = s.recv(1024).strip()
print(data.decode())

print('Send to Global')
print(str(data.decode()))

print('@@@@@')
print('Send Finished!')
print('@@@@@')

s.close()
