import socketserver, sys, threading
import os
import struct
from time import ctime
import time


class ThreadedTCPRequestHandler(socketserver.BaseRequestHandler):
    # 继承BaseRequestHandler基类，然后必须重写handle方法，并且在handle方法里实现与客户端的所有交互
    def handle(self):
        
        
        cur = threading.current_thread()
        print('[%s] Client connected from %s and [%s] is handling with him.' % (ctime(), self.request.getpeername(), cur.name))

        print('connected by', str(self.client_address))           
            
        fileinfo_size = struct.calcsize('128sl')
        buf = self.request.recv(fileinfo_size)
        if buf:
            filename, filesize = struct.unpack('128sl', buf)
            fn = filename.decode().strip('\00')
            new_filename = os.path.join('./', fn)
            print ('File Name: {0}, filesize if {1}'.format(new_filename,
                                                                filesize))

            fp = open(new_filename, 'wb')
            print ('Starting Upload...')
            recvd_size = 0
            while not recvd_size == filesize:
                if filesize - recvd_size > 1024:
                    data = self.request.recv(1024)
                    recvd_size += len(data)
                    print(recvd_size)
                else:
                    data = self.request.recv(filesize - recvd_size)
                    recvd_size += len(data)
                    print('Remain Size:' , len(data))
                    print('recvd_size: ', recvd_size)
                fp.write(data)
            fp.close()
            print ('File Already Upload...')
             
        self.request.send(str('Global Got the File').encode())
        print('Cleint file recive completely!')
                    
        
        print(str(self.client_address), ' closed the connect!')
        self.request.close()

class ThreadedTCPServer(socketserver.ThreadingTCPServer, socketserver.TCPServer):
    daemon_threads = True
    allow_reuse_address = True

if __name__ == "__main__":
    host, port = "120.108.205.9", 50001

    server = ThreadedTCPServer((host, port), ThreadedTCPRequestHandler)
    ip, port = server.server_address
    print('Global Server start at: %s:%s' % (host, port))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        sys.exit(0)