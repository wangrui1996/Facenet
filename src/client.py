import socket
import socketserver
import time
import threading
import time

class Client:
    __port = 9999
    __host = "127.0.0.1"
    def __init__(self, port=__port, host=__host):
        try:
            self.addr = (host, port)
            self.serverSocket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            #self.serverSocket.bind(('', port))

            #self.serverSocket.listen(1)
            print("serverPort number is {}, and ready to send data!".format(port))
        except:
            print("The number of {} is used, please change!".format(port))

    def senddata(self, data):
        data = data.encode()
        self.serverSocket.sendto(data, self.addr)
        print("finishing get data {}".format(data))

#        connectionSocket, addr = self.serverSocket.accept()
#        connectionSocket.sendall(data)
#        connectionSocket.send(data.encode())
        pass

