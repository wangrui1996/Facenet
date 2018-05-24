import time
import cv2
from scipy import misc
import os
import random
import string

__config = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data/config/camera.txt')

def read_camera_config(path=__config):  # function to set camera from config
    f = open(path,"r")
    lines = f.readlines()
    labels = []
    values = []
    number = 0
    for line in lines:
        list = line.split()
        if len(list) < 2:
            continue
        labels.append(list[0])
        values.append(list[1])
        number = number + 1
#    print(labels, values, number)
    return labels, values, number

class Camera:

    __faces = []
    __time = 0
    __interval = 120
    __delay = 0
    def __init__(self, label, path):
        try:
            path = int(path)
            print("open video from camra")
        except:
            print("open video from files")
        self.__cap = cv2.VideoCapture(path)
        self.__cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.__label = label
        self.__savePath = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output/' + self.getLabel())
        if not os.path.exists(self.__savePath):
            os.mkdir(self.__savePath)

    def capFrame(self):
        for i in range(self.__delay):
            self.__cap.grab()
        return self.__cap.read()

    def getLabel(self):
        return self.__label

    def showFrame(self):
        print(self.__label)

    def getNum(self):
        return len(self.__faces)

    def remark(self):
        self.__time = time.time()

    def addFace(self, faces):
        if len(faces) > self.getNum():
            self.remark()
            self.__faces = faces
        else:
            pass

    def clear(self):
        self.__faces.clear()

    def save(self):
        for img in self.__faces:
            path = self.__savePath + "/" + self.getLabel() + time.strftime('_%Y-%m-%d_%H:%M:%S_',time.localtime(time.time())) + ''.join(random.sample(string.ascii_letters + string.digits, 3)) + '.jpg'
            print(path)
            misc.imsave(path, img)
        self.clear()

    def progress(self):
        if time.time() - self.__time > self.__interval:
            self.save()
        else:
            pass




def CameraList(labels=0, values=0, number=0):
    if number == 0:
        labels, values, number = read_camera_config() # get labels, values, and number from camera config file
    list = []
    for i in range(number):
        list.append(Camera(labels[i], values[i]))
    return list, number

if __name__ == '__main__':
    labels, values, number = read_camera_config()
    list = CameraList(labels, values, number)
    for i in list:
        ret, frame = i.capFrame()
        while ret:
            cv2.imshow("demo", frame)
            cv2.waitKey(0)


