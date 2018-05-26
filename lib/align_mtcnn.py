from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from lib import detect_face, facenet
import os
import cv2
import numpy as np
from scipy import misc
import random
import string


class FaceDetect:

    DESCRIPTION = "This is a Face detect class"
    __minsize = 20 # minimum size of face
    __threshold = [ 0.9, 0.9, 0.9 ]  # three steps's threshold
    __factor = 0.709
    __input = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data/facedir')
    __out = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data/middle')
    __videoout = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data/videoout')
    __model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data/model/mtcnn')
    def __init__(self, sess):
        self.__pnet, self.__rnet, self.__onet = detect_face.create_mtcnn(sess, self.__model_path)

    def setDetect(self, minsize, threshold, factor):
        self.__minsize = minsize
        self.__threshold = threshold
        self.__factor = factor

    def detetface(self, img):
        faceBound, point = detect_face.detect_face(img, self.__minsize, self.__pnet, self.__rnet, self.__onet, self.__threshold, self.__factor)
        return faceBound, point

    def translateface(self, input=__input, output=__out):
        criter = 249
        dataset = facenet.get_dataset(input)
        for cls in dataset:
            assert (len(cls.image_paths)>0, 'There must be at least one image for each class in the dataset')
        paths, labels = facenet.get_image_paths_and_labels(dataset)
        try:
            [os.mkdir(os.path.join(output, label)) for label in set(labels)]
        except:
            print('Please detect the folder under the {} first and try again'.format(output))
            return 0
        for i in range(len(paths)):
            path = paths[i]
            img = cv2.imread(path)
            w, h, d = img.shape
            if w > criter:
                img = cv2.resize(img, (int(h*criter/w), criter))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#            cv2.imshow("input", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
#            cv2.waitKey(0)

            box,_ = detect_face.detect_face(img, self.__minsize, self.__pnet, self.__rnet, self.__onet, self.__threshold, self.__factor)
            det = box[:, 0:4]
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            for j in range(np.shape(box)[0]):
                res = img[int(det[j][1]):int(det[j][3]), int(det[j][0]):int(det[j][2])]
#                cv2.imshow("result",res)
#                cv2.waitKey(0)
                savepath = os.path.join(output + "/" + labels[i], path.split('/')[-1])
                #print(savepath)
                cv2.imwrite(savepath, res)

    def translatevideo(self, cap, videoout=__videoout):
        print("output dir is {}".format(videoout))
#        if not os.path.exists(video):
#            print("file is not exit, and use camera")
#            video = 0
        ret, frame = cap.read()
        sub = 0
        top = 0
        while ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            box, pot = detect_face.detect_face(frame, self.__minsize, self.__pnet, self.__rnet, self.__onet, self.__threshold, self.__factor)
            try:
                length = np.shape(pot)[1]
            except:
                length = 0
            for i in range(length):
                prefix = ''.join(random.sample(string.ascii_letters + string.digits, 8))
                det = box[i, 0:4]
                img = frame[int(det[1]):int(det[3]), int(det[0]):int(det[2])]
                savepath = os.path.join(videoout + "/" + prefix + str(sub) + ".jpg")
                misc.imsave(savepath, img)
                sub = sub + 1
                print("top save the number of photo is {}".format(sub))

            ret, frame = cap.read()
            top = top + 1
            print("top to progress the number of fps is {}".format(top))


