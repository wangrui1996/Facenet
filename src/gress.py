import cv2
from src.client import Client
from lib.align_mtcnn import FaceDetect
from lib.classifier import Classifier
import tensorflow as tf
import numpy as np
import json
from src.camera import CameraList


def reg_face(frame, detect, classifier, client, cera, mode):
    cols, rows, _= np.shape(frame)
    bounding_boxes, pot = detect.detetface(frame)    # get the face bounding, and key point
    det = bounding_boxes[:, 0:4]     # get the boxes
#    score = bounding_boxes[:, 4]    # get the score of boxes, but we not need.

    font = cv2.FONT_HERSHEY_DUPLEX    # set the typeface
    face_images = []
    num,_ = det.shape              # get the number of detect face
    if num > 0:
        for i in range(num):
            det[i][1] = max(0, det[i][1])
            det[i][0] = max(0, det[i][0])
            res = frame[int(det[i][1]):int(det[i][3]), int(det[i][0]):int(det[i][2])]    # get the image of face in each frame
             #   result = cv2.cvtColor(res, cv2.COLOR_RGB2BGR)
             #        cv2.imshow("demo", res)
             #        cv2.imwrite(os.path.join("/home/rui/wangrui", str(sub) + '.jpg'), result)
            cv2.rectangle(frame, (int(det[i][0]), int(det[i][1])), (int(det[i][2]), int(det[i][3])), (0, 255, 0))  # draw the face in frame
            #    print(det[i][0], det[i][1], det[i][2], det[i][3])
            face_images.append(res)
             #            cv2.imshow("face",  res)
             #            cv2.putText(frame, str(score[i]), (int(det[i][0]), int(det[i][1])), font, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        labels, values = classifier.classifier(face_images)  # translate the face and get the labels and scores
        for i in range(num):

                #Letter = cera + " " + labels[i] + " 100 " + str(int(det[i][0])) + " " + str(int(det[i][1]))
            if mode == 0:
                if values[i] < 0.6:     # if we set mode is 0, we need the values more less, and result will more confience

                    #flag = True

                    #data = cera + " " + labels[i] + " 100 " + str(int(det[i][0])) + " " + str(int(det[i][1]))

                    # calcu distance

    #                distance =  165.0 * rows / float(det[i][2] - det[i][0]) * 8 / 4.5056 / 1000   # get distance between camera and people, if you need
    #                Letter = str(labels[i]) + " | " + str(values[i]) + " | " + str(distance)
                    Letter = str(labels[i]) + " | " + str(values[i])
                    cv2.putText(frame, Letter, (int(det[i][0]), int(det[i][1])), font, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
                    ''' if we need send data , we can use it, but now we needn't
                    axis_x = (int(det[i][0]) + int(det[i][2]))//2
                    axis_y = cols - (int(det[i][1]) + int(det[i][3]))//2
    #                    print(axis_x, axis_y)
                    data = {
                        'camera' : cera.getLabel(),
                        'people' : labels[i],
                        'distance' : distance,
                        'x' : axis_x,
                        'y' : axis_y,
                    }
                    json_str = json.dumps(data)   #  
                    client.senddata(json_str)
                    '''
                else:
                    pass
            else:   # if we classifier it by SVM, we need the values more bigger , and the result will more confience
                if values[i] > 0:

                    #flag = True

                    #data = cera + " " + labels[i] + " 100 " + str(int(det[i][0])) + " " + str(int(det[i][1]))

                    # calcu distance

    #                distance =  165.0 * rows / float(det[i][2] - det[i][0]) * 8 / 4.5056 / 1000   # get distance between camera and people, if you need
    #                Letter = str(labels[i]) + " | " + str(values[i]) + " | " + str(distance)
                    Letter = str(labels[i]) + " | " + str(values[i])
                    cv2.putText(frame, Letter, (int(det[i][0]), int(det[i][1])), font, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
                    ''' the same as pre
                    axis_x = (int(det[i][0]) + int(det[i][2]))//2
                    axis_y = cols - (int(det[i][1]) + int(det[i][3]))//2
    #                    print(axis_x, axis_y)
                    
                    data = {
                        'camera' : cera.getLabel(),
                        'people' : labels[i],
                        'distance' : '100',
                        'x' : axis_x,
                        'y' : axis_y,
                    }
                    json_str = json.dumps(data)
                    client.senddata(json_str)
                    '''
                else:
                    pass
            #cols, rows = np.shape(pot)
            #for i in range(rows):
            #    for j in range(5):
            #        cv2.rectangle(frame, (int(pot[j][i]), int(pot[j+5][i])), (int(pot[j][i]), int(pot[j+5][i])), (0, 255, 0))
    return frame






def progress(port, mode):
#    print(cap)

    cameList, number = CameraList()     # set the input, from camera or Video

    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            detect = FaceDetect(sess)  # face detect class
            classifier = Classifier(sess, mode=mode)   # face recognition class
    if(mode == 2):
        classifier.train()        # if mode is 2, train svm
        return 0
    client = Client(port)         # if we need send data by UDP, set it.
    while True:
        for cam in cameList:              # from camera get frame and progress it in detect and recognition face
            _, frame = cam.capFrame()     # get face from class camera
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)   # BGR into RGB
            res = reg_face(frame, detect, classifier, client, cam, mode)  # main progress
            res = cv2.cvtColor(res, cv2.COLOR_RGB2BGR)    # RGB into BGR
            cv2.imshow(cam.getLabel(), res)    # show result
            cv2.waitKey(1)
#            ret, frame = cap.read()

#            classifier.classifier(image)


def GetFace():
    print("How to get face?")
    mode = input("1: From Camera, 2: From Video, 3: Transtale facedir into middle")
    mode = int(mode)
    if mode == 2:
        video = input("Please input video path: ")
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            detect = FaceDetect(sess)
            if mode == 1:
                cap = cv2.VideoCapture(0)
                detect.translatevideo(cap)
            elif mode == 2:
                if (len(video) == 0):
                    print("No Input, and exit")
                    exit(1)
                else:
                    cap = cv2.VideoCapture(video)
                detect.translatevideo(cap)
            else:
                detect.translateface()





if __name__ == '__main__':
    import os
    path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'cera/1.avi')
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            detect = FaceDetect(sess)
            classifier = Classifier(sess)
    cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture(path)
    ret, frame = cap.read()
    client = Client()
    while ret:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = reg_face(frame, detect, classifier, client, "demo")
        res = cv2.cvtColor(res, cv2.COLOR_RGB2BGR)
        cv2.imshow("demo", res)
        cv2.waitKey(1)
        ret, frame = cap.read()

