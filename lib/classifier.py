"""An example of how to use your own dataset to train a classifier that recognizes people.
"""
# MIT License
# 
# Copyright (c) 2016 David Sandberg
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from lib import facenet
import os
import math
import pickle
from sklearn.svm import SVC


class Classifier:

    __featuremodel = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data/model/20180408-102900')
    __classmodel = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data/model/svm/model')
    __facedir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data/middle')
    __mode = 0

    __batch_size = 10
    __image_size = 160

    #  mode 0: 分类 欧氏距离
    #  mode 1: 分类 SVM
    #  mode 3: 训练 SVM
    def __init__(self, sess, modelpath=__featuremodel, facedir=__facedir, classmodel=__classmodel, mode=__mode):
         #Load the model
         self.__mode = mode
         print(self.__mode)
         self.sess = sess
         print('Loading feature extraction model')
         facenet.load_model(modelpath)
         # Get input and output tensors
         self.images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
         self.embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
         self.phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
         self.embedding_size = self.embeddings.get_shape()[1]

         if mode == 0:
             facedir = facenet.get_dataset(facedir)
             self.paths, self.labels = facenet.get_image_paths_and_labels(facedir)
             self.imageslen = len(self.paths)
             print("calculating features from user face dataset")
             nrof_images = len(self.paths)
             nrof_batches_per_epoch = int(math.ceil(1.0*nrof_images / self.__batch_size))
             self.emb_array = np.zeros((nrof_images, self.embedding_size))
             print(self.embedding_size)
             for i in range(nrof_batches_per_epoch):
                 start_index = i * self.__batch_size
                 end_index = min((i+1)*self.__batch_size, nrof_images)
                 paths_batch = self.paths[start_index:end_index]
                 images = facenet.load_data(paths_batch, False, False, image_size=self.__image_size)
                 feed_dict = { self.images_placeholder:images, self.phase_train_placeholder:False }
                 self.emb_array[start_index:end_index,:] = self.sess.run(self.embeddings, feed_dict=feed_dict)
       #         print(self.emb_array[0,0])
       #         print(self.emb_array[1,0])
       #         print(self.emb_array[2,0])
             print("finished calculating features from user face dataset")
         elif mode == 1:
             with open(classmodel, 'rb') as infile:
                 (self.model, self.class_names) = pickle.load(infile)
         else:
             print("mode of input is not classifier, next we will start to train the SVM or do nothing")

    def classifier(self, image_path):
#        img = np.zeros((1, 160, 160, 3))
#        img[0, :, :, :] = image
        length = len(image_path)
        print("Calculating features for images")
        images = facenet.load_data(image_path, False, False, image_size=self.__image_size)
        #images = image[np.newaxis, :, :, :]
        emb_array = np.zeros((length, self.embedding_size))
        feed_dict = { self.images_placeholder:images, self.phase_train_placeholder:False}
        emb_array = self.sess.run(self.embeddings, feed_dict=feed_dict)

        print("finished extract features for images")

        #print()

        print(self.__mode)
        if self.__mode == 0:
            print("yes")
            face_array = np.reshape(self.emb_array, (1, -1))
            face_array = np.tile(face_array, (length, 1))
            emb_array = np.tile(emb_array, (1, self.imageslen))
            res = emb_array - face_array
    #        print(input_image[0,0],  face_array[0,0])
    #        print(input_image[0,512], face_array[0, 512])
    #        print(input_image[0,1024], face_array[0, 1024])
    #        print(np.shape(input_image))
    #        print(np.shape(face_array))
    #        print(input_image)
    #        print(face_array)
            res = pow(res, 2)
            #for i in range(512, 1000):
            #    print(res[0,i])
            mid = [ np.reshape(np.sum(res[:,i*512:(i+1)*512], 1), (-1, 1)) for i in range(self.imageslen)]

            #print(mid)
            res = mid[0]
            for i in range(1, self.imageslen):
                res = np.append(res, mid[i], 1)
            aix = np.argmin(res, 1)
            values = np.min(res, 1)
            labels = [self.labels[i] for i in aix]
    #        print(labels, value)

        else:
            predictions = self.model.predict_proba(emb_array)
            print(predictions)
            best_class_indices = np.argmax(predictions, axis=1)
            best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
            print(best_class_probabilities)
            labels = [self.class_names[i] for i in best_class_indices]
            values = best_class_probabilities
#            for i in range(len(best_class_indices)):
#                print('%4d $s: %.3f' % (i, self.class_names[best_class_indices[i]], best_class_probabilities[i]))

        return labels, values
        #classifier_filename_exp = os.path.expanduser(args.classifier_filename)
        # Classify images
        #print('Loaded classifier model from file "%s"' % classifier_filename_exp)


    def train(self, facedir=__facedir, image_size=160, batch_size=90):
        dataset = facenet.get_dataset(facedir)
        paths, labels = facenet.get_image_paths_and_labels(dataset)

        print("Number of classes: %d" % len(dataset))
        print("Number of images %d" % len(paths))

        nrof_images = len(paths)
        nrof_batchs_per_epoch = int(math.ceil(1.0*nrof_images / batch_size))
        emb_array = np.zeros((nrof_images, self.embedding_size))
        for i in range(nrof_batchs_per_epoch):
            start_index = i*batch_size
            end_index = min((i+1)*batch_size, nrof_images)
            paths_batch = paths[start_index:end_index]
            images = facenet.load_data(paths_batch, False, False, image_size)
            feed_dict = { self.images_placeholder:images, self.phase_train_placeholder:False }
            emb_array[start_index:end_index,:] = self.sess.run(self.embeddings, feed_dict=feed_dict)
        print("Training classifier")
        model = SVC(kernel='linear', probability=True)
        model.fit(emb_array, labels)

        class_names = [ cls.name for cls in dataset ]

        with open(self.__classmodel, 'wb') as outfile:
            pickle.dump((model, class_names), outfile)

        print("Saved classifier model too file '%s'" % self.__classmodel)



if __name__ == '__main__':
#    image = cv2.imread("/home/rui/Files/PycharmProjects/classfier/data/img/2.png")
#    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#    image = cv2.resize(image, (160, 160))
    images = []
    img = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data/middle/hu/demo.jpg')
    images.append(img)
    print(images)
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            classifier = Classifier(sess, mode=2)
            classifier.train()
#            labels, values = classifier.classifier(images)

 #   for i in range(len(labels)):
  #      print(labels[i], values[i])

