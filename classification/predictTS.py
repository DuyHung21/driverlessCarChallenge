import cv2
import numpy as np
import os
import scipy.misc
import pandas as pd
import tensorflow as tf
import pandas as pd
import mahotas
import imutils
import pickle

LABEL_MAP = {
    "0": 7,
    "1": 7,
    "2": 7,
    "3": 7,
    "4": 7,
    "5": 7,
    "6": 7,
    "7": 7,
    "8": 7,
    "14": 1,
    "34": 2,
    "33": 3,
    "43": 4,
    "44": 5,
    "17": 6,
    "45": 7,
    "46": 7,
    "48": 7
}
SIGN_NAMES = ["Dung", "Re trai", "Re phai", "Cam re trai", "Cam re phai", "Mot chieu", "Toc do", "Khac", "Khong phai"]

TARGET_SIGN = 2

INPUT_SHAPE = (32, 32, 1)


font                   = cv2.FONT_HERSHEY_SIMPLEX
fontScale              = 1
fontColor              = (255,255,255)
lineType               = 2
minW = 20
minH = 20
kernel = np.ones((3,3),np.uint8)

def rgb2gray(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)[:,:,0]
    return img

def bgr2gray(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.equalizeHist(img)
def resize_image(image, shape=INPUT_SHAPE[:2]):
    return cv2.resize(image, shape)

def is_ts(image):
    return True


def init():
    global detection_graph
    global sess
    global image_tensor
    global is_training
    global output_prediction
    global output_probability
    
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile('checkpoint/frozen_model.pb', 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    sess = tf.Session(graph=detection_graph)
    image_tensor = detection_graph.get_tensor_by_name('input/x:0')
    is_training = detection_graph.get_tensor_by_name('input/is_training:0')
    output_prediction = detection_graph.get_tensor_by_name('prediction/prediction:0')
    output_probability = detection_graph.get_tensor_by_name('prediction/probability:0')
    print ("Done init")
            
def predict(cropped):
    print ("Begin predict")
    print (cropped.shape)

    resize_img = resize_image(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
    resize_img = np.expand_dims(resize_img, axis=0)
    (prediction, probability) = sess.run(
        [output_prediction, output_probability],
        feed_dict={image_tensor: resize_img, is_training: False})
        #print (np.max(probability))
    if np.max(probability) > 0.9:
        #cv2.imshow("Cropped", cropped)
        print (int(prediction));
        if str(int(prediction)) in LABEL_MAP:
            name_id = LABEL_MAP[str(int(prediction))]
        else:
            print (int(prediction))
            name_id = 8
        return name_id
    else:
        return -1

def finalize():
    sess.close()
    





