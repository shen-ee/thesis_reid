# -*- coding: utf-8 -*-
import cv2
import numpy as np
import os

IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224

dataset_dir = "/Users/shenyi/Desktop/thesisCode/data/cuhk03_release/labeled/val/"

def format_id(id):
    if(id<10):
        return "000"+str(id)
    if(id<100):
        return "00"+str(id)
    if(id<1000):
        return "0"+str(id)

def get_img(id):
    query_imgs = []
    query_labels = []
    for person_id in range(id,id+1):
        for photo_id in range(0,1):
            img_dir = dataset_dir+format_id(person_id)+"_0"+str(photo_id)+".jpg"
            if(os.path.exists(img_dir)):
                query_img = cv2.imread(img_dir)
                query_img = cv2.resize(query_img, (IMAGE_WIDTH, IMAGE_HEIGHT))
                query_img = cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB)
                query_img = np.reshape(query_img, (IMAGE_HEIGHT, IMAGE_WIDTH, 3)).astype(float)

                query_imgs.append(query_img)
                query_labels.append(person_id)

                print(person_id,photo_id)

    test_imgs = []
    test_labels = []
    for person_id in range(100):
        for photo_id in range(5,6):
            img_dir = dataset_dir+format_id(person_id)+"_0"+str(photo_id)+".jpg"
            if(os.path.exists(img_dir)):
                test_img = cv2.imread(img_dir)
                test_img = cv2.resize(test_img, (IMAGE_WIDTH, IMAGE_HEIGHT))
                test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
                test_img = np.reshape(test_img, (IMAGE_HEIGHT, IMAGE_WIDTH, 3)).astype(float)

                test_imgs.append(test_img)
                test_labels.append(person_id)

                print(person_id,photo_id)
    return np.asarray(query_imgs, np.float32),np.asarray(test_imgs, np.float32),\
           np.asarray(query_labels, np.int32),np.asarray(test_labels, np.int32)
