# coding:utf-8
from tkinter import *
from PIL import Image, ImageTk
import tools
import tensorflow as tf
import cuhk03_dataset
import cv2
import numpy as np
import codecs

import time
import datetime

from keras import optimizers
from keras.utils import np_utils, generic_utils
from keras.models import Sequential,Model,load_model
from keras.layers import Dropout, Flatten, Dense,Input
from keras.applications.resnet50 import ResNet50
from keras.applications.imagenet_utils import preprocess_input
from keras import backend as K
from keras.layers.core import Lambda
from sklearn.preprocessing import normalize
from keras.preprocessing.image import ImageDataGenerator
from keras.initializers import RandomNormal
import keras

import numpy.linalg as la

import cuhk03_api
'''
这个界面是gihub:https://github.com/shenyi1028/keras_reid 的拓展界面
'''

dataset_dir = "/Users/shenyi/Desktop/thesisCode/data/cuhk03_release/labeled/val/"
model_dir = "/Users/shenyi/Documents/GitHub/person-reid/logs/"
dataset_name = "CUHK03"


class App:
    def __init__(self):
        #  界面
        self.current_person_id_cam1 = -1
        # self.current_person_id = -1
        self.current_img_id_cam1 = 0
        # self.current_img_id_cam2 = 0

        self.root = Tk()
        self.root.title('Person Re-id')
        self.frame0 = Frame(self.root)
        self.frame1 = Frame(self.root)
        self.frame2 = Frame(self.root)
        self.frame_match = Frame(self.root)  # 显示匹配结果

        # frame0
        label_dataset_name = Label(self.frame0,text="数据集名称："+dataset_name)
        label_dataset_dir = Label(self.frame0,text="数据集路径："+dataset_dir)
        label_model_dir = Label(self.frame0,text="模型路径："+model_dir)
        label_id1 = Label(self.frame0,text="选择id号:")
        self.entry_id1 = Entry(self.frame0)
        button_confirm1 = Button(self.frame0,text="确认",command = self.confirm1)
        button_confirm2 = Button(self.frame0, text="开始重识别", command=self.confirm_reid)

        # frame1
        self.label_person_image_cam1 = Label(self.frame1)
        self.label_filename1 = Label(self.frame1)
        self.button_next1 = Button(self.frame1, text="下一张", command=self.next1)

        # frame2
        self.label_person_image_cam2 = Label(self.frame2)
        self.label_filename2 = Label(self.frame2)
        self.button_next2 = Button(self.frame2, text="下一张", command=self.next2)

        # frame_match
        self.label_match = Label(self.frame_match,text="匹配结果：+++")
        self.label_confidence = Label(self.frame_match,text="置信度：")
        self.text_console = Text(self.frame_match)

        self.frame0.grid(row = 0,columnspan=2)
        self.frame1.grid(row = 1,column = 0)
        self.frame2.grid(row=1, column=1)
        self.frame_match.grid(row=2,columnspan=2)

        label_dataset_name.grid(row=0,sticky=W,columnspan=6)
        label_dataset_dir.grid(row=1,sticky=W,columnspan=6)
        label_model_dir.grid(row=2,sticky=W,columnspan=6)
        label_id1.grid(row = 3,sticky = W)
        self.entry_id1.grid(row = 3,column = 1)
        button_confirm1.grid(row = 3,column=2)
        button_confirm2.grid(row=3, column=5)

        self.label_person_image_cam1.pack()
        self.label_filename1.pack()
        self.button_next1.pack()

        self.label_person_image_cam2.pack()
        self.label_filename2.pack()
        self.button_next2.pack()

        self.label_match.grid(row=0,sticky=W)
        self.label_confidence.grid(row=1,sticky=W)
        self.text_console.grid(row=2,rowspan=3)


        # keras_start

        identity_num = 6273

        base_model = ResNet50(weights='imagenet', include_top=False, input_tensor=Input(shape=(224, 224, 3)))
        x = base_model.output
        feature = Flatten(name='flatten')(x)
        fc1 = Dropout(0.5)(feature)
        preds = Dense(identity_num, activation='softmax', name='fc8',
                      kernel_initializer=RandomNormal(mean=0.0, stddev=0.001))(fc1)  # default glorot_uniform
        net = Model(input=base_model.input, output=preds)
        feature_model = Model(input=base_model.input, output=feature)
        self.class_triplet_model = Model(input=base_model.input, output=[preds, feature])
        self.class_triplet_model.load_weights('model/naivehard_more_last.h5')

        # keras_end

        self.root.mainloop()

    def euclidSimilar2(self,query_ind, test_all):
        le = len(test_all)
        dis = np.zeros(le)
        for ind in range(le):
            sub = test_all[ind] - query_ind
            dis[ind] = la.norm(sub)
        print("dis : ", dis)
        ii = sorted(range(len(dis)), key=lambda k: dis[k])
        #    embed()
        #    print(ii[:top_num+1])
        return ii,dis

    def single_query(self,query_feature, test_feature, query_label, test_label, test_num):
        test_label_set = np.unique(test_label)  # np.unique() 删除重复的元素，并从大到小返回一个新的元祖或列表
        # single_num = len(test_label_set)
        test_label_dict = {}
        topp1 = 0
        topp5 = 0
        topp10 = 0
        for ind in range(len(test_label_set)):
            test_label_dict[test_label_set[ind]] = np.where(test_label == test_label_set[ind])  #
        # print(test_label_dict)
        for ind in range(test_num):
            query_int = np.random.choice(len(query_label))
            label = query_label[query_int]  # 随机取一个query id
            # print("query ID:",label)
            temp_int = np.random.choice(test_label_dict[label][0], 1)
            temp_gallery_ind = temp_int  # 当前正确的test id
            for ind2 in range(len(test_label_set)):  # 在每一个人里循环
                temp_label = test_label_set[ind2]
                if temp_label != label:
                    temp_int = np.random.choice(test_label_dict[temp_label][0], 1)
                    temp_gallery_ind = np.append(temp_gallery_ind, temp_int)
            print("temp_gallery_ind:",temp_gallery_ind)
            single_query_feature = query_feature[query_int]
            test_all_feature = test_feature[temp_gallery_ind]
            result_ind,dis = self.euclidSimilar2(single_query_feature, test_all_feature)
            print(result_ind)
            print(temp_gallery_ind[result_ind[0]])
            top1id = temp_gallery_ind[result_ind[0]]
            query_temp = result_ind.index(0)
            if query_temp < 1:
                topp1 = topp1 + 1
            if query_temp < 5:
                topp5 = topp5 + 1
            if query_temp < 10:
                topp10 = topp10 + 1
        topp1 = topp1 / test_num * 1.0
        topp5 = topp5 / test_num * 1.0
        topp10 = topp10 / test_num * 1.0
        print('single query')
        print('top1: ' + str(topp1) )
        print('top5: ' + str(topp5) )
        print('top10: ' + str(topp10))
        self.text_console.insert(2.0,'top1: ' + str(topp1) + '\n')
        self.text_console.insert(2.0, 'top5: ' + str(topp5) + '\n')
        self.text_console.insert(2.0, 'top10: ' + str(topp10) + '\n')
        self.text_console.insert(2.0, 'top1 ID: ' + str(top1id) + '\n')
        self.text_console.insert(2.0, 'top1 euclid distance: ' + str(dis[0]) + '\n')
        return topp1,top1id

    def confirm1(self):  # 确认键的操作
        self.current_person_id_cam1 = int(self.entry_id1.get())
        self.current_img_id_cam1 = 0
        person_id =tools.format_id(self.current_person_id_cam1)

        bm1 = Image.open(dataset_dir + person_id + "_00.jpg")
        tkimg1 = ImageTk.PhotoImage(bm1)

        self.label_person_image_cam1.config(image=tkimg1)
        self.label_filename1.config(text=person_id + "_00.jpg")
        self.label_person_image_cam1.image = tkimg1  # keep a reference


    def confirm_reid(self):  # 确认键的操作
        self.text_console.delete(1.0, END)
        self.text_console.insert(1.0,"行人ID:"+str(self.current_person_id_cam1)+"，开始重识别......\n")

        query_img, test_img, query_label, test_label = cuhk03_api.get_img(self.current_person_id_cam1)
        test_img = preprocess_input(test_img)
        query_img = preprocess_input(query_img)

        start = time.time()
        _, test_feature = self.class_triplet_model.predict(test_img)  # [_,2048]
        test_feature = normalize(test_feature)
        end = time.time()

        # output feature
        f = codecs.open("testfeature.txt", 'w', 'utf-8')
        for feature in test_feature:
            for dim in feature:
                f.write(str(dim))
                f.write(" ")
            f.write('\n')
        # output feature

        t1 = start-end
        self.text_console.insert(3.0,"提取gallery特征所需时间："+str(end-start)+"s\n")
        print(len(test_feature[0]))
        start = time.time()
        _, query_feature = self.class_triplet_model.predict(query_img)

        # output feature
        f = codecs.open("queryfeature_nonorm.txt", 'w', 'utf-8')
        for feature in query_feature:
            for dim in feature:
                f.write(str(dim))
                f.write(" ")
            f.write('\n')
        # output feature

        query_feature = normalize(query_feature)
        end = time.time()

        # output feature
        f = codecs.open("queryfeature.txt", 'w', 'utf-8')
        for feature in query_feature:
            for dim in feature:
                f.write(str(dim))
                f.write(" ")
            f.write('\n')
        # output feature

        self.text_console.insert(3.0, "提取query特征所需时间：" + str(end-start) + "s\n")
        start = time.time()
        top1,top1id = self.single_query(query_feature, test_feature, query_label, test_label, test_num=1)
        end = time.time()
        self.text_console.insert(3.0, "计算欧几里得距离所需时间：" + str(end-start) + "s\n")

        person_id = tools.format_id(top1id)
        self.current_person_id_cam2 = top1id
        self.current_img_id_cam2 = 5
        bm2 = Image.open(dataset_dir + person_id + "_0"+str(self.current_img_id_cam2)+".jpg")
        tkimg2 = ImageTk.PhotoImage(bm2)



        # 补一下左图，不知道为什么显示不出来
        self.current_person_id_cam1 = int(self.entry_id1.get())
        self.current_img_id_cam1 = 0
        person_id = tools.format_id(self.current_person_id_cam1)
        bm1 = Image.open(dataset_dir + person_id + "_00.jpg")
        tkimg1 = ImageTk.PhotoImage(bm1)

        self.label_person_image_cam2.config(image=tkimg2)
        self.label_filename2.config(text=person_id + "_05.jpg")
        self.label_person_image_cam1.image = tkimg2  # keep a reference

        self.label_person_image_cam1.config(image=tkimg1)
        self.label_filename1.config(text=person_id + "_00.jpg")
        self.label_person_image_cam1.image = tkimg1  # keep a reference
        #


    def next1(self):  # 确认键的操作
        person_id = tools.format_id(self.current_person_id_cam1)

        self.current_img_id_cam1 += 1
        if(self.current_img_id_cam1 == 5):
            self.current_img_id_cam1 = 0
        bm1 = Image.open(dataset_dir + person_id + "_0"+str(self.current_img_id_cam1)+".jpg")
        tkimg1 = ImageTk.PhotoImage(bm1)

        self.label_person_image_cam1.config(image=tkimg1)
        self.label_person_image_cam1.image = tkimg1  # keep a reference
        self.label_filename1.config(text=person_id + "_0" + str(self.current_img_id_cam1) + ".jpg")



    def next2(self):  # 确认键的操作
        person_id2 = tools.format_id(self.current_person_id_cam2) # 把2变成0002这样

        self.current_img_id_cam2 += 1
        if (self.current_img_id_cam2 == 10):
            self.current_img_id_cam2 = 5
        bm2 = Image.open(dataset_dir + person_id2 + "_0" + str(self.current_img_id_cam2) + ".jpg")
        tkimg2 = ImageTk.PhotoImage(bm2)

        self.label_person_image_cam2.config(image=tkimg2)
        self.label_person_image_cam2.image = tkimg2  # keep a reference
        self.label_filename2.config(text=person_id2 + "_0" + str(self.current_img_id_cam2) + ".jpg")

app = App()