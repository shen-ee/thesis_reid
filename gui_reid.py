# coding:utf-8
from tkinter import *
from PIL import Image, ImageTk
import tools
import tensorflow as tf
import cuhk03_dataset
import cv2
import numpy as np

'''
这个界面是为了进行真正地重识别工作
首先你要选中一个id
接下来你会在左边地展示框中看到五张他在cam1中地图片
'''

dataset_dir = "/Users/shenyi/Desktop/thesisCode/data/cuhk03_release/labeled/train/"
model_dir = "/Users/shenyi/Documents/GitHub/person-reid/logs/"
dataset_name = "CUHK03"



FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer('batch_size', '150', 'batch size for training')
tf.flags.DEFINE_integer('max_steps', '210000', 'max steps for training')
tf.flags.DEFINE_string('logs_dir', model_dir, 'path to logs directory')
tf.flags.DEFINE_string('data_dir', 'data/', 'path to dataset')
tf.flags.DEFINE_float('learning_rate', '0.01', '')
tf.flags.DEFINE_string('mode', 'train', 'Mode train, val, test')
tf.flags.DEFINE_string('image1', '', 'First image path to compare')
tf.flags.DEFINE_string('image2', '', 'Second image path to compare')

IMAGE_WIDTH = 60
IMAGE_HEIGHT = 160



def preprocess(images, is_train):
    def train():
        split = tf.split(images, [1, 1])
        shape = [1 for _ in range(split[0].get_shape()[1])]
        for i in range(len(split)):
            split[i] = tf.reshape(split[i], [FLAGS.batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, 3])
            split[i] = tf.image.resize_images(split[i], [IMAGE_HEIGHT + 8, IMAGE_WIDTH + 3])
            split[i] = tf.split(split[i], shape)
            for j in range(len(split[i])):
                split[i][j] = tf.reshape(split[i][j], [IMAGE_HEIGHT + 8, IMAGE_WIDTH + 3, 3])
                split[i][j] = tf.random_crop(split[i][j], [IMAGE_HEIGHT, IMAGE_WIDTH, 3])
                split[i][j] = tf.image.random_flip_left_right(split[i][j])
                split[i][j] = tf.image.random_brightness(split[i][j], max_delta=32. / 255.)
                split[i][j] = tf.image.random_saturation(split[i][j], lower=0.5, upper=1.5)
                split[i][j] = tf.image.random_hue(split[i][j], max_delta=0.2)
                split[i][j] = tf.image.random_contrast(split[i][j], lower=0.5, upper=1.5)
                split[i][j] = tf.image.per_image_standardization(split[i][j])
        return [tf.reshape(tf.concat(split[0], axis=0), [FLAGS.batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, 3]),
            tf.reshape(tf.concat(split[1], axis=0), [FLAGS.batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, 3])]
    def val():
        split = tf.split(images, [1, 1])
        shape = [1 for _ in range(split[0].get_shape()[1])]
        for i in range(len(split)):
            split[i] = tf.reshape(split[i], [FLAGS.batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, 3])
            split[i] = tf.image.resize_images(split[i], [IMAGE_HEIGHT, IMAGE_WIDTH])
            split[i] = tf.split(split[i], shape)
            for j in range(len(split[i])):
                split[i][j] = tf.reshape(split[i][j], [IMAGE_HEIGHT, IMAGE_WIDTH, 3])
                split[i][j] = tf.image.per_image_standardization(split[i][j])
        return [tf.reshape(tf.concat(split[0], axis=0), [FLAGS.batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, 3]),
            tf.reshape(tf.concat(split[1], axis=0), [FLAGS.batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, 3])]
    return tf.cond(is_train, train, val)  # if is_train == true , train()

def network(images1, images2, weight_decay):
    with tf.variable_scope('network'):
        # Tied Convolution
        conv1_1 = tf.layers.conv2d(images1, 20, [5, 5], activation=tf.nn.relu,
            kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay), name='conv1_1')
        pool1_1 = tf.layers.max_pooling2d(conv1_1, [2, 2], [2, 2], name='pool1_1')
        conv1_2 = tf.layers.conv2d(pool1_1, 25, [5, 5], activation=tf.nn.relu,
            kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay), name='conv1_2')
        pool1_2 = tf.layers.max_pooling2d(conv1_2, [2, 2], [2, 2], name='pool1_2')
        conv2_1 = tf.layers.conv2d(images2, 20, [5, 5], activation=tf.nn.relu,
            kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay), name='conv2_1')
        pool2_1 = tf.layers.max_pooling2d(conv2_1, [2, 2], [2, 2], name='pool2_1')
        conv2_2 = tf.layers.conv2d(pool2_1, 25, [5, 5], activation=tf.nn.relu,
            kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay), name='conv2_2')
        pool2_2 = tf.layers.max_pooling2d(conv2_2, [2, 2], [2, 2], name='pool2_2')

        # Cross-Input Neighborhood Differences
        trans = tf.transpose(pool1_2, [0, 3, 1, 2])
        shape = trans.get_shape().as_list()
        m1s = tf.ones([shape[0], shape[1], shape[2], shape[3], 5, 5])
        reshape = tf.reshape(trans, [shape[0], shape[1], shape[2], shape[3], 1, 1])
        f = tf.multiply(reshape, m1s)

        trans = tf.transpose(pool2_2, [0, 3, 1, 2])
        reshape = tf.reshape(trans, [1, shape[0], shape[1], shape[2], shape[3]])
        g = []
        pad = tf.pad(reshape, [[0, 0], [0, 0], [0, 0], [2, 2], [2, 2]])
        for i in range(shape[2]):
            for j in range(shape[3]):
                g.append(pad[:,:,:,i:i+5,j:j+5])

        concat = tf.concat(g, axis=0)
        reshape = tf.reshape(concat, [shape[2], shape[3], shape[0], shape[1], 5, 5])
        g = tf.transpose(reshape, [2, 3, 0, 1, 4, 5])
        reshape1 = tf.reshape(tf.subtract(f, g), [shape[0], shape[1], shape[2] * 5, shape[3] * 5])
        reshape2 = tf.reshape(tf.subtract(g, f), [shape[0], shape[1], shape[2] * 5, shape[3] * 5])
        k1 = tf.nn.relu(tf.transpose(reshape1, [0, 2, 3, 1]), name='k1')
        k2 = tf.nn.relu(tf.transpose(reshape2, [0, 2, 3, 1]), name='k2')

        # Patch Summary Features
        l1 = tf.layers.conv2d(k1, 25, [5, 5], (5, 5), activation=tf.nn.relu,
            kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay), name='l1')
        l2 = tf.layers.conv2d(k2, 25, [5, 5], (5, 5), activation=tf.nn.relu,
            kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay), name='l2')

        # Across-Patch Features
        m1 = tf.layers.conv2d(l1, 25, [3, 3], activation=tf.nn.relu,
            kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay), name='m1')
        pool_m1 = tf.layers.max_pooling2d(m1, [2, 2], [2, 2], padding='same', name='pool_m1')
        m2 = tf.layers.conv2d(l2, 25, [3, 3], activation=tf.nn.relu,
            kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay), name='m2')
        pool_m2 = tf.layers.max_pooling2d(m2, [2, 2], [2, 2], padding='same', name='pool_m2')

        # Higher-Order Relationships
        concat = tf.concat([pool_m1, pool_m2], axis=3)
        reshape = tf.reshape(concat, [FLAGS.batch_size, -1])
        fc1 = tf.layers.dense(reshape, 500, tf.nn.relu, name='fc1')
        fc2 = tf.layers.dense(fc1, 2, name='fc2')

        return fc2

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


        # tensorflow_start


        FLAGS.batch_size = 1

        learning_rate = tf.placeholder(tf.float32, name='learning_rate')
        self.images = tf.placeholder(tf.float32, [2, FLAGS.batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, 3], name='images')
        labels = tf.placeholder(tf.float32, [FLAGS.batch_size, 2], name='labels')
        self.is_train = tf.placeholder(tf.bool, name='is_train')
        global_step = tf.Variable(0, name='global_step', trainable=False)
        weight_decay = 0.0005
        tarin_num_id = 0
        val_num_id = 0

        #if FLAGS.mode == 'train':
        #   tarin_num_id = cuhk03_dataset.get_num_id(FLAGS.data_dir, 'train')
        #elif FLAGS.mode == 'val':
        #    val_num_id = cuhk03_dataset.get_num_id(FLAGS.data_dir, 'val')
        images1, images2 = preprocess(self.images, self.is_train)

        print('Build network')
        self.text_console.insert(1.0,"Build network done.")
        logits = network(images1, images2, weight_decay)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits))
        self.inference = tf.nn.softmax(logits)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()

        ckpt = tf.train.get_checkpoint_state(FLAGS.logs_dir)
        print(ckpt.model_checkpoint_path)
        if ckpt and ckpt.model_checkpoint_path:
            print('Restore model')
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)

        # tensorflow_end

        self.root.mainloop()

    def test_pair(self, image1_dir, image2_dir):
        image1 = cv2.imread(image1_dir)
        image1 = cv2.resize(image1, (IMAGE_WIDTH, IMAGE_HEIGHT))
        image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
        image1 = np.reshape(image1, (1, IMAGE_HEIGHT, IMAGE_WIDTH, 3)).astype(float)
        image2 = cv2.imread(image2_dir)
        image2 = cv2.resize(image2, (IMAGE_WIDTH, IMAGE_HEIGHT))
        image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
        image2 = np.reshape(image2, (1, IMAGE_HEIGHT, IMAGE_WIDTH, 3)).astype(float)
        test_images = np.array([image1, image2])
        feed_dict = {self.images: test_images, self.is_train: False}
        prediction = self.sess.run(self.inference, feed_dict=feed_dict)
        print(bool(not np.argmax(prediction[0])))
        if (bool(not np.argmax(prediction[0])) == True):
            self.label_match.config(text="匹配结果：吻合")
        else:
            self.label_match.config(text="匹配结果：不吻合")
        self.label_confidence.config(text="置信度："+str(prediction[0][0])+","+str(prediction[0][1]))

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
        self.text_console.delete(0.0, END)
        self.text_console.insert(1.0,"行人ID:"+str(self.current_person_id_cam1)+"，开始重识别......")


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
        person_id1 = tools.format_id(self.current_person_id_cam1) # 把2变成0002这样



app = App()