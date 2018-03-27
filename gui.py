# coding:utf-8
from tkinter import *
from PIL import Image, ImageTk
import tools
import tensorflow as tf

dataset_dir = "/Users/shenyi/Desktop/thesisCode/data/cuhk03_release/labeled/train/"
dataset_name = "CUHK03"
class App:
    def __init__(self):
        self.current_person_id = -1
        self.current_img_id_cam1 = 0
        self.current_img_id_cam2 = 5

        self.root = Tk()
        self.root.title('Person Re-id')
        self.frame0 = Frame(self.root)
        self.frame1 = Frame(self.root)
        self.frame2 = Frame(self.root)
        self.frame_match = Frame(self.root)  # 显示匹配结果

        label_dataset_name = Label(self.frame0,text="数据集名称："+dataset_name)
        label_dataset_dir = Label(self.frame0,text="数据集路径："+dataset_dir)
        w = Label(self.frame0,text="选择id号:")
        self.e = Entry(self.frame0)
        button = Button(self.frame0,text="确认",command = self.confirm)

        # frame1
        self.label_person_image_cam1 = Label(self.frame1)
        self.label_filename1 = Label(self.frame1)
        self.button1 = Button(self.frame1, text="下一张", command=self.confirm1)

        # frame2
        self.label_person_image_cam2 = Label(self.frame2)
        self.label_filename2 = Label(self.frame2)
        self.button2 = Button(self.frame2, text="下一张", command=self.confirm2)

        # frame_match
        label_match = Label(self.frame_match,text="匹配结果：+++")

        self.frame0.grid(row = 0,columnspan=2)
        self.frame1.grid(row = 1,column = 0)
        self.frame2.grid(row=1, column=1)
        self.frame_match.grid(row=2,columnspan=2)

        label_dataset_name.grid(row=0,sticky=W,columnspan=3)
        label_dataset_dir.grid(row=1,sticky=W,columnspan=3)
        w.grid(row = 2,sticky = W)
        self.e.grid(row = 2,column = 1)
        button.grid(row = 2,column=2)

        self.label_person_image_cam1.pack()
        self.label_filename1.pack()
        self.button1.pack()

        self.label_person_image_cam2.pack()
        self.label_filename2.pack()
        self.button2.pack()

        label_match.grid(row=0,sticky=W)

        self.root.mainloop()

    def confirm(self):  # 确认键的操作
        self.current_person_id = int(self.e.get())
        person_id =tools.format_id(self.current_person_id)

        bm1 = Image.open(dataset_dir + person_id + "_00.jpg")
        tkimg1 = ImageTk.PhotoImage(bm1)

        bm2 = Image.open(dataset_dir + person_id + "_05.jpg")
        tkimg2 = ImageTk.PhotoImage(bm2)

        self.label_person_image_cam1.config(image=tkimg1)
        self.label_filename1.config(text=person_id + "_00.jpg")
        self.label_person_image_cam1.image = tkimg1  # keep a reference

        self.label_person_image_cam2.config(image=tkimg2)
        self.label_filename2.config(text=person_id + "_05.jpg")
        self.label_person_image_cam2.image = tkimg2  # keep a reference



    def confirm1(self):  # 确认键的操作
        person_id = tools.format_id(self.current_person_id)

        self.current_img_id_cam1 += 1
        if(self.current_img_id_cam1 == 5):
            self.current_img_id_cam1 = 0
        bm1 = Image.open(dataset_dir + person_id + "_0"+str(self.current_img_id_cam1)+".jpg")
        tkimg1 = ImageTk.PhotoImage(bm1)

        self.label_person_image_cam1.config(image=tkimg1)
        self.label_person_image_cam1.image = tkimg1  # keep a reference
        self.label_filename1.config(text=person_id + "_0" + str(self.current_img_id_cam1) + ".jpg")


    def confirm2(self):  # 确认键的操作
        person_id = tools.format_id(self.current_person_id)

        self.current_img_id_cam2 += 1
        if (self.current_img_id_cam2 == 10):
            self.current_img_id_cam2 = 5
        bm2 = Image.open(dataset_dir + person_id + "_0" + str(self.current_img_id_cam2) + ".jpg")
        tkimg2 = ImageTk.PhotoImage(bm2)

        self.label_person_image_cam2.config(image=tkimg2)
        self.label_person_image_cam2.image = tkimg2  # keep a reference
        self.label_filename2.config(text=person_id + "_0" + str(self.current_img_id_cam2) + ".jpg")

app = App()