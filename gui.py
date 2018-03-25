# coding:utf-8
from Tkinter import *
from PIL import Image, ImageTk

dataset = "/Users/shenyi/Desktop/thesisCode/data/cuhk03_release/labeled/train/"



class App:
    def __init__(self):
        self.current_img_id_cam1 = 0
        self.current_img_id_cam2 = 5

        self.root = Tk()
        self.root.title('Person Re-id')
        frame0 = Frame(self.root)
        self.frame1 = Frame(self.root)
        self.frame2 = Frame(self.root)

        label_dataset = Label(frame0,text="数据集路径:"+dataset)
        w = Label(frame0,text="选择id号:")
        e = Entry(frame0)
        button = Button(frame0,text="确认",command = self.confirm)

        # frame1
        bm1 = Image.open(dataset+"0000_00.jpg")
        tkimg1 = ImageTk.PhotoImage(bm1)
        self.label_person_image_cam1 = Label(self.frame1,image = tkimg1)
        self.button1 = Button(self.frame1, text="下一张", command=self.confirm1)

        # frame2
        bm2 = Image.open(dataset + "0000_05.jpg")
        tkimg2 = ImageTk.PhotoImage(bm2)
        self.label_person_image_cam2 = Label(self.frame2,image = tkimg2)
        self.button2 = Button(self.frame2, text="下一张", command=self.confirm2)

        # frame1.pack()
        # frame2.pack()
        frame0.grid(row = 0,columnspan=2)
        self.frame1.grid(row = 1,column = 0)
        self.frame2.grid(row=1, column=1)

        label_dataset.grid(row=0,sticky=W,columnspan=3)
        w.grid(row = 1,sticky = W)
        e.grid(row = 1,column = 1)
        button.grid(row = 1,column=2)

        self.label_person_image_cam1.pack()
        self.button1.pack()

        self.label_person_image_cam2.pack()
        self.button2.pack()

        self.root.mainloop()

    def confirm(self):  # 确认键的操作
        final_string = ''
        final_string = dataset


    def confirm1(self):  # 确认键的操作
        self.current_img_id_cam1 += 1
        if(self.current_img_id_cam1 == 5):
            self.current_img_id_cam1 = 0
        print (self.current_img_id_cam1)
        bm1 = Image.open(dataset + "0000_0"+str(self.current_img_id_cam1)+".jpg")
        tkimg1 = ImageTk.PhotoImage(bm1)
        self.label_person_image_cam1.config(image=tkimg1)
        self.label_person_image_cam1.image = tkimg1  # keep a reference


    def confirm2(self):  # 确认键的操作
        self.current_img_id_cam2 += 1
        if (self.current_img_id_cam2 == 10):
            self.current_img_id_cam2 = 5
        print (self.current_img_id_cam1)
        bm2 = Image.open(dataset + "0000_0" + str(self.current_img_id_cam2) + ".jpg")
        tkimg2 = ImageTk.PhotoImage(bm2)
        self.label_person_image_cam2.config(image=tkimg2)
        self.label_person_image_cam2.image = tkimg2  # keep a reference


app = App()