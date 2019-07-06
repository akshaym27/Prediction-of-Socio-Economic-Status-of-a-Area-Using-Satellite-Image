from PyQt4 import QtGui
from PyQt4 import QtCore
from PyQt4.QtGui import QFileDialog
import sys
from PyQt4.QtCore import *
from PyQt4.QtGui import *
import time 
import login
import home
import error_log
import err_img
import MySQLdb
import numpy as np
import cv2
import os
from PIL import Image

import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import pickle

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D
from tensorflow.python.keras.callbacks import TensorBoard

import tensorflow as tf
from tensorflow.python.keras.models import load_model
import imageio

import operator



fname=""

class ExampleApp(QtGui.QMainWindow, login.Ui_UserLogin):
    def __init__(self):
        super(self.__class__, self).__init__()
        self.setupUi(self) 
        self.pushButton.clicked.connect(self.log)
        self.pushButton_2.clicked.connect(self.can)
        
    def log(self):
        i=0
        db = MySQLdb.connect("localhost","root","root","socio")
        cursor = db.cursor()
        a=self.lineEdit.text()
        b=self.lineEdit_2.text()
        sql = "SELECT * FROM user WHERE username='%s' and pass='%s'" % (a,b)
        try:
            cursor.execute(sql)
            results = cursor.fetchall()
            for row in results:
                i=i+1
        except Exception as e:
           print e
        if i>=0:
            print "login success"
            self.hide()
            self.home=home()
            self.home.show()
            
        else:
            print "login failed"
            self.errlog=errlog()
            self.errlog.show()
                    
        db.close()
        
    def can(self):
        sys.exit()
        
   

        

class home(QtGui.QMainWindow, home.Ui_Home):
    def __init__(self):
        super(self.__class__, self).__init__()
        self.setupUi(self)
        self.pushButton.clicked.connect(self.selimg)
        self.pushButton_2.clicked.connect(self.seldir)
        self.pushButton_3.clicked.connect(self.cnnperform)
        self.pushButton_5.clicked.connect(self.ex)
        self.pushButton_6.clicked.connect(self.preproc)
        self.pushButton_7.clicked.connect(self.pred)

    def selimg(self):
        global fname
        self.QFileDialog = QtGui.QFileDialog(self)
        #self.QFileDialog.show()
        fname = QFileDialog.getOpenFileName(self, 'Open file','c:\\',"Image files (*.jpg *.png)")
        print fname
        label = QLabel(self.label_5)
        pixmap = QPixmap(fname)
        label.setPixmap(pixmap)
        label.resize(pixmap.width(),pixmap.height())
        label.show()

    
    def seldir(self):
        self.QFileDialog = QtGui.QFileDialog(self)
        folder = QFileDialog.getExistingDirectory(self, "Select Folder")
        print folder
        

    def preproc(self):
        global fname
        if fname=="":
            self.errimg=errimg()
            self.errimg.show()
        else:
            filename = fname
            print "file for processing",filename
            image =cv2.imread(str(filename))
            #print type(image)
            cv2.imshow("Original Image", image)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            cv2.imshow("1 - Grayscale Conversion", gray)
            gray = cv2.bilateralFilter(gray, 11, 17, 17)
            cv2.imshow("2 - Bilateral Filter", gray)
            edged = cv2.Canny(gray, 270, 400)
            cv2.imshow("4 - Canny Edges", edged)

    def cnnperform(self):
            DATADIR = "Datasets"
        
            CATEGORIES = ["agriculture", "building", "road","water"]
            
            training_data = []  
            IMG_SIZE = 250
            
            def create_training_data():
                for category in CATEGORIES:
                    path = os.path.join(DATADIR,category)
                    class_num = CATEGORIES.index(category)  # get the classification.
            
                    for img in tqdm(os.listdir(path)):  # iterate over each image per pothole and no 
                        try:
                            img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE)  # convert to array
                            new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # resize to normalize data size
                            training_data.append([new_array, class_num])  # add this to our training_data
                        except Exception as e:
                            print("general exception", e, os.path.join(path,img))    
            create_training_data()
            
            random.shuffle(training_data)
            
            X = []
            y = []
            
            for features,label in training_data:
                X.append(features)
                y.append(label)
            
            X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
            
            pickle_out = open("X.pickle","wb")
            pickle.dump(X, pickle_out)
            pickle_out.close()
            
            pickle_out = open("y.pickle","wb")
            pickle.dump(y, pickle_out)
            pickle_out.close()

            pickle_in = open("X.pickle","rb")
            X = pickle.load(pickle_in)

            pickle_in = open("y.pickle","rb")
            y = pickle.load(pickle_in)

            X = X/255.0

            dense_layers = [0]
            layer_sizes = [64]
            conv_layers = [3]

            for dense_layer in dense_layers:
                for layer_size in layer_sizes:
                    for conv_layer in conv_layers:
                        NAME = "{}-conv-{}-nodes-{}-dense-{}".format(conv_layer, layer_size, dense_layer, int(time.time()))
                        print(NAME)

                        model = Sequential()

                        model.add(Conv2D(layer_size, (3, 3), input_shape=X.shape[1:]))
                        model.add(Activation('relu'))
                        model.add(MaxPooling2D(pool_size=(2, 2)))

                        for l in range(conv_layer-1):
                            model.add(Conv2D(layer_size, (3, 3)))
                            model.add(Activation('relu'))#rectifier linear unit
                            model.add(MaxPooling2D(pool_size=(2, 2)))

                        model.add(Flatten())

                        for _ in range(dense_layer):
                            model.add(Dense(layer_size))
                            model.add(Activation('relu'))

                        model.add(Dense(1))
                        model.add(Activation('sigmoid'))

                        tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))

                        model.compile(loss='binary_crossentropy',
                                      optimizer='adam',
                                      metrics=['accuracy'],
                                      )

                        model.fit(X, y,
                                  batch_size=32,
                                  epochs=4,
                                  validation_split=0.3,
                                  callbacks=[tensorboard])


            model.summary()
            model.save('CNN.model')
        

    def pred(self):
        global fname
        CATEGORIES = ["agriculture", "building", "road","water"]


        def prepare(filepath):
            IMG_SIZE = 250  
            img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
            return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)


        model = load_model("CNN.model")

        prediction = model.predict([prepare(str(fname))])

        print "...",prediction
        predictn=CATEGORIES[int(prediction[0][0])]
        
        img = cv2.imread(str(fname), 1)
        color = ('b','g','r')
        qtdBlue = 0
        qtdGreen = 0
        qtdRed = 0
        totalPixels = 0

        for channel,col in enumerate(color):
            histr = cv2.calcHist([img],[channel],None,[256],[1,256])
            plt.plot(histr,color = col)
            plt.xlim([0,256])
            totalPixels+=sum(histr)
            #print histr
            if channel==0:
                qtdBlue = sum(histr)
            elif channel==1:
                qtdGreen = sum(histr)
            elif channel==2:
                qtdRed = sum(histr)

        qtdBlue = (qtdBlue/totalPixels)*100
        qtdGreen = (qtdGreen/totalPixels)*100
        qtdRed = (qtdRed/totalPixels)*100

        qtdBlue = filter(operator.isNumberType, qtdBlue)
        qtdGreen = filter(operator.isNumberType, qtdGreen)
        qtdRed = filter(operator.isNumberType, qtdRed)
        
        img = cv2.imread(str(fname))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 11, 17, 17)
        edges=cv2.Canny(gray,100,200)
        building=0
        elect=0
        road=0
        agri=0
        water=0
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        i=0
        cnts,heir= cv2.findContours(closed.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2:]
        for c in cnts: 	
            peri = cv2.arcLength(c, True) 	
            approx = cv2.approxPolyDP(c, 0.02 * peri, True) 
            cv2.drawContours(img, [approx], -1, (0, 255, 0), 2)	
            x,y,w,h =cv2.boundingRect(c)
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)	
            i=i+1;
            newImage=img[y:y+h,x:x+w]
            if len(c)<10:
                building+=1
                elect+=1
            if len(c)>60 and len(c)<100:
                water+=1
            if len(c)>100 and len(c)<500:
                agri+=1
            if len(c)>500:
                road+=1
                water+=1
            
        print "building=",building
        print "agri=",agri
        print "water=",water
        print "road=",road

        print "building resources %f "%(float(building)*100/i)
        print "elect resources %f "%(float(elect)*100/i)
        print "agri resources %f "%(float(agri)*100/i)
        print "water resources %f "%(float(water)*100/i)
        print "road resources %f "%(float(road)*100/i)
        br=float(building)*100/i
        er=float(elect)*100/i
        ar=float(agri)*100/i
        wr=float(water)*100/i
        
        self.lineEdit.setText(str(ar))
        self.lineEdit_2.setText(str(er))
        self.lineEdit_4.setText(str(wr))
        self.lineEdit_3.setText(str(br))
        if br>10 and ar>10 and wr>10:
            self.lineEdit_6.setText("Socio economic condition of area is very good")
        elif br>10 and ar<10 and wr<10:
            self.lineEdit_6.setText("Socio economic condition of area is fairly good")
        elif br<10 and wr<50 and ar<50:
            self.lineEdit_6.setText("Socio economic condition of area is fairly bad")
        else:
            self.lineEdit_6.setText("Socio economic condition of area is bad")
       
               
        plt.title("Red: "+str(qtdRed)+"%; Green: "+str(qtdGreen)+"%; Blue: "+str(qtdBlue)+"%")
        #plt.show()
        #cv2.imshow('dst_rt', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows
            
    def ex(self):
        sys.exit()
        

class errlog(QtGui.QMainWindow, error_log.Ui_Error):
    def __init__(self):
        super(self.__class__, self).__init__()
        self.setupUi(self)
        self.pushButton.clicked.connect(self.ok1)
    def ok1(self):
        self.hide()

class errimg(QtGui.QMainWindow, err_img.Ui_Error):
    def __init__(self):
        super(self.__class__, self).__init__()
        self.setupUi(self)
        self.pushButton.clicked.connect(self.ok1)
    def ok1(self):
        self.hide()



def main():
    app = QtGui.QApplication(sys.argv)  
    form = ExampleApp()                 
    form.show()                         
    app.exec_()                         


if __name__ == '__main__':              
    main()                             
