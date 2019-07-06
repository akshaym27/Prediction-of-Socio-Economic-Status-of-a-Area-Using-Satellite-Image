import cv2
import numpy as np
from matplotlib import pyplot as plt
import operator

img = cv2.imread('sat2.png', 1)
cv2.imshow('Imagem:',img)

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
    print histr
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

plt.title("Red: "+str(qtdRed)+"%; Green: "+str(qtdGreen)+"%; Blue: "+str(qtdBlue)+"%")
plt.show()
