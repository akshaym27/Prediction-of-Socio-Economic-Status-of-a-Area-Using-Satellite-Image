import cv2
import numpy as np
from matplotlib import pyplot as plt
import operator

img = cv2.imread('sat2.png', 1)
#cv2.imshow('Imagem:',img)

categories = ('building','agriculture','water')
qtdWater = 0
qtdAgree = 0
qtdBuild = 0
totalPixels = 0

for channel,col in enumerate(categories):
    histr = cv2.calcHist([img],[channel],None,[256],[1,256])
    plt.plot(histr,color = col)
    plt.xlim([0,256])
    totalPixels+=sum(histr)
    print histr
    if channel==0:
        qtdWater = sum(histr)
    elif channel==1:
        qtdAgree = sum(histr)
    elif channel==2:
        qtdBuild = sum(histr)

qtdWater = (qtdWater/totalPixels)*100
qtdAgree = (qtdAgree/totalPixels)*100
qtdBuild = (qtdBuild/totalPixels)*100

qtdWater = filter(operator.isNumberType, qtdWater)
qtdAgree = filter(operator.isNumberType, qtdAgree)
qtdBuild = filter(operator.isNumberType, qtdBuild)

plt.title("Building: "+str(qtdBuild)+"%; Agree: "+str(qtdAgree)+"%; Water: "+str(qtdWater)+"%")
plt.show()
