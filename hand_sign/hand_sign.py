import math
import time
import cv2
import mediapipe as mp
import hand_tracking as ht
import numpy as np
import tensorflow
from cvzone import ClassificationModule as cm
print("hi")

video=cv2.VideoCapture(0)
detect=ht.handDetector(handNo=1)
# folder="data/U"
count=0
label=['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
classify=cm.Classifier('Model/keras_model.h5','Model/labels.txt')
while True:
    s,image=video.read()
    hand=detect.handDetection(image)
    position,box=detect.findPosition(hand,draw=False)
    if(len(position)!=0):

        a=position[4][1]
        b=position[4][2]
        d = position[6][1]
        e = position[6][2]
        # c=position[10][1]
        # f=position[10][2]
        # g=position[14][1]
        # h=position[14][2]
        # i=position[18][1]
        # j=position[18][2]
        x = box[0][0]
        y = box[0][1]
        w=box[0][2]
        h=box[0][3]
        ratio=h/w
        white=np.ones((300,300,3),np.uint8)*255
        if(ratio>1):
            cut=image[y-20:y+h+20,x-20:x+h+5]
            cut = cv2.resize(cut, (200, 300))
            shape=cut.shape
            white[:shape[0],0:shape[1]]=cut
            prediction,index=classify.getPrediction(white,draw=False)
            # if(label[index]=='A'or label[index]=='S'or label[index]=='M'or label[index]=='N' or label[index]=='T'):
            #     d=math.sqrt((d-a)**2+(e-b)**2)
            #     if(d>=20 and d<=30):
            #         index=0


        else:
            cut = image[y - 20:y + h + 20, x - 20:x + h + 80]
            cut=cv2.resize(cut,(200,300))
            shape = cut.shape
            white[:shape[0], 0:shape[1]] = cut
            prediction, index = classify.getPrediction(white, draw=False)
        key=cv2.waitKey(1)
        if (key == ord('s')):
            count+=1
            print(count)
            cv2.imwrite(f'{folder}/image_{time.time()}.jpg',white)
        cv2.putText(hand,label[index],(x,y),cv2.FONT_HERSHEY_SIMPLEX,2,(255,0,0))
        cv2.imshow('white',white)


    cv2.imshow("Hand_Sign",image)
    if(cv2.waitKey(1)& 0xff==ord('z')):
       break