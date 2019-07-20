# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 01:21:47 2019

@author: lenovo
"""

import cv2
import numpy as np
import os
def distance(x1,x2):
    return np.sqrt(sum((x1-x2)**2))
def knn(train,querypoint,k=5):
    vals=[]
    for i in range (train.shape[0]):
        ix=train[i,:-1]
        iy=train[i,-1]
        d=distance(querypoint,ix)
        vals.append((d,iy))
    vals=sorted(vals)
    #labels=np.array(vals)[:,-1]
    vals=vals[:k]
    vals=np.array(vals)
    new_vals=np.unique(vals[:,1],return_counts=True)
    index=new_vals[1].argmax()
    pred=new_vals[0][index]
    return pred    
#init camera
cap=cv2.VideoCapture(0)
#Face detection
face_cascading=cv2.CascadeClassifier("C:/Users/lenovo/haracascade_frontalface_alt.xml")
skip=0
dataset_path='E:/machine learning course/images/'
face_data=[]
labels=[]
class_id=0
names={}
#Data preperation
for fx in os.listdir(dataset_path):
    if fx.endswith('.npy'):
        print('Loaded'+fx)
        names[class_id]=fx[:-4]
        data_item=np.load(dataset_path+fx)
        face_data.append(data_item)
        #create labels
        target=class_id*(np.ones((data_item.shape[0],)))
        class_id+=1
        labels.append(target)
face_dataset=np.concatenate(face_data,axis=0)
face_dataset=face_dataset.reshape((-1,30000))
face_labels=np.concatenate(labels,axis=0)
print(face_dataset.shape)
print(face_labels.shape)
face_labels=face_labels.reshape((-1,1))
trainset=np.concatenate((face_dataset,face_labels),axis=1)
while True:
    ret,frame=cap.read()
    if(ret==False):
        continue
    faces=face_cascading.detectMultiScale(frame,1.3,5)
    for face in faces:
        x,y,w,h=face
        
        #offset
        offset=10
        face_section=frame[y-offset:y+h+offset,x-offset:x+w+offset]
        face_section=cv2.resize(face_section,(100,100))
        pred=knn(trainset,face_section.flatten())
        pred_name=names[int(pred)]
        cv2.putText(frame,pred_name,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,1,(251,0,0),2,cv2.LINE_AA)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,251,251),2)
        cv2.imshow("Frames",frame)
    key=cv2.waitKey(1) & 0xFF
    if(key==ord('q')):
        break
cap.release()
cv2.destroyAllWindows()