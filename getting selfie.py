# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 23:45:42 2019

@author: lenovo
"""
import cv2
import numpy as np
cap=cv2.VideoCapture(0)
face_cascade=cv2.CascadeClassifier("C:/Users/lenovo/haracascade_frontalface_alt.xml")
skip=0
face_data=[]
file_name=input("Enter the File name")
while True:
    ret,frame=cap.read()
    if ret==False:
        continue
    gray_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale(frame,1.3,5)
    faces=sorted(faces,key=lambda f:f[2]*f[3],reverse=True)
    for face in faces:
        (x,y,w,h)=face
        cv2.rectangle(frame,(x,y),(x+w,y+h),(251,0,0),2)
    offset=10
    face_section=frame[y-offset:y+h+offset,x-offset:x+w+offset]
    face_section=cv2.resize(face_section,(100,100))
    skip+=1
    if skip%10==0:
        face_data.append(face_section)
        print(len(face_data))
    cv2.imshow("frame",frame)
    cv2.imshow("face section",face_section)
    key_pressed=cv2.waitKey(1) & 0xFF
    if (key_pressed==ord('q')):
        break
faces_data=np.asarray(face_data)
faces_data=faces_data.reshape((faces_data.shape[0],-1))
print(faces_data.shape)
#print(face_data)
np.save("E:/machine learning course/images/"+file_name+'.npy',face_data)
print("Data saved sucessfully")
cap.release()
cv2.destroyAllWindows()