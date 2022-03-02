from time import sleep
import cv2
import numpy as np
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('OpenCV-Python-Series-master\OpenCV-Python-Series-master\src\cascades\data\haarcascade_frontalface_default.xml')
run = True
i  = 0
while run:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,1.1,4)
    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0),2)
        roi_gray = gray[y:y+h, x:x+h]
        roi_color = frame[y:y+h, x:x+h]
        img_item_c = './images/anup/'+str(i) +'.png'
        cv2.imwrite(img_item_c, roi_color)
        
    # display resulating farme
    cv2.imshow('frame', frame)
    key = cv2.waitKey(1) & 0xFF 
    print(key)
    sleep(0.1)
    i = i + 1
    if key == 113:
        run = False
cap.release()
cv2.destroyAllWindows()