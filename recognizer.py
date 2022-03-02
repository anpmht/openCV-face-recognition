import cv2
import numpy as np
import pickle
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_default.xml')

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainner.yml")
run = True
i  = 0
while run:
    ret, frame = cap.read()
    frame = cv2.imread("images/kit-harington/6.jpg", cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,1.1, 4)
    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0),2)
        roi_gray = gray[y:y+h, x:x+h]
        roi_color = frame[y:y+h, x:x+h]
        id_ , conf = recognizer.predict(roi_gray)
        
        print(id_,conf)


        if conf>=4 and conf <= 85:
            font = cv2.FONT_HERSHEY_SIMPLEX
            if(id_ == 0):
                name = "Anup_Marahatta"
            elif(id_ ==1):
                name = "Emilia Clarke"
            elif(id_ ==2):
                name = "Justin"
            elif(id_ ==3):
                name = "Kit harington"
            elif(id_ ==4):
                name = "nicolaj coster"
            elif(id_ ==5):
                name = "peter dinklage"

            color = (255, 255, 255)
            stroke = 2
            cv2.putText(frame, name, (x,y), font, 1, color, stroke, cv2.LINE_AA)
        


        img_item_g = f"my_image_g_{i}.png"
        img_item_c = f"my_image_c_{i}.png"

        # cv2.imwrite(img_item_g, roi_gray)
        # cv2.imwrite(img_item_c, roi_color)
        
    # display resulating farme
    cv2.imshow('frame', frame)
    key = cv2.waitKey(1) & 0xFF 
    print(key)
    if key == 113:
        run = False
cap.release()
cv2.destroyAllWindows()