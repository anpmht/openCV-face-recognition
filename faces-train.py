import cv2
import os
from  PIL import Image
import numpy as np
import pickle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
print(BASE_DIR)
image_dir = os.path.join(BASE_DIR,"images")
recognizer = cv2.face.LBPHFaceRecognizer_create()

y_labels = []
x_train = []
label_ids = {}
current_id = 0
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('cascades\data\haarcascade_frontalface_default.xml')
run = True
i  = 0
for root, dirs, files in os.walk(image_dir):
    for file in files:
        print(file)
        if file.endswith("png") or file.endswith("jpg"):
            path = os.path.join(root,file)
            label  = os.path.basename(root).replace(" ", "-").lower()
            #print(label, path)

            if not label in label_ids:
                label_ids[label] = current_id
                current_id += 1
                
            id_ = label_ids[label]
            # print(label_ids)
            # y_labels.append(label)
            # x_train.append(path)

            pil_image = Image.open(path).convert("L")
            size = (500,500)
            final_image = pil_image.resize(size,Image.ANTIALIAS)
            image_array = np.array(final_image, "uint8")
            #print(image_array.dtype)
            #print(image_array)
            faces = face_cascade.detectMultiScale(image_array,scaleFactor=1.5, minNeighbors=5)
            
            for (x,y,w,h ) in faces:
                roi = image_array[y:y+h, x:x+w]
                x_train.append(roi)
                y_labels.append(id_)

print(y_labels)
#print(x_train)

with open("labels.pickle", 'wb') as f:
    pickle.dump(label_ids,f)

recognizer.train(x_train, np.array(y_labels))
recognizer.save("trainner.yml")