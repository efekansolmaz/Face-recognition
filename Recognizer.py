import cv2
import numpy as np
from PIL import Image
import os
import time

path='data'

recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml");

def imgsandlables (path):
    imagePaths = [os.path.join(path,i) for i in os.listdir(path)]     
    indfaces=[]
    ids = []
    for imagePath in imagePaths:
        img = Image.open(imagePath).convert('L')
        imgnp = np.array(img,'uint8')
        id = int(os.path.split(imagePath)[-1].split(".")[0])
        
        faces = detector.detectMultiScale(imgnp)
        for (x,y,w,h) in faces:
            indfaces.append(imgnp[y:y+h,x:x+w])
            ids.append(id)
    return indfaces,ids




#def distance (firstface,faces):
#    ((firstface[0]-faces[1][0])**2)+((firstface[1]-faces[1][1]**2))**1/2
    
faces,ids = imgsandlables (path)
#recognizer.train(faces, np.array(ids))
recognizer.read('C:\\Users\\efekan\\Desktop\\Real-time-Face-Recognition-using-OpenCV-and-webcam-1\\trainingData.yml')
#firstface = []
#faces[0] = firstface
#print(firstface)
names = [0]

cam= cv2.VideoCapture(0)

while True:
    _, img =cam.read()
    img = cv2.flip(img, 1) 
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    time.sleep(0.2)
    faces = detector.detectMultiScale( gray, scaleFactor = 1.3, minNeighbors = 5,)

    for(x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
        #print("0" , faces)

        id, confidence = recognizer.predict(gray[y:y+h,x:x+w])
        #print(x,y,w,h)
        old = []
        old = np.array(faces[0])
        # Check if confidence is less them 100 ==> "0" is perfect match 
        if (confidence < 85):
            confidence = "  {0}%".format(round(100 - confidence))
            
        else:
            id += 1
            #print(id)
            names.append(id)
            count = 0
            while(True):
                _, img = cam.read()
                img = cv2.flip(img, 1) 
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = detector.detectMultiScale(gray,scaleFactor = 1.3,minNeighbors = 5 )
                for (x,y,w,h) in faces:
                    cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
                    #print(x,y,w,h)
                    count += 1
                    time.sleep(0.2)
                    
                    #faces = np.array(faces)
                    #firstface = np.array(faces[0])
                    #distances = np.linalg.norm(faces-firstface, axis=1)
                    #min_index = np.argmin(distances)
                    #print(min_index)
                    
                    print(old)
                    if (len(faces) > 1):
                        if ((((old[0]-faces[1][0])**2)+((old[1]-faces[1][1]**2))**1/2) > (((old[0]-old[0])**2)+((old[1]-old[1]**2))**1/2)):
                            #min_val = np.min(faces)
                            #print(min_val)
                            print("1",faces)

                            cv2.imwrite("data/" + str(id) + '.' +  str(count) + ".jpg", gray[faces[1][1]:faces[1][1]+faces[1][3],faces[1][0]:faces[1][0]+faces[1][2]])
                            cv2.imshow('image', img)
                        else :
                            print((((old[0]-faces[1][0])**2)+((old[1]-faces[1][1]**2))**1/2) , (((old[0]-old[0])**2)+((old[1]-old[1]**2))**1/2))
                            print("2",faces)
                            cv2.imwrite("data/" + str(id) + '.' +  str(count) + ".jpg", gray[faces[0][1]:faces[0][1]+faces[0][3],faces[0][0]:faces[0][0]+faces[0][2]])
                            cv2.imshow('image', img)
                    else:
                        print("3",faces)
                        cv2.imwrite("data/" + str(id) + '.' +  str(count) + ".jpg", gray[y:y+h,x:x+w])
                        cv2.imshow('image', img)
                
                k = cv2.waitKey(100) &  0xFF == ord('s')
                if k == 10:
                    break
                elif count >= 10: 
                    break
            print(names)
            #print(ids)
            faces,ids = imgsandlables (path)
            recognizer.train(faces, np.array(ids))
            recognizer.write('C:\\Users\\efekan\\Desktop\\Real-time-Face-Recognition-using-OpenCV-and-webcam-1\\trainingData.yml')

        cv2.putText(img, str(id), (x-5,y-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)
        cv2.putText(img, str(confidence), (x+5,y+h-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2) 
    cv2.imshow('camera',img) 

    k = cv2.waitKey(10) & 0xff 
    if k == 27:
        break
