'''
Shofiyati Nur Karimah 
Face detector using Viola&Jones detector from openCV

Last edit: 03/08/2021


References:
# https://towardsdatascience.com/real-time-eye-tracking-using-opencv-and-dlib-b504ca724ac6
# https://livecodestream.dev/post/detecting-face-features-with-python/
# https://www.programmersought.com/article/53015914776/
'''

import cv2
from model import FacialExpressionModel
import numpy as np
from datetime import datetime 
import csv
import dlib

facec = cv2.CascadeClassifier('sources/haarcascade_frontalface_default.xml') #load V&J detector opencv
detector = dlib.get_frontal_face_detector() #load the detector from dlib
predictor = dlib.shape_predictor("sources/shape_predictor_68_face_landmarks.dat") #load the predictor dlib
model = FacialExpressionModel("sources/eng_model_serv.json", "sources/eng_model_weights_serv.h5")
font = cv2.FONT_HERSHEY_SIMPLEX

start_time = datetime.now()
time_format = "{:%H:%M:%S}"
extension = "csv"
prefix = 'logs/log_engagement'
file_name_format = "{:s}-{:%Y%m%d_%H%M}.{:s}"
file_name = file_name_format.format(prefix, start_time, extension)
header = "Time, States, Confidence"
num_landmarks = 68

with open(file_name, "w") as f:
    f.write(header+"\n")
    
class VideoCamera(object):
    def __init__(self): #capturing video        
        self.video = cv2.VideoCapture(1) #the source of video
        #self.video = cv2.VideoCapture("/videos/facial_exp.mkv")

    def __del__(self): #releasing camera
        self.video.release()

    # returns camera frames along with bounding boxes and predictions
    def get_frame(self): #extracting frames
        _, fr = self.video.read()
        gray_fr = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        faces = facec.detectMultiScale(gray_fr, scaleFactor=1.3, minNeighbors=3) #grayscalling the picture
        faces_dlib = detector(gray_fr) #use detector to find landmarks using dlib
        
        for (x, y, w, h) in faces:
            fc = gray_fr[y:y+h, x:x+w]

            roi = cv2.resize(fc, (48, 48))
            conf, pred= model.predict_emotion(roi[np.newaxis, :, :, np.newaxis]) #feed the picture to pre-tained model
            confi = " {:.2f}%".format(conf*100)

            #prevPred = ""
            #currentPred = ""
            #currentPred=pred
            with open(file_name, "a+") as f:
                fieldnames = ['Time', 'State', 'Confidence']
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                tic = datetime.now()
                tic_format = time_format.format(tic)
                writer.writerow({'Time':str(tic_format), 'State':pred, 'Confidence':confi})

                #f.write(start_time_string + " " + pred + " " + confi + "\n")                
                #f.close()

            #if (currentPred != prevPred):
                #prevPred = currentPred
                #f.close()

            #draw rectangle
            cv2.putText(fr, pred+confi, (x, y), font, 1, (0, 255, 0), 4) #bounding boxing the face and adding label
            cv2.rectangle(fr,(x,y),(x+w,y+h),(0,255,0),4) #color= Blue, stroke=2

        for fc_dlib in faces_dlib:
            #frame += 1
            x1 = fc_dlib.left() #left point
            y1 = fc_dlib.top() 
            x2 = fc_dlib.right() 
            y2 = fc_dlib.bottom()

            ##draw rectacngle
            #cv2.rectangle(img=fr, pt1=(x1,y1), pt2=(x2,y2), color=(0,255,0), thickness=4)

            #look for the landmarks
            landmarks = predictor(image=gray_fr, box=fc_dlib)
            
            #loop for through all the points
            for n in range(num_landmarks):
                x = landmarks.part(n).x
                y = landmarks.part(n).y
            
                #draw circle
                cv2.circle(img=fr, center=(x, y), radius=3, color=(0,255,0), thickness=-1)         
        #encode OpenCV raw frame to jpg and displaying it
        _, jpeg = cv2.imencode('.jpg', fr)
        return jpeg.tobytes()
    
   