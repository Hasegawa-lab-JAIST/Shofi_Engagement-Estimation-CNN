'''
	-The given code extracts all the frames for the entire dataset and saves these frames in the folder of the video clips.
	-Kindly have ffmpeg (https://www.ffmpeg.org/) (all credits) in order to successfully execute this script.
	-The script must in the a same directory as the Dataset Folder.
'''

import os
#import subprocess
import cv2
#import glob
#glob.glob('*')

dataset = os.listdir('DataSet_DAiSEE/')
if '.DS_Store' in dataset:
    dataset.remove('.DS_Store')
facec = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#def split_video(video_file, image_name_prefix, destination_path):
#    return subprocess.check_output('ffmpeg -i "' + destination_path+video_file + '" ' + image_name_prefix + '%d.jpg -hide_banner', shell=True, cwd=destination_path)

def get_frame(video_file,destination_path):#extracting frames
    video = cv2.VideoCapture(destination_path+video_file) 
    _, fr = video.read() 
    #fr = cv2.flip(fr, -1)  
    # gray_fr = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY) 
    #grayscalling the picture
    faces = facec.detectMultiScale(fr, scaleFactor=1.3, minNeighbors=5)  
    for (x, y, w, h) in faces: 
        fc = fr[y:y+h, x:x+w] 
        
        roi = cv2.resize(fc, (48, 48)) 
        cv2.rectangle(fr,(x,y),(x+w,y+h),(255,0,0),2)   
        # Save the captured image into the datasets folder
        return cv2.imwrite(clip + ".jpg", roi)  
    #cv2.imshow('image', fc) 
    '''
    k = cv2.waitKey(100) & 0xff  
    if k == 27: 
        break 
    # Take 30 face sample and stop video
    elif count >= 30: 
        break 
          
def listdir_nohidden(dataset):
    return glob.glob(os.dataset.join(dataset, '*'))

''' 
for ttv in dataset:
    if not ttv.startswith('.'):
        #ttv.remove('.DS_Store')
        users = os.listdir('DataSet_DAiSEE/'+ttv+'/')
        for user in users:
            if not user.startswith('.'):
                currUser = os.listdir('DataSet_DAiSEE/'+ttv+'/'+user+'/')
                for extract in currUser:
                    if not extract.startswith('.'):
                        clip = os.listdir('DataSet_DAiSEE/'+ttv+'/'+user+'/'+extract+'/')[0]
                        print (clip[:-4])
                        path = os.path.abspath('.')+'/DataSet_DAiSEE/'+ttv+'/'+user+'/'+extract+'/'
                        #split_video(clip, clip[:-4], path)
                        get_frame(clip,path)
'''                      
for ttv in dataset:
    if '.DS_Store' in ttv: 
        ttv.remove('.DS_Store')
        users = os.listdir('DataSet/'+ttv+'/')
        for user in users:
            if '.DS_Store' in user:
                user.remove('.DS_Store')
                currUser = os.listdir('DataSet/'+ttv+'/'+user+'/')
                for extract in currUser:
                    if '.DS_Store' in extract:
                        extract.remove('.DS_Store')
                        clip = os.listdir('DataSet/'+ttv+'/'+user+'/'+extract+'/')[0]
                        print (clip[:-4])
                        path = os.path.abspath('.')+'/DataSet/'+ttv+'/'+user+'/'+extract+'/'
                        #split_video(clip, clip[:-4], path)
                        get_frame(clip,path)
                        
                        
print ("================================================================================\n")
print ("Frame Extraction Successful")



import os
cam = cv2.VideoCapture(0)
cam.set(3, 640) # set video width
cam.set(4, 480) # set video height
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# For each person, enter one numeric face id
face_id = input('\n enter user id end press <return> ==>  ')
print("\n [INFO] Initializing face capture. Look the camera and wait ...")
# Initialize individual sampling face count
count = 0
while(True):
    ret, img = cam.read()
    img = cv2.flip(img, -1) # flip video image vertically
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)     
        count += 1
        # Save the captured image into the datasets folder
        cv2.imwrite("dataset/User." + str(face_id) + '.' +  
                    str(count) + ".jpg", gray[y:y+h,x:x+w])
        cv2.imshow('image', img)
        k = cv2.waitKey(100) & 0xff # Press 'ESC' for exiting video
    if k == 27:
        break
    elif count >= 30: # Take 30 face sample and stop video
         break
 
    
 def get_frame(self): #extracting frames
        _, fr = self.video.read()
        gray_fr = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        faces = facec.detectMultiScale(gray_fr, scaleFactor=1.3, minNeighbors=5) #grayscalling the picture
                       
        for (x, y, w, h) in faces:
            fc = gray_fr[y:y+h, x:x+w]
            
            roi = cv2.resize(fc, (48, 48))
'''