#Import required modules
import cv2
import dlib
from imutils import face_utils
import numpy as np
from sklearn.svm import SVC
from sklearn.externals import joblib
import math
#Set up some required objects
EMOTIONS = ['ANGRY', 'DISGUSTED', 'FEARFUL', 'HAPPY', 'SAD', 'SURPRISED', 'NEUTRAL']
detector = dlib.get_frontal_face_detector() #Face detector
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
data = {}
clf = joblib.load('savemodels/KNN.pkl')
print('Hello There! \nType 1 to specify video path \nType 2 to use Webcam\n')
choice = input()
if choice==1:
    print('Type in your path name:')
    video_capture = cv2.VideoCapture(raw_input())
else:   
    video_capture = cv2.VideoCapture(0)

def get_landmarks(image):
    detections = detector(image, 1)
    for k,d in enumerate(detections): #For all detected face instances individually
        shape = predictor(image, d) #Draw Facial Landmarks with the predictor class
        xlist = []
        ylist = []
        for i in range(1,68): #Store X and Y coordinates in two lists
            xlist.append(float(shape.part(i).x))
            ylist.append(float(shape.part(i).y))
            
        xmean = np.mean(xlist)
        ymean = np.mean(ylist)
        xcentral = [(x-xmean) for x in xlist]
        ycentral = [(y-ymean) for y in ylist]

        landmarks_vectorised = []
        for x, y, w, z in zip(xcentral, ycentral, xlist, ylist):
            landmarks_vectorised.append(w)
            landmarks_vectorised.append(z)
            meannp = np.asarray((ymean,xmean))
            coornp = np.asarray((z,w))
            dist = np.linalg.norm(coornp-meannp)
            landmarks_vectorised.append(dist)
            landmarks_vectorised.append(int(math.atan((y-ymean)/(x-xmean))*360/math.pi))
        data['landmarks_vectorised'] = landmarks_vectorised
    if len(detections) < 1: 
        data['landmarks_vestorised'] = "error"
        return False
    else:
        return True   

while True:
    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    clahe_image = clahe.apply(gray)
    lm = get_landmarks(clahe_image)
    detections = detector(clahe_image, 1) #Detect the faces in the image
    if lm:
        array = np.array(data['landmarks_vectorised'])
        label = clf.predict(array.reshape(1,-1))    
    for k,d in enumerate(detections): #For each detected face 
        (x, y, w, h) = face_utils.rect_to_bb(d)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, EMOTIONS[int(label)], (x - 10, y - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.imshow("image", frame) #Display the frame
    if cv2.waitKey(1) & 0xFF == ord('q'): #Exit program when the user presses 'q'
        break