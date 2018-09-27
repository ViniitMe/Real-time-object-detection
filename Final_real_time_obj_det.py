'''Title: Object Detection and Name calling      
   9th June 2018
   Author:Vinit'''
        

import cv2
import imutils
import numpy as np
import pyttsx as tx
from statistics import mode,median 

def say(put):
	engine=tx.init()
	voices=engine.getProperty('voices')
	engine.setProperty('voice',voices[2].id)
	engine.setProperty('rate',100)
	engine.say(put) 
	engine.runAndWait()

num=[]

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat","bottle", "bus", "car", "cat", "chair", "cow", "table","dog", "horse", "motorbike", "person", "pottedplant", "sheep","sofa", "train", "Digital-display"]

#We are going to leave the classes of ignore set (14 things can be setected now)
ignore=set(["aeroplane","boat","cow","horse","sheep","train"])

#loading pretrained model using opencv dnn module
model=cv2.dnn.readNetFromCaffe("MobileNetSSD_deploy.prototxt.txt","MobileNetSSD_deploy.caffemodel")

cap=cv2.VideoCapture(0)

while (1):
	ret,frame=cap.read()

	#grab the dimension of the image
	(h,w)=frame.shape[:2]

	#Image preprocessing with opencv blob function of dnn module
	blob=cv2.dnn.blobFromImage(cv2.resize(frame,(300,300)),0.007843,(300,300),127.5)
	#cv2.dnn.blobFromImage(image, scalefactor=1.0, size, mean, swapRB=True)

	model.setInput(blob)
	detections=model.forward()

	for i in np.arange(0,detections.shape[2]):
		confidence=detections[0,0,i,2]
		if confidence > 0.3:
			idx=int(detections[0,0,i,1])

			if CLASSES[idx] in ignore:
				continue

			num.append("{}".format(CLASSES[idx]))
			if(len(num)>30):
				try:
					print(mode(num))
					say((mode(num)))
					
				except:
					print(median(num))
					say(median(num))
				
				del num[:]
			
			box=detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")
			label="{}: {:.2f}%".format(CLASSES[idx],confidence*100)
			cv2.rectangle(frame,(startX,startY),(endX,endY),(0,255,0),2)
			cv2.putText(frame,label,(startX,startY-15),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)


	cv2.imshow("frame",frame)
	if cv2.waitKey(1) & 0xFF==ord('q'):
		break

cv2.destroyAllWindows()
cap.release()
