import jetson.inference
import jetson.utils
import time
import cv2
import numpy as np 

#some constants
input_URI = "peds.mp4"
threshold = 0.5
network = "ssd-mobilenet-v2"
#network = "ssd-inception-v2"
overlay = "box,lables,conf"
fpsFilt=0
dispW= 720
dispH=480
font=cv2.FONT_HERSHEY_SIMPLEX

# load the object detection network
net = jetson.inference.detectNet(network,  threshold)

# create video sources & outputs
cap = cv2.VideoCapture('ped.mp4')
cap.set(cv2.CAP_PROP_FRAME_WIDTH, dispW)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, dispH)


# process frames until the user exits
while cap.isOpened():

    timeStamp = time.time()

    # capture the next image
    ret,img = cap.read()
    if ret ==True:
        
        height = img.shape[0]
        width = img.shape[1]
        

    # convert to jetson inference format
        frame=cv2.cvtColor(img,cv2.COLOR_BGR2RGBA).astype(np.float32)
        frame=jetson.utils.cudaFromNumpy(frame)

        # detect objects in the image (with overlay)
        detections = net.Detect(frame,width,height)
        num_persons = 0
        #print("number of detections = ", len(detections))
        for detect in detections:
            ID=detect.ClassID
            item=net.GetClassDesc(ID)
            if item == "person":
                top= int(detect.Top)
                left=int(detect.Left)
                bottom=int(detect.Bottom)
                right=int(detect.Right)
                #center = int(detect.Center)
                cv2.rectangle(img,(left,top),(right,bottom),(0,0,255),2)
                num_persons = num_persons +1
                
        dt=time.time()-timeStamp
        timeStamp=time.time()
        fps=1/dt
        fpsFilt=.9*fpsFilt + .1*fps
        #print(str(round(fps,1))+' fps')
        cv2.putText(img,str(round(fpsFilt,1))+' fps',(0,30),font,1,(0,0,255),2)
        cv2.putText(img,str(round(num_persons))+' persons detected',(0,60),font,1,(0,230,255),2)
        cv2.imshow('detCam',img)
        if cv2.waitKey(1)==ord('q'):
            break
    else:
        break
cap.release()
cv2.destroyAllWindows()