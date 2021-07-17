# code to detect and track people
# multiprocessing added
import jetson.inference
import jetson.utils
import time
import cv2
import numpy as np 

from imutils.video import FPS
import imutils
import dlib


#some constants - hello world
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

# initialize the list of object trackers and corresponding class
# labels
trackers = []
frame_num = 1
skip_frames = 30


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
    
    # convert to rgb for dlib processing
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if (len(trackers) == 0 or frame_num % skip_frames == 0):
            # detect objects in the image (with overlay)
            print("detection run for frame no = ", frame_num)
            if len(trackers) == 0:
                len_trackers_zero_run = True
            detections = net.Detect(frame,width,height)
            for detect in detections: #for each detections
                ID=detect.ClassID
                item=net.GetClassDesc(ID)
                if item == "person":
                    similarity = False   #for current detected item similarity initialised to flase
                    top= int(detect.Top)
                    left=int(detect.Left)
                    bottom=int(detect.Bottom)
                    right=int(detect.Right)
                    centerX = (right+left)/2
                    centerY = (bottom+top)/2

                    if len_trackers_zero_run: #if there are no objects in tracker then create the trackers with new detections
                        # construct a dlib rectangle object from the bounding
                        # box coordinates and start the correlation tracker
                        print("first run trackers initialized")
                        t = dlib.correlation_tracker()
                        rect = dlib.rectangle(left, top, right, bottom)
                        t.start_track(rgb, rect)
                        # update our set of trackers and corresponding class
                        trackers.append(t)
                    else: 
                        print("similarity check initialised") 
                        for t in trackers:
                            
                            pos = t.get_position()

                            # unpack the position object
                            startX = int(pos.left())
                            startY = int(pos.top())
                            endX = int(pos.right())
                            endY = int(pos.bottom())

                            tcenterX = (endX+startX)/2
                            tcenterY = (endY+startY)/2

                            a = np.array((centerX, centerY))
                            b = np.array((tcenterX,tcenterY))

                            dist = np.linalg.norm(a-b)
                            print("calculated distance = ", dist)
                            if dist < 50 : #if the tracker object is close than 50 pixels for the current item 
                                similarity = True           #iteration then  break   
                                break   
                            elif dist > 50:
                                similarity = False

                    if similarity == False:        
                        # construct a dlib rectangle object from the bounding
                        # box coordinates and start the correlation tracker
                        print("similarity false loop activated")
                        t = dlib.correlation_tracker()
                        rect = dlib.rectangle(left, top, right, bottom)
                        t.start_track(rgb, rect)

                        # update our set of trackers and corresponding class
                        trackers.append(t)
                            

                    # put text and draw the bounding box
                    cv2.putText(img, "detection mode", (left, top - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

                    cv2.rectangle(img,(left,top),(right,bottom),(0,0,255),2)
            if len_trackers_zero_run:
                print("total objects for zero run = ", len(trackers))
            len_trackers_zero_run = False        
        
        # otherwise, we've already performed detection so let's track
	    # multiple objects
        else:
            # loop over each of the trackers
            for t in trackers:
                # update the tracker and grab the position of the tracked
                # object
                t.update(rgb)
                pos = t.get_position()

                # unpack the position object
                startX = int(pos.left())
                startY = int(pos.top())
                endX = int(pos.right())
                endY = int(pos.bottom())

                # draw the bounding box from the correlation object tracker
                cv2.rectangle(img, (startX, startY), (endX, endY),
                    (0, 255, 0), 2)
                cv2.putText(img, "tracking", (startX, startY - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2) 

                
        dt=time.time()-timeStamp
        timeStamp=time.time()
        fps=1/dt
        fpsFilt=.9*fpsFilt + .1*fps
        #print(str(round(fps,1))+' fps')
        cv2.putText(img,str(round(fpsFilt,1))+' fps',(0,30),font,1,(0,0,255),2)
        cv2.putText(img,str(round(len(trackers)))+' persons detected',(0,60),font,1,(0,230,255),2)
        cv2.imshow('detCam',img)
        if cv2.waitKey(1)==ord('q'):
            break
    else:
        break

    frame_num += 1   

cap.release()
cv2.destroyAllWindows()