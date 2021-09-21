import cv2                                            #import opencv

thres = 0.45                                          # Threshold to detect object

cap = cv2.VideoCapture(0)                             #to capture webcam input
cap.set(3,1280)                                       #length
cap.set(4,720)                                        #width
cap.set(10,150)                                       #brightness

classNames= []
classFile = 'coco.names'                              #The “COCO format” is a specific JSON structure dictating how labels and metadata are saved for an image dataset
with open(classFile,'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')    #Extraction of coco data set to the program

configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'    #SingleShot Detection model to detect objects
weightsPath = 'frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weightsPath,configPath)
net.setInputSize(320,320)
net.setInputScale(1.0/ 127.5)                         #Line 18 to 21 are the parameters to detect the object
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

while True:
    success,img = cap.read()                                             #Reading the image
    classIds, confs, bbox = net.detect(img,confThreshold=thres)          #using the net method to detect the objects.It will give us the Confidence Values,bounding  box
    print(classIds,bbox)                                                 #and the ClassIds

    if len(classIds) != 0:
        for classId, confidence,box in zip(classIds.flatten(),confs.flatten(),bbox):
            cv2.rectangle(img,box,color=(255,0,0),thickness=2)                   #To display a rectangle
            cv2.putText(img,classNames[classId-1].upper(),(box[0]+10,box[1]+30), #putting the text and confidence box
                        cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
            cv2.putText(img,str(round(confidence*100,2)),(box[0]+200,box[1]+30),
                        cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)

    cv2.imshow("Output",img)                                                     #displaying the output
    cv2.waitKey(1)