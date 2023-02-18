import cv2


#Load the web cam
cam = cv2.VideoCapture(0)


#face haarcascade clasifiers
face_detector = cv2.CascadeClassifier('frontelface.xml')
smile_detector = cv2.CascadeClassifier('haarcascade_smile.xml')

while True:
    #read frame wise
    issuccess,frame = cam.read()
    #if can't read successfully then break
    if not issuccess: break
    #take a gray scale image
    grayScaled_img = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    #Find the rectangle of face and smiles
    faces = face_detector.detectMultiScale(grayScaled_img)

    #Now make a rectangle arround it
    for (x_cordinate,y_cordinate,width,hight) in faces:
        #Create a sub frame with rectange boundries
        theFace = frame[y_cordinate:y_cordinate+hight,x_cordinate:x_cordinate+width]

        cv2.rectangle(frame,(x_cordinate,y_cordinate),(x_cordinate+width,y_cordinate+hight),(100,200,25),3)
        #Now withing that rectangle we will search for a smile
        #For it again make it a gray scale
        faceGrayScale = cv2.cvtColor(theFace,cv2.COLOR_BGR2GRAY)
        #scale factor to make the image blur so that no distrbance 
        #At least 20 rectangles in a region then there is a smile
        smiles = smile_detector.detectMultiScale(faceGrayScale,scaleFactor = 1.7,minNeighbors = 20)
        #Now Show the smile text
        if len(smiles) > 0:
            cv2.putText(frame,'Smilling',(x_cordinate+width,y_cordinate+hight+40),fontScale = 3,fontFace = cv2.FONT_HERSHEY_PLAIN,color=(255,255,255))
    #Now print the frame
    cv2.imshow('Smile',frame)
    key = cv2.waitKey(1)
    if key == 81 or key == 113: break

cam.release()
cv2.destroyAllWindows()
print("Code success")