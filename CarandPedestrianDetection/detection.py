import cv2
######reading the image file#########
cap = cv2.VideoCapture('Hello.mp4')
#Now we will use pretrained algorithm using haarcascade
classfier_car = cv2.CascadeClassifier('cars.xml')
classfier_pedestrian = cv2.CascadeClassifier('pedestrian.xml')
while True:
    (isSuccessfull,frame) = cap.read()
    if not isSuccessfull: break
    #######Make it a gray scale image######
    grayscaleimg = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    #Now we will impliment this algorithm in our frame
    rectangles_1 = classfier_car.detectMultiScale(grayscaleimg)
    #Now we will create rectangles in our image
    for (x_cordinate,y_cordinate,width,hight) in rectangles_1:
        cv2.rectangle(frame,(x_cordinate,y_cordinate),(x_cordinate+width,y_cordinate+hight),(0,255,0),3)
    
    rectangles_2 = classfier_pedestrian.detectMultiScale(grayscaleimg)
    #Now we will create rectangles in our image
    for (x_cordinate,y_cordinate,width,hight) in rectangles_2:
        cv2.rectangle(frame,(x_cordinate,y_cordinate),(x_cordinate+width,y_cordinate+hight),(255,0,0),3)
    #Now just print our image
    cv2.imshow('Hello',frame)
    key = cv2.waitKey(1)
    #stop if Q is pressed
    if key == 80 or key == 113: break
cap.release()
print("Code completed")

