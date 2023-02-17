#face detection using Haar cascade
# you can read about haar features from
# https://www.analyticsvidhya.com/blog/2022/04/object-detection-using-haar-cascade-opencv/#:~:text=What%20are%20Haar%20Cascades%3F,%2C%20buildings%2C%20fruits%2C%20etc.
#basically we are implementing haar features and using supervised learning and traning with many 
'''In cascade classfier we have most accurate haar features for frontel face detection so we just need to implement 
these haar features in our image '''
#labelled images we get very high aqueracy in detecting objects
import cv2

# Load the training data and create a classifer for it
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Choose an image to detect faces in
#We will use open cv to use our default webcam
webcam = cv2.VideoCapture(0)
while True:
    #Now read the frames one by one from the video
    #read method will read the frame and return a bool and frame itself
    successful_frame_read,frame = webcam.read()
    ############make this frame as a gray scale##########
    grayScaled_img = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    ##Run the algorithm on it and find the coordinates of face on it
    face_cordinates = trained_face_data.detectMultiScale(grayScaled_img)
    #Now make a rectangle arround it
    for (x_cordinate,y_cordinate,width,hight) in face_cordinates:
        cv2.rectangle(frame,(x_cordinate,y_cordinate),(x_cordinate+width,y_cordinate+hight),(0,255,0),5)
    cv2.imshow("Hello",frame)
    key = cv2.waitKey(1)
    ####Stop if Q key is pressed
    if key == 81 or key == 113:
        break

print("Code Completed")