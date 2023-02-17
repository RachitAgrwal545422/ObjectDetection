import cv2

# Load the training data and create a classifer for it
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Choose an image to detect faces in
img = cv2.imread('Hello.png')


############make this image as a gray scale##########
grayScaled_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#################Detect faces########################

#It will return scales where it is a face
face_cordinates = trained_face_data.detectMultiScale(grayScaled_img)
#face cordinates will have touples of left upper cordinate and height and width


#Now we have images so now we can draw squares around these coordinates
for (x_cordinate,y_cordinate,width,hight) in face_cordinates:
    cv2.rectangle(img,(x_cordinate,y_cordinate),(x_cordinate+width,y_cordinate+hight),(0,0,0),5)


#To see this file
cv2.imshow("hello",img)
#It will keep the file open else it will close the file instantly
cv2.waitKey()


print("Code Completed")