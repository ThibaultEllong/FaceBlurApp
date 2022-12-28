import cv2
import pathlib
import numpy as np


cascade_path = pathlib.Path(cv2.__file__).parent.absolute() / "data/haarcascade_frontalface_default.xml"

clf = cv2.CascadeClassifier(str(cascade_path)) # Defines the classifier used for face detection (haar cascade here)

camera = cv2.VideoCapture("guys.mp4") # Input video



while True:
    _, frame = camera.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = clf.detectMultiScale(
        gray, # Video input for detection
        scaleFactor=1.1, # Parameter specifying how much the image size is reduced at each image scale.
        minNeighbors=5, # Parameter specifying how many neighbors each candidate rectangle should have to retain it.
        minSize=(30,30), # Minimum possible object size. Objects smaller than that are ignored.
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    kernel = np.ones((30,30), np.float32)/900 # Blurring Kernel, size will vary the intensity of the blur
    blurred = cv2.filter2D(frame, -1, kernel) # Computing of the blurred image

    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y), (x+w, y+h),(0,0,0), -1) #Draws rectangles around face

    out = np.where(frame == np.array([0,0,0]), blurred, frame) # Conditionnal masking: where frame is black => blur
    cv2.imshow("Faces", out) # output

    if cv2.waitKey(1) == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()

#I used free stock footage found at: https://www.youtube.com/watch?v=tdHdy_JTMPk for testing by PexBell