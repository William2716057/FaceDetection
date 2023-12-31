import pathlib
import cv2

cascade_path = pathlib.Path(cv2.__file__).parent.absolute() / "data/haarcascade_frontalface_default.xml"
print(cascade_path)

#build classifier based on haarcascade file
clf = cv2.CascadeClassifier(str(cascade_path)) #needs to be cast as string

camera = cv2.VideoCapture(0) #should be 0 to set as computer's default

while True:
    _, frame = camera.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = clf.detectMultiScale(
        gray, 
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

#plotting
    for (x, y, width, height) in faces:
        cv2.rectangle(frame, (x, y), (x+width, y+height), (0, 255, 0), 2)#display in green box

        text = "Face Detected"
        cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
        # Display the frame with rectangles around faces
        cv2.imshow("Faces", frame)

        if cv2.waitKey(1) == ord("q"):
            break

camera.release()
cv2.destroyAllWindows()




