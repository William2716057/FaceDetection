import pathlib
import cv2

cascade_path = pathlib.Path(cv2.__file__).parent.absolute() / "data/haarcascade_frontalface_default.xml"
print(cascade_path)

#build classifier based on haarcascade file
clf = cv2.CascadeClassifier(str(cascade_path)) #needs to be cast as string

camera = cv2.VideoCapture(0) #should be 0 to set as computer's default


