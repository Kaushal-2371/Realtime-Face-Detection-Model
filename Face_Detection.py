import cv2
import face_recognition
import numpy as np
import os
import csv
from datetime import datetime


#importing all images and encoding it
raw_face_encoding = []

os.chdir("C:\Kaushal\Machine Learning\Libraries\OpenCV\RawFace")
raw_face_name = os.listdir()
for i in range(len(raw_face_name)+1):
    img = face_recognition.load_image_file(raw_face_name[i])
    raw_face_encoding.append(face_recognition.face_encodings(img)[0])
    # raw_face_name[i] = raw_face_name[i].replace(".jpg","")
    # raw_face_name[i] = raw_face_name[i][0:raw_face_name[i].index(".")]

raw_face_name = [
     "Kaushal",
     "Pratyaksh",
     "Harsh Shrma",
     "Prem"
 ]

os.chdir("C:\Kaushal\Machine Learning\Libraries\OpenCV")

#making a copy of all known faces
students = raw_face_name.copy()

#taking some more variables
cam_face_location = []                  #for giving the location of the face found
cam_face_encoding = []                  #encoings of the face found
cam_face_name = []                      #names of the face found
s = True

#now taking a date and time to be printed for the file name
now = datetime.now()
cur_date = now.strftime("%Y-%m-%d (%M)")

#now making the csv file to write on it with name and date time
f = open(cur_date +".csv","w+", newline="")
inwriter = csv.writer(f)

#now capturing web cam
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
cap.set(10, 100)

while True:
    _, img = cap.read()
    small_frame = cv2.resize(img, (0,0), fx=0.25, fy=0.25)
    rgb_small_frame = small_frame[ :, :, ::-1]                      #converting it into bgr

    if s:
        cam_face_location = face_recognition.face_locations(rgb_small_frame)
        cam_face_encoding = face_recognition.face_encodings(rgb_small_frame,cam_face_location)
        cam_face_name = []
        for new_face_encoding in cam_face_encoding:
            matches = face_recognition.compare_faces(raw_face_encoding,new_face_encoding)
            name = ""
            face_distance = face_recognition.face_distance(raw_face_encoding, new_face_encoding)      #Return a 2d array of (top, bottom, left, right)

            best_match_index = np.argmin(face_distance)                   #to find the best probablity of the known face
            if matches[best_match_index]:
                name = raw_face_name[best_match_index]

            cam_face_name.append(name)
            if name in raw_face_name:
                if name in students:
                    students.remove(name)
                    print(name)
                    current_time = now.strftime("%H-%M-%S")
                    inwriter.writerow([name,current_time])


    cv2.imshow("output", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyALLWindows()
f.close()