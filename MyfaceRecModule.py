##################################
# Copyright:mukesh9871@gmail.com #
##################################

import face_recognition
import cv2
import os
import glob
import numpy as np
from datetime import datetime

class MyFaceRec:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []

        # Resize frame for a faster speed
        self.thresholdValue = 0.6
        self.frame_resizing = 0.25
        self.face_attendence_log = dict()

    # Encode faces from a folder
    def encode_known_faces(self, images_path):
        """
        Load encoding images from path
        :param images_path:
        :return:
        """
        # Load Images
        images_list = glob.glob(os.path.join(images_path, "*.jpg"))

        print("{} images found that need to be encoded.".format(len(images_list)))

        # Store image encoding and names
        for img_path in images_list:
            img = cv2.imread(img_path)
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Get encoding
            img_encoding = face_recognition.face_encodings(rgb_img)[0]

            # Get the filename only from the initial file path.
            basename = os.path.basename(img_path)
            (filename, ext) = os.path.splitext(basename)

            # Store file name and file encoding
            self.known_face_encodings.append(img_encoding)
            self.known_face_names.append(filename)
        print("Images encoded and loaded")

    #detect the 
    def detect_known_faces(self, frame):
        small_frame = cv2.resize(frame, (0, 0), fx=self.frame_resizing, fy=self.frame_resizing)
        # Find all the faces and face encodings in the current frame of video
        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        face_locations_list = face_recognition.face_locations(rgb_small_frame)
        face_encoding_list = face_recognition.face_encodings(rgb_small_frame, face_locations_list)

        face_names = []
        face_match_percentages = []
        faceLocations = []
        for face_encoding in face_encoding_list:
            # See if the face is a match for the known face(s)
            #matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
            name = "Unknown"
            matchPer = 1.000

            # The known face with the smallest distance to the new face
            face_distances_list = []
            face_distances_list = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances_list) #argmin() function return index of minimum value
            #if matches[best_match_index]:
            if face_distances_list[best_match_index] <= self.thresholdValue:
                name = self.known_face_names[best_match_index]
                matchPer = face_distances_list[best_match_index]
            face_names.append(name)
            face_match_percentages.append(matchPer)

            #add face name and time in log
            now = datetime.now()
            dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
            if name in self.face_attendence_log.keys():
                previousTime = self.face_attendence_log[name]
                self.face_attendence_log[name] = [previousTime[0], dt_string]
            else:
                self.face_attendence_log[name] = [dt_string,dt_string]

        # Convert to numpy array to adjust coordinates with frame resizing quickly
        face_locations = np.array(face_locations_list)
        face_locations = face_locations * 4
        return face_locations, face_names, face_match_percentages


    def print_attendence_log(self):
        print(" Name    |       Entry           |      Exit")
        for key,value in self.face_attendence_log.items():
            print(key,"  | ",value[0]," | ",value[1])
