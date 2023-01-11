#Attendence Management using face recognition
#@Mukesh
import cv2
from MyfaceRecModule import MyFaceRec

def FaceRecognitionFromVideo():
    textThikness = 2
    boxThikness = 2
    fontScale = 1
    blackColor = (0,0,0)
    limeColor = (0,255,0)
    esc_press = 27

    myFaceRecObj = MyFaceRec()
    myFaceRecObj.encode_known_faces("images/")

    # Load Camera
    video_Capture = cv2.VideoCapture(1)         

    while True:
        ret, frame = video_Capture.read()
        cv2.putText(frame,"Press Esc to exit",(10,20),cv2.FONT_ITALIC,fontScale,blackColor,textThikness)

        # Detect Faces
        face_locations, face_names, matchPercentages = myFaceRecObj.detect_known_faces(frame)
        for face_loc, name, mp in zip(face_locations, face_names, matchPercentages):
            y1, y2, x2, x1 = face_loc
            name = name + " " + str("{:.2f}".format(mp))
            cv2.putText(frame, name,(x1, y1 - 10), cv2.FONT_ITALIC, fontScale, limeColor, textThikness)
            cv2.rectangle(frame, (x1, y1), (y2, x2), limeColor, boxThikness)

        window_name = "Live Face Recognition"
        cv2.imshow(window_name, frame)

        key = cv2.waitKey(1)
        if key == esc_press:
          break
    
    myFaceRecObj.print_attendence_log()
    video_Capture.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)

if __name__ == "__main__":
    FaceRecognitionFromVideo()