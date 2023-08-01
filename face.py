import cv2

def detect_bounding_box(vid):
    classifier = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    grayscale_image = cv2.cvtColor(vid, cv2.COLOR_BGR2GRAY)
    faces = classifier.detectMultiScale(grayscale_image, 1.1, 6, minSize=(40, 40))
    for (x, y, w, h) in faces:
        w = h
        height = h
        #i made the images the same length and width
        cv2.rectangle(vid, (x, y), (x + w, y + h), (0, 255, 0), 4)
    if cv2.waitKey(1) & 0xFF == ord("c"):
        print ('c')
        face = grayscale_image[y:y+height, x:x+height]

        face = cv2.resize(face, (48,48))

        return face, True
    #to take pictures, press c while its running
    #wait like 2 seconds between each picture
    return None, False

def GetFace():
    height = 0
    vid_capture = cv2.VideoCapture(0)
    while True:
      result, vid_frame = vid_capture.read()
      if result == False:
        break
    
      faces, result = detect_bounding_box(vid_frame)
      if result:
        vid_capture.release()
        cv2.destroyAllWindows()
        return faces
      cv2.imshow("Big Brother", vid_frame)
      if cv2.waitKey(1) & 0xFF == ord("q"):
        break
        #to stop the program, press q

    vid_capture.release()
    cv2.destroyAllWindows()