import cv2
cam = cv2.VideoCapture(0)
detector = cv2.CascadeClassifier("cascades/data/haarcascade_frontalface_alt2.xml")


Num=0
while(True):
  ret, frame = cam.read()
  if ret == True:
   # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('frame', frame)
    cv2.imshow('gray', gray)
    faces = detector.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        Num=Num+1
        cv2.imwrite("traindata/inconnue"+str(Num) + ".jpg", gray[y:y+h,x:x+w])
        #cv2.imshow('frame',frame)
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break
    elif Num>100:
        break
  else : print("pb in caaaaaaaaaaaaaaam")
cam.release()
cv2.destroyAllWindows()
