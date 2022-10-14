import cv2 as cv
import mediapipe as mp
import time
import HandTractingModel as htm



# 摄像头
cap = cv.VideoCapture(0)
pTime = 0
cTime = 0

detector = htm.handDetector()
while True:
    success, img = cap.read()
    img = detector.findHands(img, draw=True)
    lmList = detector.findPosition(img, draw=True)
    if len(lmList) != 0:
        print(lmList[8])
    # fps
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    # 显示fps到屏幕
    cv.putText(img, str(int(fps)), (10, 70), cv.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    cv.imshow('image', img)
    cv.waitKey(1)