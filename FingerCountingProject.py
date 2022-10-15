import cv2 as cv
import time
import os
import HandTractingModel as htm


wCam, hCam = 640, 480


cap = cv.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

folderPath = 'FingerImages'
myList = os.listdir(folderPath)
print(myList)
overlayList = []
for imPath in myList:
    image = cv.imread(f'{folderPath}/{imPath}')
    overlayList.append(image)
print(len(overlayList))
pTime = 0

detector = htm.handDetector()
tipsIds = [4, 8, 12, 16, 20]

while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)

    if len(lmList) != 0:
        fingers = []
        # print(lmList)
        # 大拇指 左手
        if lmList[tipsIds[0]][1] > lmList[tipsIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
        # 其他四指
        for id in range(1, 5):
            if lmList[tipsIds[id]][2] < lmList[tipsIds[id]-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        # print(fingers)
        totalFingers = fingers.count(1)
        print(totalFingers)


        h, w, c = overlayList[totalFingers].shape
        img[0:h, 0:w] = overlayList[totalFingers]
        cv.rectangle(img, (60, 250), (150, 400), (0, 255, 0), cv.FILLED)
        cv.putText(img, str(totalFingers), (60, 380), cv.FONT_HERSHEY_COMPLEX, 5, (255, 0, 0), 4)

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv.putText(img, f'FPS:{int(fps)}', (400, 70), cv.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
    cv.imshow('image', img)
    cv.waitKey(1)

