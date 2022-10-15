import cv2 as cv
import numpy as np
import time
import os
import HandTractingModel as htm

brushTickness = 15
eraserTickness = 50

folderPath = 'Header'
myList = os.listdir(folderPath)
# print(myList)
overlayList = []

for imPath in myList:
    image = cv.imread(f'{folderPath}/{imPath}')
    overlayList.append(image)

# print(len(overlayList))
header = overlayList[0]
drawColor = (0, 0, 255)


cap = cv.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

detector = htm.handDetector()
xp, yp = 0, 0

imgCanvas = np.zeros((720, 1280, 3), dtype='uint8')
while True:
    success, img = cap.read()
    # img = cv.flip(img, 1)
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)
    if len(lmList) != 0:
        # print(lmList)

        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]

        fingers = detector.fingersUp()
        # print(fingers)

        # 模式选择
        if fingers[1] & fingers[2]:
            cv.rectangle(img, (x1, y1-50), (x2, y2+50), (255, 0, 255), cv.FILLED)
            print("Selection Mode")
            # 画面高度135
            if y1 < 135:
                if 250< x1 < 450:
                    header = overlayList[0]
                    drawColor = (0, 0, 255)
                elif 550< x1 < 750:
                    header = overlayList[1]
                    drawColor = (255, 0, 0)
                elif 800< x1 < 950:
                    header = overlayList[2]
                    drawColor = (0, 255, 0)
                elif 1050< x1 < 1200:
                    header = overlayList[3]
                    drawColor = (0, 0, 0)
                cv.circle(img, (x1, y1), 15, drawColor, cv.FILLED)
            xp, yp = x1, y1

        if fingers[1] & fingers[2]==False:
            cv.circle(img, (x1, y1), 15, (255, 0, 255), cv.FILLED)
            print("Drawing Mode")
            if xp == 0 and yp == 0:
                xp, yp = x1, y1

            # 橡皮
            if drawColor == (0, 0, 0):
                cv.line(img, (xp, yp), (x1, y1), drawColor, eraserTickness)
                cv.line(imgCanvas, (xp, yp), (x1, y1), drawColor, eraserTickness)
            else:
                # draw
                cv.line(img, (xp, yp), (x1, y1), drawColor, brushTickness)
                cv.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushTickness)

            xp, yp = x1, y1

    # 合并画板
    imgGray = cv.cvtColor(imgCanvas, cv.COLOR_BGR2GRAY)
    _, imgInv = cv.threshold(imgGray, 50, 255, cv.THRESH_BINARY_INV)
    imgInv = cv.cvtColor(imgInv, cv.COLOR_GRAY2BGR)
    img = cv.bitwise_and(img, imgInv)
    img = cv.bitwise_or(img, imgCanvas)


    h, w, c = overlayList[0].shape
    img[0:135, 0:1280] = header

    # 合并通道
    # img = cv.addWeighted(img, 0.5, imgCanvas, 0.5, 0)

    cv.imshow('image', img)
    # cv.imshow('imageC', imgCanvas)
    # cv.imshow('Inv', imgInv)

    cv.waitKey(1)




