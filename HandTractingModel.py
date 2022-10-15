import cv2 as cv
import mediapipe as mp
import time


mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

class handDetector():
    def __init__(self, mode=False, maxHand=2, detectionCon=1, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHand
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

        self.tipsIds = [4, 8, 12, 16, 20]
    def findHands(self, img, draw=True):
        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.results = hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                # 编号和地标
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img
    def findPosition(self, img, handNo=0, draw=True):
        xList=[]
        yList = []
        bbox = []
        self.lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                # print(id, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                xList.append(cx)
                yList.append(cy)
                # print(id, cx, cy)
                self.lmList.append([id, cx, cy])
                # if id == 0:
                if draw:
                    cv.circle(img, (cx, cy), 10, (0, 0, 255), cv.FILLED)

            xmin, xmax = min(xList), max(xList)
            ymin, ymax = min(yList), max(yList)
            bbox = xmin, ymin, xmax, ymax
            if draw:
                cv.rectangle(img, (xmin-20, ymin-20), (xmax+20, ymax+20), (0, 255, 0), 2)

        return self.lmList, bbox

    def fingersUp(self):
        fingers = []
        # print(lmList)
        # 大拇指 左手
        if self.lmList[self.tipsIds[0]][1] > self.lmList[self.tipsIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
        # 其他四指
        for id in range(1, 5):
            if self.lmList[self.tipsIds[id]][2] < self.lmList[self.tipsIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        return fingers



def main():
    # 摄像头
    cap = cv.VideoCapture(0)

    pTime = 0
    cTime = 0

    detector = handDetector()
    while True:
        success, img = cap.read()
        img = detector.findHands(img)

        lmList = detector.findPosition(img)
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



if __name__ == '__main__':
    main()
