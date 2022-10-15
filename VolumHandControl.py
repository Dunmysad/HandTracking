import cv2 as cv
import time
import numpy as np
import HandTractingModel as htm
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# pycaw
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
# volume.GetMute()
# volume.GetMasterVolumeLevel()

# (-65.25, 0.0, 0.03125)
volRange = volume.GetVolumeRange()

volume.SetMasterVolumeLevel(0, None)

# 设置声音最值
minVol = volRange[0]
maxVol = volRange[1]


wCam, hCam = 640, 480


cap = cv.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
pTime = 0
detector = htm.handDetector()
vol = 0
volBar = 400
volPer = 0

while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)
    if len(lmList) != 0:
        # print(lmList[4], lmList[8])

        x1, y1 = lmList[4][1], lmList[4][2]
        x2, y2 = lmList[8][1], lmList[8][2]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        cv.circle(img, (x1, y1), 8, (0, 0, 255), cv.FILLED)
        cv.circle(img, (x2, y2), 8, (0, 0, 255), cv.FILLED)
        cv.line(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
        cv.circle(img, (cx, cy), 8, (0, 0, 255), cv.FILLED)

        length = math.hypot(x2 - x1, y2 - y1)
        # print(length)

        # hands range 50 - 300
        # volume range -65 - 0

        vol = np.interp(length, [20, 280], [minVol, maxVol])
        volBar = np.interp(length, [50, 300], [400, 150])
        volPer = np.interp(length, [50, 300], [0, 100])
        print(int(length), vol)
        volume.SetMasterVolumeLevel(vol, None)

        if length < 20:
            cv.circle(img, (cx, cy), 8, (255, 0, 255), cv.FILLED)

    cv.rectangle(img, (50, 150), (85, 400), (0, 255, 0), 3)
    cv.rectangle(img, (50, int(volBar)), (85, 400), (0, 255, 0), cv.FILLED)
    cv.putText(img, f'{int(volPer)} %', (40, 450), cv.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 3)


    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime=cTime

    cv.putText(img, f'FPS:{int(fps)}', (30, 70), cv.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)
    cv.imshow('image', img)
    cv.waitKey(1)