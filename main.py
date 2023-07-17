import math
import cv2
import cvzone
from cvzone.ColorModule import ColorFinder
import numpy as np

# detection by color

# cap = cv2.VideoCapture('./splash.mp4')
cap = cv2.VideoCapture('./court/hit2.mp4')

myColorFinder = ColorFinder(False)
hsvVals = {'hmin': 8, 'smin': 157, 'vmin': 112, 'hmax': 11, 'smax': 228, 'vmax': 170}

# {'hmin': 6, 'smin': 165, 'vmin': 120, 'hmax': 9, 'smax': 205, 'vmax': 138}
# {'hmin': 9, 'smin': 190, 'vmin': 99, 'hmax': 10, 'smax': 217, 'vmax': 136}
posList = []

results = open("locations.csv", 'w')

# EKF = open("locations1.csv", "r")

fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))
print(int(cap.get(3)), int(cap.get(4)))

x_fix = 457 / int(cap.get(4))
y_fix = 400 / int(cap.get(3))
# print(x_fix, y_fix)

# point = EKF.readline()
k = 0
ekf_i = []
while True:
    success, image = cap.read()
    img = cv2.imread('ball2.png')

    if success:
        imgColor, mask = myColorFinder.update(image, hsvVals)
        imgContours, contours = cvzone.findContours(image, mask, minArea=20)

        for k, p in enumerate(ekf_i):
            cv2.circle(imgContours, p, 3, (0, 255, 255), cv2.FILLED)
            if k > 0:
                cv2.line(imgContours, ekf_i[k], ekf_i[k - 1], (255, 0, 255), 2)

        if contours and 380 < contours[0]['center'][0] < 1500:
            posList.append(contours[0]['center'])

        for i, pos in enumerate(posList):
            cv2.circle(imgContours, pos, 3, (0, 255, 0), cv2.FILLED)
            if i == 0:
                cv2.line(imgContours, pos, pos, (0, 0, 255), 2)
            else:
                cv2.line(imgContours, pos, posList[i - 1], (0, 0, 255), 2)
            if i == len(posList) - 1:
                results.write(str(posList[i][0]) + ", " + str(int((cap.get(4)) - posList[i][1])) + "\n")
                # results.write(str(posList[i][0] * x_fix) + ", " + str(y_fix * int((cap.get(3))-posList[i][1])) +"\n")
                # print(posList[i][0] * x_fix, y_fix * int((cap.get(3)) - posList[i][1]))

        # imgContours = cv2.resize(imgContours, (0, 0), None, 0.7, 0.7)
        out.write(imgContours)
        cv2.imshow("ImageColor", imgContours)
        # cv2.imshow("ImageColor", imgColor)

        cv2.waitKey(1)
        # print(len(posList))

    else:
        break
