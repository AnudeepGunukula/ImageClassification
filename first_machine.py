from cv2 import cv2
import numpy as np
import pywt
import joblib


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
img = cv2.imread('test.jpeg')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.3, 5)
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_img = img[y:y+h, x:x+w]

img = roi_img
img = cv2.resize(img, (32, 32))
org = img
img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
img = np.float32(img)
img /= 255
# plt.figure(figsize=(20,20))
coef = pywt.wavedec2(img, 'haar', level=5)
coef_list = list(coef)
coef_list[0] *= 0
jpg = pywt.waverec2(coef_list, 'haar')
jpg *= 255
jpg = np.uint8(jpg)
# cv2.imshow('img',jpg)
jpg = cv2.resize(jpg, (32, 32))
comb = np.vstack((org.reshape(32*32*3, 1), jpg.reshape(32*32, 1)))
comb = comb.reshape(1, 4096).astype(float)
print(comb)
model = joblib.load('first_machine.pkl')
res = model.predict(comb)[0]
print(res)
stars = {0: 'conormcgregor', 1: 'kajal_agarwal',
         2: 'sakshimalik', 3: 'sunnyleone', 4: 'viratkohli'}

print(stars[res])
if res == 0:
    conor = cv2.imread('conor.jpeg')
    cv2.imshow('img', conor)
    cv2.waitKey(0)
elif res == 1:
    kajal = cv2.imread('kajal.jpeg')
    cv2.imshow('img', kajal)
    cv2.waitKey(0)
elif res == 2:
    sakshi = cv2.imread('sakshi.jpeg')
    cv2.imshow('img', sakshi)
    cv2.waitKey(0)
elif res == 3:
    sunny = cv2.imread('sunny.jpeg')
    cv2.imshow('img', sunny)
    cv2.waitKey(0)
elif res == 4:
    virat = cv2.imread('virat.jpeg')
    cv2.imshow('img', virat)
    cv2.waitKey(0)
