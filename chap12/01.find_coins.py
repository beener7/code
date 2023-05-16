from header.coin_preprocess import *
from header.coin_utils import *
from Common.histogram import draw_hist_hue
import cv2

image,th_img = preprocessing(70)
if image is None: raise Exception("영상 파일 읽기 에러")

circles = find_coins(th_img)
for center, radius in circles:
    cv2.circle(image,center,radius,(0,255,0),2)


cv2.imshow("preprocessed image",th_img)
cv2.imshow("coin image",image)
cv2.waitKey(0)