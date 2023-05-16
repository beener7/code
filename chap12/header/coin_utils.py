import numpy as np,cv2


def make_coin_img(src,circles):
    coins = []
    for center,radius in circles:
        r = radius * 3
        cen = (r//2, r//2)
        mask = np.zeros((r,r,3),np.uint8)
        cv2.circle(mask,cen,radius,(255,255,255),cv2.FILLED)
        #cv2.imshow("mask_"+str(center),mask)

        coin = cv2.getRectSubPix(src,(r,r),center)
        coin = cv2.bitwise_and(coin,mask)
        coins.append(coin)
    return coins

def calc_histo_hue(coin):
    hsv = cv2.cvtColor(coin,cv2.COLOR_BGR2HSV)
    hsize,ranges = [32],[0,180]
    hist = cv2.calcHist([hsv],[0],None,hsize,ranges)
    return hist.flatten()