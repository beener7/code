from header.coin_preprocess import *
from header.coin_utils import *
from Common.histogram import draw_hist_hue

coin_no = 15
image,th_img = preprocessing(coin_no)
circles = find_coins(th_img)
coin_imgs = make_coin_img(image,circles)
coin_hists = [calc_histo_hue(coin) for coin in coin_imgs]

for i,img in enumerate(coin_imgs):
    h,w = 200,256
    hist_img = draw_hist_hue(coin_hists[i],(h,w,3))

    merge =np.zeros((h,w+h,3),np.uint8)
    merge[:,:w] = hist_img
    merge[:,w:] = cv2.resize(img,(h,h))
    cv2.imshow("hist&coin - "+str(i),merge)

cv2.waitKey(0)

