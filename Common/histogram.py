import numpy as np,cv2


def make_palette(rows):
    hue = [round(i*180/rows) for i in range(rows)]
    hsv = [[[h,255,255]] for h in hue]
    hsv = np.array(hsv,np.uint8)
    return cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)

def draw_hist_hue(hist,shape = (200,256,3)):
    hsv_palette = make_palette(hist.shape[0])
    hist_img = np.full(shape,255,np.uint8)
    cv2.normalize(hist,hist,0,shape[0],cv2.NORM_MINMAX)

    gap = hist_img.shape[1]/hist.shape[0]
    for i,h in enumerate(hist):
        x,w = int(round(i* gap)),int(round(gap))
        color = tuple(map(int,hsv_palette[i][0]))
        cv2.rectangle(hist_img,(x,0,w,int(h)),color,cv2.FILLED)

    return cv2.flip(hist_img,0)

