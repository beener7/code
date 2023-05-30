import numpy as np,cv2

def color_candidate_img(image,center):
    h,w = image.shape[:2]
    fill = np.zeros((h+2,w+2),np.uint8) #채움 행렬
    dif1, dif2 = (25,25,25),(25,25,25) #채움 색상 범위
    flags = 0xff00 + 4 +cv2.FLOODFILL_FIXED_RANGE #채움 방향 및 방법
    flags += cv2.FLOODFILL_MASK_ONLY #결과 영상만 채움

    pts = np.random.randint(-15,15,(20,2) )  #임의 좌표 20개 생성
    pts = pts + center #중심 좌표로 평행이동
    for x,y in pts: #임의 좌표 순회
        if 0 <= x < w and 0 <= y < h: #후보 영역 내부이면
            _,_,fill,_ = cv2.floodFill(image,fill,(x,y),255,dif1,dif2,flags) #채움 누적

    return cv2.threshold(fill,120,255,cv2.THRESH_BINARY)[1]



def rotate_plate(image,rect): #입력 영상, 하나의 번호판 영역 회전 사각형 정보
    center,(w,h),angle = rect # 번호판 영역 중심점,크기,회전 각도

    crop_img = cv2.getRectSubPix(image,(w,h),center) # 한 번호판 후보영역 가져오기
    crop_img = cv2.cvtColor(crop_img,cv2.COLOR_BGR2GRAY) #번호판 후보영역을 명암도 영상으로 변환
    return cv2.resize(crop_img,(144,28)) #번호판 후보영역을 (144,28)크기로 변환