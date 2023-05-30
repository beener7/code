import numpy as np,cv2

def preprocessing(car_no):
    image = cv2.imread("images/car/%02d.jpg" %car_no, cv2.IMREAD_COLOR)
    if image is None: return None,None

    kernel = np.ones((5,17),np.uint8) # 닫힘 연산 마스크(커널)
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) #명암도 영상 변환
    gray = cv2.blur(gray,(5,5)) #블러링
    gray = cv2.Sobel(gray,cv2.CV_8U,1,0,3) #수직 에지 검출

    th_img = cv2.threshold(gray,120,255,cv2.THRESH_BINARY)[1] #이진화 수행
    morph = cv2.morphologyEx(th_img,cv2.MORPH_CLOSE,kernel,iterations = 3)

    #cv2.imshow("th_img",th_img);cv2.imshow("morph",morph) #결과 표시
    return image,morph


def verify_aspect_size(size):
    w,h = size
    if h == 0 or w == 0: return False

    aspect = h/w if h > w else w/h # 세로가 길면 역수 취함
    chk1 = 3000<(h*w) < 12000 #번호판 넓이 조건
    chk2 = 2.0<aspect <6.5 #번호판 종횡비 조건
    return (chk1 and chk2)


def find_candidates(image):
    results = cv2.findContours(image,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    contours = results[0] if int(cv2.__version__[0]) >= 4 else results[1]

    rects = [cv2.minAreaRect(c) for c in contours] #회전 사각형 반환
    candidates = [(tuple(map(int,center)),tuple(map(int,size)),angle) # 정수형 변환
                   for center, size, angle in rects if verify_aspect_size(size)]

    return candidates