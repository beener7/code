from header.plate_preprocess import * # 전처리 및 후보 영역 검출 함수
from header.plate_candidate import * # 후보 영역 개선 및 후보 영상 생성 함수

car_no = int(input("자동차 영상 번호(0~15): "))
image, morph = preprocessing(car_no)
candidates = find_candidates(morph) # 번호판 후보 영역 검색

fills = [color_candidate_img(image,size) for size, _, _ in candidates] # 후보 영역 재생성
new_candis = [find_candidates(fill) for fill in fills] # 재생성 영역 검사
new_candis = [cand[0] for cand in new_candis if cand] # 재후보 있으면 저장
candidate_imgs = [rotate_plate(image,cand) for cand in new_candis] #후보 영역 영상

svm = cv2.ml.SVM_load("SVMtrain.xml") # 학습된 데이터 적재
rows = np.reshape(candidate_imgs,(len(candidate_imgs),-1)) # 1행 데이터들로 변환
_,results = svm.predict(rows.astype("float32")) #분류 수행
correct = np.where(results == 1)[0] #정답 인덱스 찾기

print("분류 결과:\n",results)
print("번호판 영상 인덱스:",correct)

for i, idx in enumerate(correct): #
    cv2.imshow("plate image_"+str(i),candidate_imgs[idx]) #후보 영역 영상 출력
    cv2.resizeWindow("plate image_"+str(i),(250,28)) # 윈도우 크기 조정

for i,candi in enumerate(new_candis):
    color = (0,255,0) if i in correct else(0,0,255) #후보 영역 확인 및 색 지정
    cv2.polylines(image,[np.int32(cv2.boxPoints(candi))],True,color,2) # 후보영역 표시


print("번호판 검출완료") if len(correct) > 0 else print("번호판 미검출")

cv2.imshow("image",image)
cv2.waitKey(0)
