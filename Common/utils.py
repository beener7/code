import numpy as np,cv2,time

def put_string(frame,text,pt,value = None,color = (120,200,90)):
    text = str(text) + str(value)
    shade = (pt[0]+2,pt[1]+2)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame,text,shade,font,0.7,(0,0,0),2)
    cv2.putText(frame,text,pt ,font,0.7,color,2)