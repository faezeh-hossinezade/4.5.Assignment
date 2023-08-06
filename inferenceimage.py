import cv2
import time
from networkx import enumerate_all_cliques
import tensorflow as tf
import numpy as np
from functools import partial
from TFLiteFaceDetector import UltraLightFaceDetecion
from TFLiteFaceAlignment import CoordinateAlignmentModel

fd = UltraLightFaceDetecion("OpenVtuber/weights/RFB-320.tflite",conf_threshold=0.88)
fa = CoordinateAlignmentModel("OpenVtuber/weights/coor_2d106.tflite")

# cap = cv2.VideoCapture(0)
image=cv2.imread("Assignment/Input/Align_face_input.webp")
color = (0, 0, 255)


# start_time = time.perf_counter()

boxes, scores = fd.inference(image)

for pred in fa.get_landmarks(image, boxes):
    for i,p in enumerate(np.round(pred).astype(np.int8)):
        cv2.circle(image, tuple(p), 4, color,-1)
        cv2.putText(image,str(i),tuple(p),cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,0.5,(0,255,0))
        
    lips_landmarks=[]
    # for i in []:
    #     lips_landmarks.append(pred[i])
    # lips_landmarks=np.array(lips_landmarks,dtype=np.int8)
    # print(lips_landmarks)
mask=np.zeros(image.shape,dtype=np.uint8)
cv2.drawContours(mask,[lips_landmarks],-1,(255,255,255),-1)
mask=mask/255
result=image*mask
# print(time.perf_counter() - start_time)

cv2.imshow("result", image)
cv2.imwrite("Assignment/Output/Output interfrenceimage.jpg",image)
cv2.waitKey(1000)

