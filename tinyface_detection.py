import argparse
import cv2
import numpy as np
#from moviepy.editor import *
import csv
import os
import time
from prlab.detection.tinyface.detection import TinyFaceDetection
from prlab.utils.video import VideoReader
import tensorflow as tf

def key_faces(bboxes):
    key_faces = []
    if len(bboxes)>0:
        key_cmp   = []
        key_position_x =[]
        key_position_y =[]
        key_width =[]
        key_height =[]
        
        key_id    = list(range(len(bboxes)))
        for i, d in enumerate(bboxes):
            (x, y, w, h) = (int(d[0]), int(d[1]), int(d[2]), int(d[3]))
            key_cmp.append(w*h)
            key_position_x.append(x)
            key_position_y.append(y)
            key_width.append(w)
            key_height.append(h)
        key_faces = list(zip(key_cmp, key_position_x, key_position_y, key_width, key_height))
        key_faces.sort(reverse=True)
    return key_faces[0]





path     = 'dfdc_train_part_0'         #영상있는 폴더
path_out = 'dfdc_train_part_0_out'     #이미지 만들 폴더





data_split = os.listdir(path)
count = 0

for fname in data_split:
    

    if fname.endswith('.mp4'):
        count +=1
        start = time.time()

        if os.path.exists(os.path.join(path_out,fname[:-4])) == False:
            os.makedirs(os.path.join(path_out,fname[:-4]))

        detector = TinyFaceDetection.getDetector()

        cap = cv2.VideoCapture(os.path.join(path,fname))
        i=0
        dem = 0
        frame = 1
        while(cap.isOpened()):
            ret, img = cap.read()
            if ret == False:
                break
            if frame%10 == 1:   
                bboxes, _  = detector.detect(img)

                if len(bboxes)>0:

                    dem = dem + 1


                    d = key_faces(bboxes)

                    key_face = [max(0,int(d[1])), max(0,int(d[2])), max(0,int(d[3])), max(0,int(d[4]))]


                    (x,y,w,h)= (int(key_face[0]), int(key_face[1]), int(key_face[2]), int(key_face[3]))


                    #  x, y, w, h 가 얼굴 자를 좌표

                    if img[y:y+h,x:x+w].shape[0]>0 and img[y:y+h,x:x+w].shape[1]>0:
                        k_face=img[y:y+h,x:x+w]
                        cv2.imwrite(os.path.join(path_out,fname[:-4]+'_','1_{:04d}.jpg'.format(dem)), k_face)

            i+=1
            frame+=1

        cap.release()
        end = time.time() 
        print(fname, ' : ', round(end-start,5),'sec')
       
