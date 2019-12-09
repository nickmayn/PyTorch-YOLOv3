import argparse

import cv2
import mss
import numpy as np
import torch
import time
import pandas as pd
import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 

from models import *
from utils.utils import *


class VideoLoader:
    def __init__(self, path):
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise RuntimeError()

    def __next__(self):
        if not self.cap.isOpened():
            raise StopIteration()

        ret, image = self.cap.read()
        if not ret:
            self.cap.release()
            raise StopIteration()

        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def __iter__(self):
        return self

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", type=str, default="data/samples", help="path to save images")
    parser.add_argument("--data", type=str, default="locations.csv", help = "path to data file")
    parser.add_argument("--vid", type=str, default="../UVA4-13-19Game2.mp4", help="path to video file")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    
    opt = parser.parse_args()
    print(opt)
    
    # read in our pitch locations
    pitches = pd.read_csv(opt.data, header=None)
    strikezone = pd.read_csv('lab.txt', sep= " ", header=None)
    strikezone.columns = ['frame', 'label', 'cx', 'cy', 'w', 'h']
    pitches.columns = ['x', 'y','label', 'frame']
    
    pitches = pitches[pitches.label == 'ball']
    
    vid = VideoLoader(opt.vid)
    
    for frameID, image in enumerate(vid):
        if(frameID in list(pitches.frame)):
            org_h, org_w = image.shape[:2]
            
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            xval = pitches.x[pitches.frame == frameID]
            yval = pitches.y[pitches.frame == frameID]
            zoneval = strikezone[strikezone.frame == frameID]
            
            org_h, org_w = image.shape[:2]
            x1 = int((zoneval.cx - zoneval.w / 2) * org_w)
            y1 = int((zoneval.cy - zoneval.h / 2) * org_h)
            x2 = int((zoneval.cx + zoneval.w / 2) * org_w)
            y2 = int((zoneval.cy - zoneval.h / 2) * org_h)
            x3 = int((zoneval.cx - zoneval.w / 2) * org_w)
            y3 = int((zoneval.cy + zoneval.h / 2) * org_h)
            x4 = int((zoneval.cx + zoneval.w / 2) * org_w)
            y4 = int((zoneval.cy + zoneval.h / 2) * org_h)
            
            # begin work on scaling.
            xpad = x1 - 100
            ypad = y1 - 100
            xMaxpad = x4 + 100
            yMaxpad = y4 + 100
            
            # find the range
            xran = xMaxpad - xpad
            yran = yMaxpad - ypad
            
            
            
            print(str(x1)+ ' ' + str (y1) + '\n' + str(x2)+ ' ' + str (y2) + '\n'+ str(x3)+ ' ' + str (y3) + '\n'+ str(x4)+ ' ' + str (y4))
            cv2.rectangle(image, (x1, y1), (x4, y4), (255, 0, 0, 255), thickness=2)
            for i in range(len(xval)):
                cv2.circle(image, (xval.iloc[i], yval.iloc[i]), 10, (140, 255, 0) , 3)
                plocx = (xval.iloc[i] - xpad)/xran
                plocy = (yval.iloc[i] - ypad)/yran
                f = open("output.csv", "a")
                f.write(str(int(plocx*240)) + " " + str(int(plocy*240)) + '\n')
                f.close()
            cv2.imshow('image', image)
            key = cv2.waitKey()
            if key == 27 or key == ord('q'):
                break
            if key == ord('c'):
                continue
        
    
    
    
    
    
    