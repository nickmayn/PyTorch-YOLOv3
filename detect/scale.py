import argparse

import cv2
import mss
import numpy as np
import torch
import time
import pandas as pd
import os,sys,inspect
import matplotlib.pyplot as plt
from matplotlib import style
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 

from models import *
from utils.utils import *
style.use('ggplot')


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
    parser.add_argument("--output", type=str, default="./pitchCoordinates.csv", help="path to save coordinate file")
    parser.add_argument("--data", type=str, default="Pitch_Locations.csv", help = "path to load pitch data")
    parser.add_argument("--vid", type=str, default="../DukeGm1.mp4", help="path to video file")
    parser.add_argument("--zone", type=str, default='lab.txt', help="path to Strikezone dimension file")
    parser.add_argument("--fig", default=True, help="Determines if a figure will be saved with the plot or not.")
    
    opt = parser.parse_args()
    print(opt)  
    
    # Read in our pitch data and our strikezone data.
    pitches = pd.read_csv(opt.data, header=0)
    strikezone = pd.read_csv(opt.zone, sep= " ", header=None)
    strikezone.columns = ['frame', 'label', 'cx', 'cy', 'w', 'h']
    
    # Read in the video to get the dimensions of the video
    vid = VideoLoader(opt.vid)
    image = next(vid)
    org_h, org_w = image.shape[:2]
    
    # iterate through the frames we care about
    for frame in pitches.Frame:
        xval = pitches.cx_y[pitches.Frame == frame]
        yval = pitches.cy_y[pitches.Frame == frame]
            
        zoneval = strikezone[:1]
        
        org_h, org_w = image.shape[:2]
    
        x1 = int((zoneval.cx - zoneval.w / 2) * org_w)
        y1 = int((zoneval.cy - zoneval.h / 2) * org_h)        
        x4 = int((zoneval.cx + zoneval.w / 2) * org_w)
        y4 = int((zoneval.cy + zoneval.h / 2) * org_h)
    
        xran = abs(x4 - x1)
        yran = abs(y4 - y1)
        xpad = 60
        ypad = 50
    
        plocx = (int(125*(xval - x1)/xran)) + xpad
        plocy = (int(140*(yval - y1)/yran)) + ypad
            
        f = open(opt.output, "a")
        f.write(str(plocx) + "," + str(plocy) + "\n")
        f.close()
        
    if opt.fig:
        pitchLocs = pd.read_csv(opt.output, header = None)
        pitchLocs.columns = ['x', 'y']
    
        plt.scatter(pitchLocs.x, pitchLocs.y)
        plt.xlim(-10, 240)
        plt.ylim(250, -10)

        plt.hlines(y=50, xmin=60, xmax=185, linewidth=2, color='g', alpha = 0.3)
        plt.hlines(y=190, xmin=60, xmax=185, linewidth=2, color='g', alpha = 0.3)
        plt.vlines(x=60, ymin=50, ymax=190, linewidth=2, color='g', alpha = 0.3)
        plt.vlines(x=185, ymin=50, ymax=190, linewidth=2, color='g', alpha = 0.3)
        
        plt.xlabel("XOS X coordinate")
        plt.ylabel("XOS Y coordinate")
        plt.savefig("figure.pdf")
            
            
            
            