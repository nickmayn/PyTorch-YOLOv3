from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *

import os
import sys
import time
import datetime
import argparse

from PIL import Image
import cv2

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator

WINDOW_NAME = 'YOLO'

def open_window(width, height):
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, width, height)
    cv2.moveWindow(WINDOW_NAME, 0, 0)
    cv2.setWindowTitle(WINDOW_NAME, 'YOLO')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", type=str, default="data/samples", help="path to dataset")
    parser.add_argument("--model_def", type=str, default="config/yolov3-custom.cfg", help="path to model definition file")
    parser.add_argument("--weights_path", type=str, default="checkpoints/yolov3_ckpt_345.pth", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="data/custom.names", help="path to class label file")
    parser.add_argument("--conf_thres", type=float, default=0.9, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.3, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument('--width', default=1280, type=int)
    parser.add_argument('--height', default=720, type=int)
    opt = parser.parse_args()
    print(opt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs("output", exist_ok=True)

    # Set up model
    model = Darknet(opt.model_def, img_size=opt.img_size).to(device)

    if opt.weights_path.endswith(".weights"):
        # Load darknet weights
        model.load_darknet_weights(opt.weights_path)
    else:
        # Load checkpoint weights
        model.load_state_dict(torch.load(opt.weights_path))

    model.eval()  # Set in evaluation mode

    classes = load_classes(opt.class_path)  # Extracts class labels from file

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    imgs = []  # Stores image paths
    img_detections = []  # Stores detections for each image index

    # Bounding-box colors
    cmap = plt.get_cmap("tab20b")
    bbox_colors = []
    for i in np.linspace(0, 1, 10):
        r,g,b,a = cmap(i)
        bbox_colors += [(b*255.0, g*255.0, r*255.0)]

    text_size = 4

    cap = cv2.VideoCapture('test.mp4')
    if not cap.isOpened():
        print('Cannot open video')
        exit(1)

    print("\nPerforming object detection:")
    prev_time = time.time()

    open_window(opt.width, opt.height)
    full_scrn = False

    cv2.waitKey(10)

    batch_i = 0
    while cap.isOpened():
        if cv2.getWindowProperty(WINDOW_NAME, 0) < 0:
            # Check to see if the user has closed the window
            # If yes, terminate the program
            break

        # Configure input
        ret, image = cap.read()
        if not ret:
            print("End of Videos")
            break

        img = transforms.ToTensor()(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).cuda()
        img, _ = pad_to_square(img, 0)
        input_imgs = F.interpolate(img.unsqueeze(0), size=opt.img_size, mode="nearest")

        # Get detections
        with torch.no_grad():
            detections = model(input_imgs)
            detections = non_max_suppression(detections, opt.conf_thres, opt.nms_thres)

        # Log progress
        current_time = time.time()
        inference_time = datetime.timedelta(seconds=current_time - prev_time)
        prev_time = current_time
        print("\t+ Batch %d, Inference Time: %s" % (batch_i, inference_time))
        batch_i += 1

        thickness = (image.shape[0] + image.shape[1]) // 600

        for box_attr in detections:
            if box_attr is None:
                continue

            detections = rescale_boxes(box_attr, opt.img_size, image.shape[:2])
            unique_labels = box_attr[:, -1].cpu().unique()

            # Rescale boxes to original image
            for x1, y1, x2, y2, conf, cls_conf, cls_pred in box_attr:
                print("\t+ Label: %s, Conf: %.5f" % (classes[int(cls_pred)], cls_conf.item()))
                if cls_conf < 0.6:
                    continue
                
                color = bbox_colors[int(cls_pred)]
                label = classes[int(cls_pred)] + f' {cls_conf * 100:.0f}%'
                cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

                (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)

                text_end = x1 + text_width + thickness, y1 - text_height - baseline - thickness // 2

                cv2.rectangle(image, (x1 - thickness // 2, y1), text_end, color, thickness=cv2.FILLED)
                cv2.putText(image, label, (x1, y1 - baseline), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

        cv2.imshow(WINDOW_NAME, image)
 
        # Press Q on keyboard to stop recording
        key = cv2.waitKey(10)
        if key == 27 or key == ord('q'):
            break
        elif key == ord('F') or key == ord('f'): # toggle fullscreen
            full_scrn = not full_scrn
            cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN,
                                    cv2.WINDOW_FULLSCREEN if full_scrn else cv2.WINDOW_NORMAL)
            
    cap.release()
    cv2.destroyAllWindows()
