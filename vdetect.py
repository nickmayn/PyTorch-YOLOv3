import argparse
import datetime
import os
import sys
import time

import cv2
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import mss

from models import *
from utils.datasets import *
from utils.utils import *

def prepare_image(img):
    patch = opt.img_size

    img = img[:, :, :3]
    img = img.unsqueeze(0).permute(0, 3, 1, 2) # NHWC -> NCHW
    img *= 1/255.0

    n, c, h, w = img.shape
    dim_diff = np.abs(h - w)
    # (upper / left) padding and (lower / right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # Determine padding
    pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
    # Add padding
    img = F.pad(img, pad, "constant", value=0)

    img = F.interpolate(img, size=patch, mode="nearest")

    return img

class DataPrefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.next_image = next(self.loader)
            self.next_input = torch.from_numpy(self.next_image)
        except StopIteration:
            self.next_image = None
            self.next_input = None
            return

        if not torch.cuda.is_available():
            self.next_input = prepare_image(self.next_input.float())
            return

        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True).float()
            self.next_input = prepare_image(self.next_input)

    def __iter__(self):
        return self

    def __next__(self):
        if self.next_input is None:
            raise StopIteration

        if not torch.cuda.is_available():
            self.preload()
            return self.next_image, self.next_input

        image = self.next_image

        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        
        if input is not None:
            input.record_stream(torch.cuda.current_stream())

        self.preload()
        return image, input

class ScreenCapture():
    def __init__(self):
        self.sct = mss.mss()

        mon = self.sct.monitors[0]
        self.monitor = {
            "top": 46,
            "left": 0,
            "width": 1920,
            "height": 1080,
            "mon": mon,
        }

    def __next__(self):
        image = np.array(self.sct.grab(self.monitor))
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def __iter__(self):
        return self

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

WINDOW_NAME = 'YOLO'
def open_window(width, height):
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, width, height)
    cv2.moveWindow(WINDOW_NAME, 0, 0)
    cv2.setWindowTitle(WINDOW_NAME, 'YOLO')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", type=str, default="data/samples", help="path to dataset")
    parser.add_argument("--model_def", type=str, default="config/yolov3-tiny-softball.cfg", help="path to model definition file")
    parser.add_argument("--weights_path", type=str, default="checkpoints/yolov3_ckpt_285.pth", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="data/softball.names", help="path to class label file")
    parser.add_argument("--conf_thres", type=float, default=0.9, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.3, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument('--width', default=1280, type=int)
    parser.add_argument('--height', default=720, type=int)
    parser.add_argument('--screen', default=False, action='store_true')
    parser.add_argument('-v', default='test.mp4', type=str)
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

    if opt.screen:
        data_loader = DataPrefetcher(ScreenCapture())
    else:
        data_loader = DataPrefetcher(VideoLoader(opt.v))

    print("\nPerforming object detection:")

    open_window(opt.width, opt.height)
    full_scrn = False

    cv2.waitKey(10)

    for batch_i, (image, input_imgs) in enumerate(data_loader):
        if cv2.getWindowProperty(WINDOW_NAME, 0) < 0:
            # Check to see if the user has closed the window
            # If yes, terminate the program
            break

        start_time = time.time()

        # Get detections
        with torch.no_grad():
            detections = model(input_imgs)
            detections = non_max_suppression(detections, opt.conf_thres, opt.nms_thres)

        # Log progress
        print("\t+ Batch %d, Inference Time: %.2fms" % (batch_i, (time.time() - start_time) * 1000))

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
    
        speed_text = f"{1.0/(time.time() - start_time):.2f} fps"
        (text_width, text_height), baseline = cv2.getTextSize(speed_text, cv2.FONT_HERSHEY_PLAIN, 1, 2)
        cv2.putText(image, speed_text, (10, 2 * text_height + baseline), 
            cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255), 2)
        
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imshow(WINDOW_NAME, image)
 
        # Press Q on keyboard to stop recording
        key = cv2.waitKey(1)
        if key == 27 or key == ord('q'):
            break
        elif key == ord('F') or key == ord('f'): # toggle fullscreen
            full_scrn = not full_scrn
            cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN,
                                    cv2.WINDOW_FULLSCREEN if full_scrn else cv2.WINDOW_NORMAL)

    cv2.destroyAllWindows()
