import glob
import os
import random
import sys
import math
import threading

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset

def pad_to_square(img, pad_value=0.5):
    c, h, w = img.shape
    dim_diff = abs(h - w)
    # (upper / left) padding and (lower / right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # Determine padding
    pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
    # Add padding
    img = F.pad(img, pad, "constant", value=pad_value)
    return img, pad

class ImageFolder(Dataset):
    def __init__(self, folder_path, img_size=416):
        self.files = sorted(glob.glob("%s/*.*" % folder_path))
        self.img_size = img_size

    def __getitem__(self, index):
        img_path = self.files[index % len(self.files)]
        # Extract image as PyTorch tensor
        img = transforms.ToTensor()(Image.open(img_path))
        # Pad to square resolution and resize
        img, _ = pad_to_square(img)
        return img_path, img

    def __len__(self):
        return len(self.files)


class ListDataset(torch.utils.data.IterableDataset):
    def __init__(self, list_path, img_size=416, augment=True, target_device='cuda'):
        with open(list_path, "r") as file:
            self.img_files = file.readlines()

        self.label_files = [
            path.replace("images", "labels").replace(".png", ".txt").replace(".jpg", ".txt")
            for path in self.img_files
        ]
        self.img_size = img_size
        self.max_objects = 100
        self.augment = augment

        self.target_device = target_device

        self.iter_start = 0
        self.iter = 0
        self.end = len(self.img_files)

    def preload(self):
        if self.iter >= self.end:
            self.next_data = None
            return

        idx = self.iter % len(self.img_files)
        img_path = self.img_files[idx].rstrip()
        label_path = self.label_files[idx].rstrip()

        boxes = None
        if os.path.exists(label_path):
            boxes = torch.from_numpy(np.loadtxt(label_path).reshape(-1, 5))

        # Load image, use uint8 to save time
        data = np.array(Image.open(img_path).convert('RGB')).copy()
        img = torch.from_numpy(data)

        with torch.cuda.stream(self.stream):
            img = img.cuda(non_blocking=True)
            img, target = self.prepare(img, boxes)
            
            if self.target_device == 'cuda':
                target = target.cuda(non_blocking=True).float()
            else:
                target = target.float()

        self.iter += 1
        self.next_data = img_path, img, target

    def __next__(self):
        if self.next_data is None:
            raise StopIteration()

        torch.cuda.current_stream().wait_stream(self.stream)
        
        img_path, img, target = self.next_data

        if img is not None:
            img.record_stream(torch.cuda.current_stream())
        if target is not None and self.target_device == 'cuda':
            target.record_stream(torch.cuda.current_stream())

        self.preload()
        return img_path, img, target

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  # single-process data loading, return the full iterator
            iter_start = 0
            iter_end = self.end
        else:  # in a worker process
            # split workload
            per_worker = int(math.ceil(self.end / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = worker_id * per_worker
            iter_end = min(iter_start + per_worker, self.end)
        
        self.stream = torch.cuda.Stream()
        
        self.iter_start = iter_start
        self.iter = iter_start
        self.end = iter_end

        self.preload()
        return self

    def prepare(self, img, boxes):
        # Handle images with less than three channels
        if len(img.shape) != 3:
            img = img.unsqueeze(0)
            img = img.expand((3, img.shape[1:]))

        img = img.permute(2, 0, 1) # HWC -> CHW

        _, h, w = img.shape
        h_factor, w_factor = (h, w)

        # Pad to square resolution
        img, pad = pad_to_square(img)

        if boxes is None:
            return img, None

        _, padded_h, padded_w = img.shape

        # Extract coordinates for unpadded + unscaled image
        x1 = w_factor * (boxes[:, 1] - boxes[:, 3] / 2)
        y1 = h_factor * (boxes[:, 2] - boxes[:, 4] / 2)
        x2 = w_factor * (boxes[:, 1] + boxes[:, 3] / 2)
        y2 = h_factor * (boxes[:, 2] + boxes[:, 4] / 2)
        # Adjust for added padding
        x1 += pad[0]
        y1 += pad[2]
        x2 += pad[1]
        y2 += pad[3]
        # Returns (x, y, w, h)
        boxes[:, 1] = ((x1 + x2) / 2) / padded_w
        boxes[:, 2] = ((y1 + y2) / 2) / padded_h
        boxes[:, 3] *= w_factor / padded_w
        boxes[:, 4] *= h_factor / padded_h

        targets = torch.zeros((len(boxes), 6))
        targets[:, 1:] = boxes

        return img, targets

    def augment_func(self, img, targets):
        def horisontal_flip(images, t):
            images = torch.flip(images, [-1])
            t[:, 2] = 1 - t[:, 2]
            return images, t

        # Apply augmentations
        if self.augment:
            if random.random() < 0.5:
                img, targets = horisontal_flip(img, targets)
            if random.random() < 0.5:
                noise = torch.randn_like(img) * 0.15
                img += noise

        return img, targets

    def collate_fn(self, batch):
        paths, imgs, targets = list(zip(*batch))
        # Remove empty placeholder targets
        targets = [boxes for boxes in targets if boxes is not None]
        # Add sample index to targets
        for i, boxes in enumerate(targets):
            boxes[:, 0] = i

        batch_imgs = []
        batch_targets = []
        for img, target in zip(imgs, targets):
            img = img.float()
            img *= 1/255.0

            # Resize to input size
            img = F.interpolate(img.unsqueeze(0), size=self.img_size, mode="nearest").squeeze(0)

            if self.augment:
                img, target = self.augment_func(img, target)

            batch_imgs.append(img)
            batch_targets.append(target)

        imgs = torch.stack(batch_imgs)
        targets = torch.cat(batch_targets, 0)
        return paths, imgs, targets
