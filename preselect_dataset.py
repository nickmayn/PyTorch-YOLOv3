import argparse
import math
import numbers
import os
import random

import cv2
import numpy as np
import torch
from PIL import Image, ImageFilter
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm


class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """
    def __init__(self, channels, kernel_size, sigma, dim=2):
        super(GaussianSmoothing, self).__init__()
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                      torch.exp(-((mgrid - mean) / (2 * std)) ** 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
            )

    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        return self.conv(input, weight=self.weight, groups=self.groups)

ball_color = {
    '2019-04-11 vs JMU Gm1 Batter.mp4': np.array([170, 190, 131]),
    'Duke Game 1 Center Field.mp4': np.array([198, 244, 110]),
    'Duke Game 2 Center Field.mp4': np.array([209, 233, 135]),
    'UVA 1.mp4': np.array([201, 210, 73]),
    'UVA 3.mp4': np.array([172, 177, 63]),
    'UVA 4-12-19 CF.mp4': np.array([185, 226, 102]),
    'UVA2.mp4': np.array([205, 212, 92]),
    'UVA4-13-19Game2.mp4': np.array([192, 224, 127]),
}

crop_size = {
    '2019-04-11 vs JMU Gm1 Batter.mp4': (1000, 600),
    'UVA 4-12-19 CF.mp4': (1000, 600),
    'UVA4-13-19Game2.mp4': (1000, 600),
}

class VideoLoader:
    def __init__(self, path):
        self.cap = cv2.VideoCapture(path)
        self.frame = 0
        if not self.cap.isOpened():
            raise RuntimeError()

    def __next__(self):
        if not self.cap.isOpened():
            raise StopIteration()

        ret, image = self.cap.read()
        if not ret:
            self.cap.release()
            raise StopIteration()
            
        idx = self.frame
        self.frame += 1

        return idx, cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # return idx, image

    def __iter__(self):
        return self

pick_rate = 0.1

blur_filter = GaussianSmoothing(3, kernel_size=3, sigma=1)
blur_filter.to(torch.device('cuda'))
blur_filter.eval()

file_filter = lambda fn: any(fn.endswith(ext) for ext in [".mp4"])
filenames = [x for x in os.listdir('.') if file_filter(x)]

for f in filenames:
    name = os.path.splitext(f)[0]
    print(name)

    if not os.path.exists(name):
        os.mkdir(name)

    loader = VideoLoader(f)

    crop_w, crop_h = 0, 0
    if f in crop_size:
        crop_w, crop_h = crop_size[f]
        print(f'Crop {f} with {crop_w, crop_h}')

    c = torch.from_numpy(ball_color[f]).cuda()

    saved = 300
    with tqdm(total=saved) as pbar:
        for i, frame in loader:
            if crop_w > 0 or crop_h > 0:
                frame = frame[:crop_h, :crop_w]

            x = torch.from_numpy(frame).cuda()

            x = x.permute(2, 0, 1) # HWC -> CHW
            blur = blur_filter(x.float().unsqueeze(0)).squeeze(0)
            blur = blur.permute(1, 2, 0) # CHW -> HWC

            diff = torch.abs(blur - c).sum(axis=2).min().cpu().item()
            if diff > 12:
                continue

            if random.random() > pick_rate:
                continue

            Image.fromarray(frame).save(os.path.join(name, f'{name}_{i}.png'))
            saved -= 1

            pbar.update()

            if saved <= 0:
                break
