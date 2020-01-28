from __future__ import division

from models import *
from utils.logger import *
from utils.utils import *
from utils.vdatasets import ListDataset
from utils.validset import VaildDataset
from utils.parse_config import *

from terminaltables import AsciiTable

import os
import sys
import time
import datetime
import argparse

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim

from utils.prefetcher import DataPrefetcher

from prefetch_generator import BackgroundGenerator

class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__(), max_prefetch=1)

def evaluate(model, path, iou_thres, conf_thres, nms_thres, img_size, batch_size):
    model.eval()

    # Get dataloader
    dataset = VaildDataset(path, img_size=img_size, augment=True, multiscale=False)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=1, collate_fn=dataset.collate_fn
    )

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    labels = []
    sample_metrics = []  # List of tuples (TP, confs, pred)
    for batch_i, (_, imgs, targets) in enumerate(tqdm.tqdm(dataloader, desc="Detecting objects")):

        # Extract labels
        labels += targets[:, 1].tolist()
        # Rescale target
        targets[:, 2:] = xywh2xyxy(targets[:, 2:])
        targets[:, 2:] *= img_size

        imgs = Variable(imgs.type(Tensor), requires_grad=False)

        with torch.no_grad():
            outputs = model(imgs)
            outputs = non_max_suppression(outputs, conf_thres=conf_thres, nms_thres=nms_thres)

        sample_metrics += get_batch_statistics(outputs, targets, iou_threshold=iou_thres)

    if len(sample_metrics) == 0:
        return np.array([0.0]), np.array([0.0]), np.array([0.0]), np.array([0.0]), []

    # Concatenate sample statistics
    true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
    precision, recall, AP, f1, ap_class = ap_per_class(true_positives, pred_scores, pred_labels, labels)

    return precision, recall, AP, f1, ap_class

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=300, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="size of each image batch")
    parser.add_argument("--gradient_accumulations", type=int, default=4, help="number of gradient accums before step")
    parser.add_argument("--model_def", type=str, default="config/yolov3-tiny-softball.cfg", help="path to model definition file")
    parser.add_argument("--data_config", type=str, default="config/softball.data", help="path to data config file")
    parser.add_argument("--pretrained_weights", type=str, help="if specified starts from checkpoint model")
    parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--checkpoint_interval", type=int, default=1, help="interval between saving model weights")
    parser.add_argument("--evaluation_interval", type=int, default=1, help="interval evaluations on validation set")
    parser.add_argument("--compute_map", default=False, help="if True computes mAP every tenth batch")
    parser.add_argument("--multiscale_training", default=True, help="allow for multi-scale training")
    parser.add_argument("--n_GPUs", type=int, default=1)
    opt = parser.parse_args()
    print(opt)

    logger = Logger("logs")

    logfile = open('val.log', 'w')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs("output", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    # Get data configuration
    data_config = parse_data_config(opt.data_config)
    train_path = data_config["train"]
    valid_path = data_config["valid"]
    class_names = load_classes(data_config["names"])

    # Initiate model
    model = Darknet(opt.model_def).to(device)
    model.apply(weights_init_normal)

    # If specified we start from checkpoint
    if opt.pretrained_weights:
        if opt.pretrained_weights.endswith(".pth"):
            model.load_state_dict(torch.load(opt.pretrained_weights))
        else:
            model.load_darknet_weights(opt.pretrained_weights)

    # Get dataloader
    dataset = ListDataset(train_path, class_names, img_size=opt.img_size, augment=True)
    dataloader = DataLoaderX(
        dataset,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        collate_fn=dataset.collate_fn)

    # dataloader = DataPrefetcher(
    #     torch.utils.data.DataLoader(
    #         dataset,
    #         batch_size=opt.batch_size,
    #         shuffle=False,
    #         num_workers=opt.n_cpu,
    #         pin_memory=True,
    #         collate_fn=dataset.collate_fn),
    #     prepare=dataset.prepare
    # )

    optimizer = torch.optim.Adam(model.parameters())

    metrics = [
        "grid_size",
        "loss",
        "x",
        "y",
        "w",
        "h",
        "conf",
        "cls",
        "cls_acc",
        "recall50",
        "recall75",
        "precision",
        "conf_obj",
        "conf_noobj",
    ]

    batches_done = 0

    for epoch in range(opt.epochs):
        model.train()
    
        epoch_time = [0, 0]
        start_time = time.time()
        data_time = time.time()

        for batch_i, (_, imgs, targets) in enumerate(dataloader):
            load_time = time.time() - data_time
            model_time = time.time()

            imgs = imgs.to(device)
            targets = Variable(targets.to(device), requires_grad=False)

            if torch.cuda.is_available() and opt.n_GPUs > 1:
                loss, outputs = torch.nn.parallel.data_parallel(model, (imgs, targets), range(opt.n_GPUs))
            else:
                loss, outputs = model(imgs, targets)
            
            loss.backward()

            batches_done += 1
            if batches_done % opt.gradient_accumulations:
                # Accumulates gradient before each step
                optimizer.step()
                optimizer.zero_grad()

            # ----------------
            #   Log progress
            # ----------------

            log_str = "\n---- [Epoch %d/%d, Batch %d] ----\n" % (epoch, opt.epochs, batch_i)

            metric_table = [["Metrics", *[f"YOLO Layer {i}" for i in range(len(model.yolo_layers))]]]

            # Log metrics at each YOLO layer
            for i, metric in enumerate(metrics):
                formats = {m: "%.6f" for m in metrics}
                formats["grid_size"] = "%2d"
                formats["cls_acc"] = "%.2f%%"
                row_metrics = [formats[metric] % yolo.metrics.get(metric, 0) for yolo in model.yolo_layers]
                metric_table += [[metric, *row_metrics]]

                # Tensorboard logging
                tensorboard_log = []
                for j, yolo in enumerate(model.yolo_layers):
                    for name, metric in yolo.metrics.items():
                        if name != "grid_size":
                            tensorboard_log += [(f"{name}_{j+1}", metric)]
                tensorboard_log += [("loss", loss.item())]
                logger.list_of_scalars_summary(tensorboard_log, batches_done)

            log_str += AsciiTable(metric_table).table
            log_str += f"\nTotal loss {loss.item()}"

            train_time = time.time() - model_time
            log_str += f'\nModel time: {train_time:.2f}s'
            log_str += f'\nData time: {load_time:.2f}s'

            epoch_time[1] += load_time
            epoch_time[0] += train_time

            print(log_str)
            model.seen += imgs.size(0)
            data_time = time.time()

        total_time = time.time() - start_time
        print(f'Total Model time: {epoch_time[0]:.2f}s ({epoch_time[0] / total_time:.3f})')
        print(f'Total Data time: {epoch_time[1]:.2f}s ({epoch_time[1] / total_time:.3f})')
        print(f'Total time: {total_time:.2f}s ETA: {(opt.epochs - epoch) * total_time / 3600:.2f}h')

        if epoch % opt.evaluation_interval == 0:
            print("\n---- Evaluating Model ----")
            # Evaluate the model on the validation set
            precision, recall, AP, f1, ap_class = evaluate(
                model,
                path=valid_path,
                iou_thres=0.5,
                conf_thres=0.5,
                nms_thres=0.5,
                img_size=opt.img_size,
                batch_size=8,
            )
            evaluation_metrics = [
                ("val_precision", precision.mean()),
                ("val_recall", recall.mean()),
                ("val_mAP", AP.mean()),
                ("val_f1", f1.mean()),
            ]
            logger.list_of_scalars_summary(evaluation_metrics, epoch)

            # Print class APs and mAP
            ap_table = [["Index", "Class name", "AP"]]
            for i, c in enumerate(ap_class):
                ap_table += [[c, class_names[c], "%.5f" % AP[i]]]
            print(AsciiTable(ap_table).table)
            print(f"---- mAP {AP.mean()}")

            print(f"\n---- Evaluating Epoch:{epoch} ----", file=logfile)
            print(AsciiTable(ap_table).table, file=logfile)
            print(f"---- mAP {AP.mean()}", file=logfile)
            logfile.flush()

        if epoch % opt.checkpoint_interval == 0:
            torch.save(model.state_dict(), f"checkpoints/yolov3_ckpt_%d.pth" % epoch)
