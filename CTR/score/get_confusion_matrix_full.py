import numpy as np
import os, math, glob
import matplotlib.pyplot as plt
import pandas as pd
import torch, tqdm
import monai, cv2
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
from prettytable import PrettyTable
from torchvision import transforms
from score.evaluate import ConfusionMatrix

class TestDataset(BaseDataset):
    CLASSES = ["background", 'cancer']

    def __init__(self, GT_path, classes=None):

        with open(GT_path, "r") as f:
            self.GT_path = f.read().splitlines()
        self.class_values = [self.CLASSES.index(cls) for cls in classes]

    def __getitem__(self, i):

        Pre_mask = np.array(Image.open(self.GT_path[i].replace("image", "pre").replace(".jpg", ".png")))
        gt_mask = np.array(Image.open(self.GT_path[i].replace("image", "mask").replace(".jpg", ".png")))
        a_Pre_mask = Pre_mask.copy()
        a_gt_mask = gt_mask.copy()

        gt_masks = [(gt_mask == v) for v in self.class_values]
        gt_mask = np.stack(gt_masks, axis=-1)

        Pre_masks = [(Pre_mask == v) for v in self.class_values]
        Pre_mask = np.stack(Pre_masks, axis=-1)

        return Pre_mask.transpose(2, 0, 1), gt_mask.transpose(2, 0, 1), a_Pre_mask, a_gt_mask

    def __len__(self):
        return len(self.GT_path)


class PixMeter(object):
    def __init__(self, numClass):
        self.reset(numClass)
        self.numClass = numClass
    def reset(self, numClass):
        # tp, fp, tn, fn
        self.mx = torch.zeros((numClass, 4))
    def update(self, matrix):
        for i in range(self.numClass):
            self.mx[i][0] = self.mx[i][0] + matrix[0][i][0]
            self.mx[i][1] = self.mx[i][1] + matrix[0][i][1]
            self.mx[i][2] = self.mx[i][2] + matrix[0][i][2]
            self.mx[i][3] = self.mx[i][3] + matrix[0][i][3]


class HausdorffMeter(object):
    def __init__(self, classes):
        self.reset(classes)
        self.numClass = classes

    def reset(self, classes):
        self.hds = torch.zeros(classes)
        self.num = torch.zeros(classes)
        self.HDlist = [[], []]

    def update(self, matrix):
        for i in range(self.numClass):
            if math.isinf(matrix[0][i]) or math.isnan(matrix[0][i]):
                pass
            else:
                self.hds[i] = self.hds[i] + matrix[0][i]
                self.num[i] += 1


if __name__ == "__main__":

    CLASSES = ["background", 'cancer']
    GT_path = r"Data/TrainData/test.txt"
    batch_size = 1
    table_title = "model"
    valid_dataset = TestDataset(GT_path, classes=CLASSES)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    result_metrics = PixMeter(len(CLASSES))
    result_HD = HausdorffMeter(len(CLASSES))
    cm = ConfusionMatrix(len(CLASSES), CLASSES)
    for pre, gt, a_Pre_mask, a_gt_mask in tqdm.tqdm(valid_loader):

        #print(pre.shape, gt.shape)
        matrix = monai.metrics.get_confusion_matrix(pre, gt, include_background=True)
        cm.update(a_gt_mask[0], a_Pre_mask[0])

        # HDs = monai.metrics.compute_hausdorff_distance(pre, gt, include_background=True)
        # result_HD.update(HDs)
        result_metrics.update(matrix)
    score_table = cm.get_score()
    print("aaa", score_table)

    rm = result_metrics.mx.numpy()
    hd = result_HD.hds.numpy()
    hd_num = result_HD.num.numpy()

    tb = PrettyTable()
    tb.title = table_title
    tb.field_names = ["", "Dice", "Iou", "Sensitivity", "PPV"]

    for cls in range(len(CLASSES)):
        # result_metrics: tp-0, fp-1, tn-2, fn-3
        # Dice = 2TP/(FP + FN + 2TP)
        dice = 2.0 * rm[cls][0] / (rm[cls][1] + rm[cls][3] + 2.0 * rm[cls][0])
        # Iou = TP/(FP + FN + TP)
        Iou = rm[cls][0] / (rm[cls][1] + rm[cls][3] + rm[cls][0])
        # Sensitivity = TP/(TP+FN)
        Sensitivity = rm[cls][0] / (rm[cls][0] + rm[cls][3])
        # PPV = TP/(TP+FP)
        PPV = rm[cls][0] / (rm[cls][0] + rm[cls][1])
        # HD95 = (hd[cls] / hd_num[cls]) * 0.95
        tb.add_row([CLASSES[cls], "%.4f" % dice, "%.4f" % Iou, "%.4f" % Sensitivity, "%.4f" % PPV])
    print(tb)

