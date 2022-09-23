import numpy as np
from prettytable import PrettyTable
from sklearn.metrics import confusion_matrix


class ConfusionMatrix(object):
    def __init__(self, num_classes: int, labels: list):
        self.matrix = np.zeros((num_classes, num_classes))
        self.num_classes = num_classes
        self.labels = labels
        self.score_table = None
        self.epoch_dice = None

    def update(self, gt, pre):
        # print(gt.shape, pre.shape)
        if len(gt.shape) != 2 or len(pre.shape) != 2:
            pass
        y_true, y_pred = gt.flatten(), pre.flatten()
        img_matrix = confusion_matrix(y_true, y_pred, labels=list(range(self.num_classes)))
        self.matrix = self.matrix + img_matrix
    def reset(self):
        self.matrix = np.zeros((self.num_classes, self.num_classes))

    def get_score(self):
        i = 1
        tp = self.matrix[i, i]
        fp = np.sum(self.matrix[:, i]) - tp
        fn = np.sum(self.matrix[i, :]) - tp
        tn = np.sum(self.matrix) - tp - fp - fn

        # Dice = 2TP/(FP + FN + 2TP)
        dice = 2.0 * tp / (fp + fn + 2.0 * tp)
        # Iou = TP/(FP + FN + TP)
        iou = tp / (fp + fn + tp)
        # Sensitivity = TP/(TP+FN)
        sensitivity = tp / (tp + fn)
        # PPV = TP/(TP+FP)
        ppv = tp / (tp + fp)
        return dice, iou, sensitivity, ppv

    def get_matrix(self):
        return self.matrix

    def get_epoch_dice(self):
        return self.epoch_dice / self.num_classes
