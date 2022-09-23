import csv

import torch
import os

from PIL import ImageDraw

import AttentionUnet
import cv2
import numpy as np
import PIL.Image as Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import time


def del_mask(mask):
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=3)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    ss = []
    for contour in contours:
        s = np.array(list(contour.shape)).sum()
        ss.append(s)
    s = ss.copy()
    ss.sort(reverse=True)
    for s1 in ss[2:]:
        n = s.index(s1)
        cv2.fillPoly(mask, [contours[n]], (0, 0, 0))
    return mask


def del_mask2(mask):
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=3)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    ss = []
    for contour in contours:
        s = np.array(list(contour.shape)).sum()
        ss.append(s)
    s = ss.copy()
    ss.sort(reverse=True)
    for s1 in ss[1:]:
        n = s.index(s1)
        cv2.fillPoly(mask, [contours[n]], (0, 0, 0))
    return mask


def huawaijie(mask):
    o = mask
    kernel = np.ones((3, 3), np.uint8)
    o = cv2.morphologyEx(o, cv2.MORPH_CLOSE, kernel, iterations=3)
    o = cv2.morphologyEx(o, cv2.MORPH_OPEN, kernel, iterations=3)
    ret, thresh = cv2.threshold(o, 2, 255, cv2.THRESH_BINARY_INV)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    x1 = []
    y1 = []
    x2 = []
    y2 = []
    for c in contours:

        x, y, w, h = cv2.boundingRect(c)

        if x != 0 and y != 0 and w != o.shape[1] and h != o.shape[0]:

            cv2.rectangle(o, (x, y), (x + w, y + h), (0, 255, 0), 1)
            x1.append(x)
            y1.append(y)
            x2.append(x + w)
            y2.append(y + h)
    x11 = min(x1)
    y11 = min(y1)
    x22 = max(x2)
    y22 = max(y2)
    ro1 = o[:, x11 + 2].tolist()
    ro2 = o[:, x22 - 2].tolist()
    c1 = ro1.index(255)
    c2 = ro2.index(255)
    return x11, y11, x22, y22, c1, c2


def cute_ctr(lung, heart):
    ctr = {}
    o = lung
    c = np.sum(o / 255, axis=0)
    d = []
    for c1 in c:
        if c1 != 0:
            d.append(1)
        else:
            d.append(0)
    # print(len(d))

    mm = []
    for k in range(len(d) - 1):
        m = abs(d[k + 1] - d[k])
        if m == 1.0:
            mm.append(k + 1)
    h, w = o.shape
    lab2 = o.copy()
    mage = np.zeros(o.shape, np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    o = cv2.morphologyEx(lung, cv2.MORPH_CLOSE, kernel, iterations=3)
    try:
        gg = mm[1] + 10
        # print("mm", mm)
        mage[0:h, 0:gg] = o[0:h, 0:gg]
        # print(h, w)
        o = o[0:h, 0:gg]
        lab = cv2.Laplacian(mage, cv2.CV_64F, ksize=3)
        lab = cv2.convertScaleAbs(lab)

        lab2 = cv2.Laplacian(lab2, cv2.CV_64F, ksize=3)
        lab2 = cv2.convertScaleAbs(lab2)
        lab2[lab2 > 0] = 255
        c = np.sum(o / 255, axis=1)
        d = []
        for c1 in c:
            # print(c1)
            if c1 != 0:
                d.append(1)
            else:
                d.append(0)
        nn = []
        for k in range(len(d) - 1):
            m = abs(d[k + 1] - d[k])
            if m == 1.0:
                nn.append(k + 1)
        # print("nn", nn)
        jj = int(nn[0] + (nn[1] - nn[0]) * 0.5)
        jj2 = nn[1] - 50
        # print(jj, jj2)
        lab[0:  int(nn[0] + (nn[1] - nn[0]) * 0.5), 0:w] = np.zeros((int(nn[0] + (nn[1] - nn[0]) * 0.5), w),
                                                                    np.uint8)
        lab[nn[1] - 50:h, 0:w] = np.zeros((h - nn[1] + 50, w), np.uint8)
        mm2 = np.sum(lab / 255, axis=1).tolist()
        # print(mm2)
        ggg = []
        for i in range(len(mm2)):
            if i < len(mm2) - 1:
                ggg.append(mm2[i] - mm2[i + 1])
        ggg = np.array(ggg)
        ggg = np.maximum(ggg, -ggg)
        gggg = list(set(ggg.tolist()))
        gggg.sort(reverse=True)
        gggg2 = [i for i in gggg if i >= 2]
        # print(gggg)
        xxx = list(ggg.copy())
        if len(gggg2) < 3:
            max1 = xxx.index(gggg2[0])
        else:
            maxs = gggg2[:3]
            max1 = xxx.index(maxs[0])
            max2 = xxx.index(maxs[1])
            max3 = xxx.index(maxs[2])
            max1 = np.min(np.array([max1, max2, max3]))
        if max1 - jj < 20 or jj2 - max1 < 20:
            d = []
            for u in range(jj, jj2 - 1):
                row1 = lab[u, :].tolist()
                row2 = lab[u + 1, :].tolist()
                row_1 = [i for i, x in enumerate(row1) if x == 255]
                row_2 = [i for i, x in enumerate(row2) if x == 255]
                # print(u, abs((row_2[-1] - row_2[0]) - (row_1[-1] - row_1[0])))
                d.append(abs((row_2[-1] - row_2[0]) - (row_1[-1] - row_1[0])))
            max1 = d.index(np.array(list(d)).max()) + jj
        row = lab2[max1, :].tolist()
        row_3 = [i for i, x in enumerate(row) if x == 255]
        lung_x_begin = row_3[0]
        lung_x_end = row_3[-1]
        lung_y = max1
        center_x = int((mm[1] + mm[2]) / 2)
        x1, y1, x2, y2, c1, c2 = huawaijie(heart)
        if lung_x_end < x2:
            lung_x_end = mm[-1]
        heart_W = x2 - x1
        lung_W = lung_x_end - lung_x_begin
        ratio = round(heart_W / lung_W, 4)

        heart_x_begin_left = x1
        heart_y_begin_left = c1
        heart_x_end_right = x2
        heart_y_end_right = c2
        ctr["lung_x_begin"] = float(lung_x_begin)
        ctr["lung_x_end"] = float(lung_x_end)
        ctr["lung_y"] = float(lung_y)
        ctr["center_x"] = float(center_x)
        ctr["heart_x_begin_left"] = float(heart_x_begin_left)
        ctr["heart_y_begin_left"] = float(heart_y_begin_left)
        ctr["heart_x_end_right"] = float(heart_x_end_right)
        ctr["heart_y_end_right"] = float(heart_y_end_right)
        ctr["ratio"] = ratio
        return ctr
    except:
        return None


def cult_aortic(aortic):
    try:
        aortic1 = {}
        row_sum = np.sum(aortic / 255, axis=1)
        row_sum_list = row_sum.tolist()
        y = row_sum.max()
        row_index = row_sum_list.index(y)
        aortic = cv2.Laplacian(aortic, cv2.CV_64F, ksize=3)
        aortic = cv2.convertScaleAbs(aortic)
        cc = aortic[row_index, :]
        cc_list = []
        i = 0
        for c in cc:
            if c != 0:
                cc_list.append(i)
            i += 1
        aortic1["Aortic_x_begin"] = float(cc_list[0])
        aortic1["Aortic_x_end"] = float(cc_list[-1])
        aortic1["Aortic_y"] = float(row_index)
        return aortic1
    except:
        return None


def calculate_ctr(img, CTR, device):
    size = (img.shape[1], img.shape[0])
    imm = cv2.resize(img, (512, 512)) / 255
    imm = cv2.resize(imm, (512, 512)) / 255

    imm = imm.transpose((2, 0, 1))
    imm = imm[np.newaxis, :]
    imm = torch.tensor(imm)
    im = imm.to(device=device, dtype=torch.float32)
    pre = CTR(im)
    pre = pre >= 0.5
    pre = pre.detach().numpy()[0].transpose([1, 2, 0]) * 1

    heart = pre[:, :, 1] * 255
    lung = pre[:, :, 2] * 255
    aortic = pre[:, :, 3] * 255

    lung = lung.astype("uint8")
    lung = cv2.resize(lung, size)
    lung = del_mask(lung)
    lung[lung > 0] = 255
    heart = heart.astype("uint8")
    heart = cv2.resize(heart, size)
    heart = del_mask(heart)
    heart[heart > 0] = 255

    aortic = aortic.astype("uint8")
    aortic = cv2.resize(aortic, size)
    aortic = del_mask(aortic)
    aortic[aortic > 0] = 255

    ctr = cute_ctr(lung, heart)
    aor = cult_aortic(aortic)
    return ctr, aor


import glob

if __name__ == '__main__':
    import pandas as pd

    device = torch.device("cuda:0")
    CTR = AttentionUnet.AttU_Net(3, 3)
    state_dictunet = torch.load('66.pth', map_location=device)
    CTR.load_state_dict(state_dictunet)
    CTR.to(device)
    i = 0
    data = []
    print(1)
    for root, dirs, files in os.walk("/home/Yang/Project/CRT/sup"):
            files.sort()
            for im_name in tqdm(files):
                start = time.time()
                img_path = os.path.join(root, im_name)
                imagename = os.path.split(img_path)[-1]
                img = cv2.imread(img_path, 0)
                im5 = img.copy()
                img = np.repeat(img[..., np.newaxis], 3, 2)
                size = (img.shape[1], img.shape[0])
                imm = cv2.resize(img, (512, 512)) / 255
                imm = imm.transpose((2, 0, 1))
                imm = imm[np.newaxis, :]
                imm = torch.tensor(imm)
                im = imm.to(device=device, dtype=torch.float32)
                pre = CTR(im)
                pre = pre >= 0.5
                pre = pre.detach().cpu().numpy()[0].transpose([1, 2, 0]) * 1
                heart = pre[:, :, 1]*1
                lung = pre[:, :, 2] * 2
                mask = heart+lung
                mask[mask==3]=1

                mask = mask.astype("uint8")
                mask = cv2.resize(mask, size,interpolation=cv2.INTER_NEAREST)

                Image.fromarray(mask).convert("L").save(os.path.join(root.replace("sup","mask2"),im_name.replace(".jpg",".png")))



