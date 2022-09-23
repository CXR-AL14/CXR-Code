import glob

import cv2
import numpy as np
import torch
from PIL import Image

from skimage import measure, draw
from tqdm import tqdm
import AttentionUnet
import os

def del_mask(mask):
    kernel = np.ones((5, 5), np.uint8)
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
    kernel = np.ones((5, 5), np.uint8)
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
    pre = pre.detach().cpu().numpy()[0].transpose([1, 2, 0]) * 1
    heart = pre[:, :, 1] * 255
    lung = pre[:, :, 2] * 255
    lung = lung.astype("uint8")
    lung = cv2.resize(lung, size)
    lung = del_mask(lung)
    lung[lung > 0] = 255
    heart = heart.astype("uint8")
    heart = cv2.resize(heart, size)
    heart = del_mask2(heart)
    heart[heart > 0] = 255

    return lung, heart


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


if __name__ == '__main__':
    CTR = AttentionUnet.AttU_Net(3, 3)
    state_dictunet = torch.load(r'418.pth', map_location=torch.device("cuda:0"))
    CTR.load_state_dict(state_dictunet)
    device = torch.device("cuda:1")
    CTR.to(device)
    # image_paths = glob.glob("Test_Image/*.jpg")


    images = open("/home/Yang/Project/normals.txt").read().strip().split()

    with open("ctrs.txt", "w") as f:
        for image2 in tqdm(images):
            image_path = os.path.join("/home/Yang/Project/yolox/Normal",image2 )
            image = cv2.imread(image_path, 1)
            mask1, mask2 = calculate_ctr(image, CTR, device=device)
            mask1 = del_mask(mask1)
            mask2 = del_mask2(mask2)
            ctr = cute_ctr(mask1, mask2)
            font = cv2.FONT_HERSHEY_SIMPLEX
            contours, hierachy = cv2.findContours(mask1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            contours2, hierachy2 = cv2.findContours(mask2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            result = cv2.drawContours(image, contours, -1, (0, 0, 255), 20)
            result = cv2.drawContours(result, contours2, -1, (0, 255, 255), 20)
            ratio = "none"


            if ctr !=None:
                ratio = ctr["ratio"]
                line = "%s,%s\n" % (image2, ratio)
                f.write(line)
        
        f.close()
        # if ctr != None:
        #     y = [ctr["lung_y"], ctr["heart_y_end_right"], ctr["heart_y_begin_left"]]
        #     y.sort()
        #     cv2.line(result, (int((ctr["lung_x_end"] + ctr["lung_x_begin"]) / 2), int(y[0]) - 100),
        #              (int((ctr["lung_x_end"] + ctr["lung_x_begin"]) / 2), int(y[-1] + 100)), (255, 0, 0), 20)
        #     cv2.line(result, (int(ctr["lung_x_begin"]), int(ctr["lung_y"])),
        #              (int(ctr["lung_x_end"]), int(ctr["lung_y"])), (0, 255, 0), 20)
        #     cv2.line(result, (int(ctr["heart_x_begin_left"]), int(ctr["heart_y_begin_left"])),
        #              (int((ctr["lung_x_end"] + ctr["lung_x_begin"]) / 2), int(ctr["heart_y_begin_left"])), (0, 0, 0),
        #              20)
        #     cv2.line(result, (int((ctr["lung_x_end"] + ctr["lung_x_begin"]) / 2), int(ctr["heart_y_end_right"])),
        #              (int(ctr["heart_x_end_right"]), int(ctr["heart_y_end_right"])), (0, 0, 0), 20)
        #     cv2.putText(image, "CTR:%.4f" % round(ctr["ratio"], 4), (100, 200), font, 5, (0, 255, 0), 10)
        #
        # result = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        # # result.show()
        # result.save(image_path.replace("Test_Image", "418_2"))
