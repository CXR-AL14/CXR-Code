import os

from tqdm import tqdm

from utils.utils_map2 import get_map, voc_ap
import math
import random

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET


def draw_ap(dfs, path, iou, hosptal=""):
    classnames = [
        "Atelectasis",  # 肺不张
        "Calcification",  # 钙化
        "Consolidation",  # 实变
        "Effusion",  # 积液
        "Emphysema",  # 肺气肿
        "Fibrosis",  # 纤维灶
        "Fracture",  # 骨折
        "Mass",  # 肿块
        "Nodule",  # 结节
        "Pleural thickening",  # 胸膜增厚
        "Pneumatosis",  # 积气
        "Pneumothorax",  # 气胸
        "Postoperative metal",  # 金属
        "Venipuncture"  # PICC
    ]
    colors = [["#000000"], ["#800000"], ["#FF0000"], ["#FA8072"], ["#FFFF00"], ["#8B4513"], ["#FFA500"], ["#808000"],
              ["#7CFC00"], ["#006400"], ["#00FA9A"], ["#00BFFF"], ["#4169E1"], ["#800080"]]

    plt.rc('font', family='Times New Roman', size=15)
    labels = []
    fig = plt.figure(figsize=(11, 6), dpi=250)
    ax1 = plt.subplot(111)
    ax1.spines['bottom'].set_linewidth('1.2')
    ax1.spines['top'].set_linewidth('1.2')
    ax1.spines['left'].set_linewidth('1.2')
    ax1.spines['right'].set_linewidth('1.2')
    data_confidence = []
    for name in classnames:
        data = dfs[name]
        data_r = data.copy()
        data_r.drop_duplicates(subset=["F1"], keep='first', inplace=True)
        temp = data[data["F1"] == data_r["F1"].max()].values.tolist()[0]
        data_confidence.append([name] + temp)
        data.drop_duplicates(subset=["Recall"], keep='first', inplace=True)
        data = data[data["Precision"] != 0]
        precision = data["Precision"].values.tolist()
        recall = data["Recall"].values.tolist()
        precision1 = precision.copy()
        recall1 = recall.copy()
        ap, mrec, mprec = voc_ap(recall, precision)
        label = '%s,AP=%0.3f' % (name, ap)
        lines = []
        labels.append(label)
        x = [0.0] + recall1 + [recall1[-1]]
        y = [precision1[0]] + precision1 + [0]
        ax1.plot(x, y, linestyle='-', lw=2,
                 color=colors[classnames.index(name)][0],
                 label=label, zorder=1,
                 alpha=1.0)
    plt.xlabel('Recall', fontdict={"size": 20})
    plt.ylabel('Precision', fontdict={"size": 20})

    ax1.set_xlim([0.0, 1.02])
    ax1.set_ylim([0, 1.02])
    # plt.legend(loc='best', frameon=True, ncol=1)
    plt.rc('legend', fontsize='medium')
    plt.legend(bbox_to_anchor=(1.01, 0), loc=3, borderaxespad=0, )
    plt.tick_params(labelsize=20, )
    plt.show()
    fig.savefig(os.path.join(path, "%0.1f_%s_Pr曲线.pdf" % (iou, hosptal)))
    df_conf = pd.DataFrame(data=data_confidence, columns=["name", "confidence", "recall", "precision", "F1"])
    df_conf.to_csv(os.path.join(path, "%0.1f_%s_confidence.csv" % (iou, hosptal)), index=False)

    return df_conf


def iou(box1, box2):
    in_h = min(box1[2], box2[2]) - max(box1[0], box2[0])
    in_w = min(box1[3], box2[3]) - max(box1[1], box2[1])
    inter = 0 if in_h < 0 or in_w < 0 else in_h * in_w
    union = (box1[2] - box1[0]) * (box1[3] - box1[1]) + \
            (box2[2] - box2[0]) * (box2[3] - box2[1]) - inter
    iou = inter / union
    return iou


def get_xml_info(xml_path, mode=True):
    root = ET.parse(xml_path).getroot()
    objs = []
    for obj in root.findall('object'):
        bndbox = obj.find('bndbox')
        obj_name = obj.find('name').text
        left = bndbox.find('xmin').text
        top = bndbox.find('ymin').text
        right = bndbox.find('xmax').text
        bottom = bndbox.find('ymax').text
        # confidence = float(obj.find('truncated').text)
        if mode:
            objs.append([obj_name, [int(top), int(left), int(bottom), int(right)]])

    return objs


def get_tp_fp_fn(gt_objs, pre_objs, classes, iou_):
    result = []
    for cls in classes:
        tp = 0
        gt_cls = []
        pre_cls = []
        for gt_obj in gt_objs:
            if gt_obj[0] == cls:
                gt_cls.append(gt_obj)
        for pre_obj in pre_objs:
            if pre_obj[0] == cls:
                pre_cls.append(pre_obj)
        pre = len(pre_cls)
        gt = len(gt_cls)
        for gt_cl in gt_cls:
            ious = []
            for pre_cl in pre_cls:
                ioua = iou(gt_cl[1], pre_cl[1])
                ious.append(ioua)
            if len(ious) != 0:
                iou_max = np.array(ious).max()
                if iou_max >= iou_:
                    index = ious.index(iou_max)
                    pre_cls.remove(pre_cls[index])
                    # print(pre_cls, iou_max)
                    tp += 1
        fp = pre - tp
        fn = gt - tp
        result.append(tp)
        result.append(fp)
        result.append(fn)
    return result


def wilson(a, b):
    Q1 = 2 * a + 3.84
    Q2 = 1.96 * math.sqrt(3.84 + ((4 * a * b) / (a + b)))
    Q3 = 2 * (a + b) + 7.68
    return np.round(np.array([(Q1 - Q2) / Q3, (Q1 + Q2) / Q3]), 3)


def get_xml_model_predict(xml_path, df):
    root = ET.parse(xml_path).getroot()
    objs = []
    for obj in root.findall('object'):
        bndbox = obj.find('bndbox')
        obj_name = obj.find('name').text
        left = bndbox.find('xmin').text
        top = bndbox.find('ymin').text
        right = bndbox.find('xmax').text
        bottom = bndbox.find('ymax').text
        confidence = float(obj.find('truncated').text)
        confidence_th = df[df["name"] == obj_name].values.tolist()[0][1]
        if confidence >= confidence_th:
            objs.append([obj_name, [int(top), int(left), int(bottom), int(right)]])
    return objs


def get_tpfpfn(conf_df, path, iou, hosptal=""):
    classes = [
        "Atelectasis",
        "Calcification",
        "Consolidation",
        "Effusion",
        "Emphysema",
        "Fibrosis",
        "Fracture",
        "Mass",
        "Nodule",
        "Pleural thickening",
        "Pneumatosis",
        "Pneumothorax",
        "Postoperative metal",
        "Venipuncture"
    ]

    pre_dir = path + "/xmls"
    gt_dir = r"C:\Users\yangy\Desktop\实验结果\Annotations\Annotations"
    sample = os.listdir(pre_dir)
    results = []
    for xml in tqdm(sample):
        pre_objs = get_xml_model_predict(os.path.join(pre_dir, xml), conf_df)
        gt_objs = get_xml_info(os.path.join(gt_dir, xml))
        result = get_tp_fp_fn(gt_objs, pre_objs, classes, iou_=iou)
        n = 0
        count = 0
        for re in result:
            if not n % 3 == 0:
                count = count + re
            n += 1
        results.append([xml.replace(".xml", "")] + result)
    columns = ["xml_name"]
    for cls in classes:
        columns.append(cls + "TP")
        columns.append(cls + "FP")
        columns.append(cls + "FN")
    df = pd.DataFrame(data=results, columns=columns)
    df.to_csv(os.path.join(path, "%0.1f_%s_tpfpfn.csv" % (iou, hosptal)))

    all_precision = []
    all_recall = []
    all_f1 = []
    precisions = []
    recalls = []
    f1s = []
    dic = {}
    for i in range(len(classes)):
        name1 = classes[i] + 'TP'
        name2 = classes[i] + 'FP'
        name3 = classes[i] + 'FN'

        TP = np.sum(df[name1].values.tolist())
        FP = np.sum(df[name2].values.tolist())
        FN = np.sum(df[name3].values.tolist())

        print(classes[i] + '(' + classes[i] + ')')

        Precision = TP / (TP + FP)
        [a, b] = wilson(TP, FP)
        c = "%.3f\n(%.3f-%.3f)" % (Precision, a, b)
        precisions.append(c)
        print("Precision：\t\t", c)

        Recall = TP / (TP + FN)
        [a, b] = wilson(TP, FN)
        c = "%.3f\n(%.3f-%.3f)" % (Recall, a, b)
        recalls.append(c)
        print("Recall：\t\t", c)

        F1 = 2 * (Precision * Recall) / (Precision + Recall)
        c = "%.3f" % F1
        f1s.append(c)
        print("F1_score：\t\t", round(F1, 3))

        print('---------------------------------------------')

        all_precision.append(Precision)
        all_recall.append(Recall)
        all_f1.append(F1)

    print('Mean_Precision:', round(np.mean(all_precision), 3))
    print('Mean_Recall:', round(np.mean(all_recall), 3))
    print('Mean_f1:', round(np.mean(all_f1), 3))
    dic["Classes"] = classes + ["Mean"]
    dic["Precision"] = precisions + [round(np.mean(all_precision), 3)]
    dic["Recall"] = recalls + [round(np.mean(all_recall), 3)]
    dic["F1"] = f1s + [round(np.mean(all_f1), 3)]
    dd = pd.DataFrame(data=dic)
    dd.to_excel(os.path.join(path, "%0.1f_result.xlsx" % (iou)), index=False)


def draw_ap_by_noe(dfs, iou, hosptal=""):
    plt.rc('font', family='Times New Roman', size=15)
    colors = ["#029e73", "#0c39b9", "#de8f05"]
    if iou == 0.1:
        color = colors[2]
    elif iou == 0.3:
        color = colors[1]
    else:
        color = colors[0]
    fig = plt.figure(figsize=(16, 16), dpi=300)
    names = [
        "Atelectasis",  # 肺不张
        "Calcification",  # 钙化
        "Consolidation",  # 实变
        "Effusion",  # 积液
        "Emphysema",  # 肺气肿
        "Fibrosis",  # 纤维灶
        "Fracture",  # 骨折
        "Mass",  # 肿块
        "Nodule",  # 结节
        "Pleural thickening",  # 胸膜增厚
        "Pneumatosis",  # 积气
        "Pneumothorax",  # 气胸
        "Postoperative metal",  # 金属
        "Venipuncture"  # PICC
    ]
    doctor_names = ["Junior1", "Junior2", "Junior3", "Senior1", "Senior2", "Senior3"]
    for name in names:
        plt.subplot(4, 4, names.index(name) + 1)
        data = dfs[name]
        data.drop_duplicates(subset=["Confidence"], keep='last', inplace=True)
        data_r = data.copy()
        data_r.drop_duplicates(subset=["F1"], keep='last', inplace=True)
        temp = data[data["F1"] == data_r["F1"].max()].values.tolist()[0]
        data.drop_duplicates(subset=["Recall"], keep='first', inplace=True)
        data = data[data["Precision"] != 0]
        precision = data["Precision"].values.tolist()
        recall = data["Recall"].values.tolist()
        precision1 = precision.copy()
        recall1 = recall.copy()
        ap, mrec, mprec = voc_ap(recall, precision)
        doctor_data = pd.read_csv(r"C:\Users\yangy\Desktop\实验结果\医生\reslut\%0.1f_doctors.csv" % iou, encoding="utf-8")
        l, m = 0, 0
        plt.plot([0.0] + recall1 + [recall1[-1]], [precision1[0]] + precision1 + [0], linestyle='-', lw=2,
                 color="royalblue",
                 label='AP=%0.3f' % (ap), zorder=1,
                 alpha=1.0)
        for doctor_name in doctor_names:
            doc_da = doctor_data[doctor_data["doctor_name"] == doctor_name]
            da = doc_da[doc_da["classes"] == name].values.tolist()[0]
            if doctor_name in ["Senior1", "Senior2", "Senior3"]:  # 高级医生
                l += 1
                plt.scatter(da[3], da[2], marker="*", alpha=1, zorder=2, s=100, edgecolors="black")
            elif doctor_name in ["Junior1", "Junior2", "Junior3"]:  # 初级
                m += 1
                plt.scatter(da[3], da[2], marker="^", alpha=1, s=100, edgecolors="black")

            plt.scatter(temp[-3], temp[-2], s=100, color="orange", marker="p", edgecolors="black", alpha=1,
                        zorder=2)  # 画点
        plt.title('%s' % name)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        axes = plt.gca()
        plt.legend(loc="lower left")
        axes.set_xlim([0.0, 1.05])
        axes.set_ylim([0.0, 1.05])

    plt.show()
    fig.savefig(os.path.join(path, "%0.1f_%s_Pr曲线2.pdf" % (iou, hosptal)))


def draw_ap_by_three(hosptal=""):
    plt.rc('font', family='Times New Roman', size=15)
    fig = plt.figure(figsize=(16, 16), dpi=300)
    names = [
        "Atelectasis",  # 肺不张
        "Calcification",  # 钙化
        "Consolidation",  # 实变
        "Effusion",  # 积液
        "Emphysema",  # 肺气肿
        "Fibrosis",  # 纤维灶
        "Fracture",  # 骨折
        "Mass",  # 肿块
        "Nodule",  # 结节
        "Pleural thickening",  # 胸膜增厚
        "Pneumatosis",  # 积气
        "Pneumothorax",  # 气胸
        "Postoperative metal",  # 金属
        "Venipuncture"  # PICC
    ]
    iouss = ["0.1", "0.3", "0.5"]
    colors = ["#029e73", "#0c39b9", "#de8f05"]
    for name in names:
        plt.subplot(4, 4, names.index(name) + 1)
        for iou_ in iouss:
            data = pd.read_csv(os.path.join(path, iou_, name + ".csv"))
            data.drop_duplicates(subset=["Confidence"], keep='last', inplace=True)
            data_r = data.copy()
            data_r.drop_duplicates(subset=["F1"], keep='last', inplace=True)
            temp = data[data["F1"] == data_r["F1"].max()].values.tolist()[0]
            data.drop_duplicates(subset=["Recall"], keep='first', inplace=True)
            data = data[data["Precision"] != 0]
            precision = data["Precision"].values.tolist()
            recall = data["Recall"].values.tolist()
            precision1 = precision.copy()
            recall1 = recall.copy()
            ap, mrec, mprec = voc_ap(recall, precision)
            plt.plot([0.0] + recall1 + [recall1[-1]], [precision1[0]] + precision1 + [0], linestyle='-', lw=2,
                     color=colors[iouss.index(iou_)],
                     label='IoU=%s' % (iou_), zorder=1,
                     alpha=1.0)
        plt.title('%s' % name)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        axes = plt.gca()
        plt.legend(loc="lower left")
        axes.set_xlim([0.0, 1.05])
        axes.set_ylim([0.0, 1.05])

    plt.show()
    fig.savefig(os.path.join(path, "%s_Pr曲线3.pdf" % (hosptal)))


if __name__ == '__main__':
    ious = [0.1, 0.2, 0.3, 0.4, 0.5]
    # ious = [0.1, 0.2, 0.3, 0.4, 0.5]
    # ious = [0.1,0.3,0.5]
    # hosptals = ['TEST', 'BN', 'FD', 'NC', 'XS', 'QZX']
    hosptals = ['TEST']
    # hosptals = ['last']
    for iou_ in ious:
        for hosptal in hosptals:
            path = r"C:\Users\yangy\Desktop\实验结果\中心\825\%s" % hosptal
            draw_ap_by_three(hosptal)
            map, dfs = get_map(iou_, True, path=path, hosptal=hosptal)
            conf_df = draw_ap(dfs, path, iou_, hosptal=hosptal)
            get_tpfpfn(conf_df, path, iou_, hosptal=hosptal)
            if hosptal == "TEST":
                draw_ap_by_noe(dfs, iou_, hosptal)
