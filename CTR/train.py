
import os
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from numpy import *
from torch import optim
import matplotlib.pyplot as plt
from monai.losses import DiceCELoss
from tqdm import tqdm
import torch.nn.functional as F
from torchvision import transforms as tfs
from torch.optim.lr_scheduler import LambdaLR
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, random_split
from torch.utils.data import DataLoader, random_split
#from dataload_2D import trainDataLoad, valDataLoad, TestDataLoad
from load2D import GetLoadData, TestDataLoad
import AttentionUnet
from Unit.dice_score import dice_coeff, multiclass_dice_coeff



def dice_coef(output, target):
    smooth = 1e-5
    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    intersection = (output * target).sum()
    return (2. * intersection + smooth)/(output.sum() + target.sum() + smooth)


def Train(model, device, lr=1e-4, epochs=5000):
    optimizer = optim.RMSprop(model.parameters(), lr=lr, weight_decay=1e-8, momentum=0.8)
    criterion = nn.MSELoss(size_average=None, reduce=None, reduction='mean')
    scheduler_1 = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    best_dice = 0

    for e in range(epochs):
        trainLoss = []
        valLoss = []
        # train_loader, val_loader =  DataFc()
        n_classes = 3
        batch_size = 4
        imagesize = (512, 512)
        train_path = r'Heart_Lung_train/train'
        val_path = r'Heart_Lung_train/val'


        t_loader = TestDataLoad(train_path, imagesize)
        train_loader = DataLoader(t_loader, batch_size=batch_size, shuffle=True, pin_memory=True)

        v_loader = TestDataLoad(val_path, imagesize)
        val_loader = DataLoader(v_loader, batch_size=batch_size, shuffle=True, pin_memory=True)

        model.train()
        for image, label in tqdm(train_loader):
            image = image.to(device=device, dtype=torch.float32)
            label = label.to(device=device, dtype=torch.long)

            optimizer.zero_grad()
            pred = model(image)

            label = F.one_hot(label, n_classes).permute(0, 3, 1, 2).float()


            loss = criterion(pred, label)
            trainLoss.append(loss.item())

            loss.backward()
            optimizer.step()
            new_lr = optimizer.param_groups[0]['lr']

        model.eval()
        vdice = 0
        with torch.no_grad():
            for vimgs, vlabel in tqdm(val_loader):
                vimage = vimgs.to(device=device, dtype=torch.float32)
                vlabel = vlabel.to(device=device, dtype=torch.long)

                vpred = model(vimage)
                dicevpred = F.one_hot(vpred.argmax(dim=1), n_classes).permute(0, 3, 1, 2).float() # 先用argmax，再转成OneHot
                vlabel = F.one_hot(vlabel, n_classes).permute(0, 3, 1, 2).float()   # 转成OneHot

                vdice += multiclass_dice_coeff(dicevpred[:, 1:, ...], vlabel[:, 1:, ...], reduce_batch_first=False)
                vloss = criterion(vpred, vlabel)
                # print(vimage.shape, vpred.shape, vlab el.shape)
                valLoss.append(vloss.item())


        v_loss = mean(valLoss)
        scheduler_1.step(v_loss)
        vmdice = vdice/len(val_loader)

        if vdice > best_dice:
            best_dice = vdice
        torch.save(model.state_dict(), 'network/Attention_gant_Unet_2d_dir/result/model-AttUnet-LungHerat-512-RMSprop-MSELoss-%.5f.pth'%vmdice)
        meanTr = mean(trainLoss)
        print("EPOCH%s" % e + "trainLoss:%.5f" % meanTr + " valLoss:%.5f" % v_loss + "  Lr：%f" % new_lr + " dice:%.4f" % vmdice)
        torch.cuda.empty_cache()

        resultsave_val = 'network/Attention_gant_Unet_2d_dir/result/512-dice_val_log.txt'
        with open(resultsave_val, "a") as file:
            file.write("valLoss:" + str(v_loss) + " " + "vdice:" + str(vdice) + "\n")



if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    if torch.cuda.is_available():
        print("GPU")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = AttentionUnet.AttU_Net(3, 3).cuda()
    Train(net, device)