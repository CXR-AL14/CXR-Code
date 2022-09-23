import torch, cv2, glob, os, torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
class GetLoadData(Dataset):
    def __init__(self, train_table_path, img_size, mode):
        self.imagelist = GetLoadData.getPathList(train_table_path, mode)
        self.img_size = img_size

    def __len__(self):
        return len(self.imagelist)

    def getPathList(tablePath, mode):
        table = pd.read_csv(tablePath)[mode]
        imgPathList = []
        for p in table:
            for i in glob.glob(p + "/*"):
                imgPathList.append(i)
        return imgPathList

    def __getitem__(self, idx):
        imagePath = self.imagelist[idx]
        maskPath  = imagePath.replace("image", "mask")
        img = cv2.resize(cv2.imread(imagePath), self.img_size)/255
        mask = cv2.resize(cv2.imread(maskPath), self.img_size)

        if mask.shape[2] != 1:
            mask = mask[:, :, 0:1]
        if mask.max() > 1:
            mask[mask > 0] = 1

        img = img.transpose((2, 0, 1))
        mask = mask.transpose((2, 0, 1))

        return img, mask


class TestDataLoad(Dataset):
    def __init__(self, dataPath, imageSize, mod):
        with open(dataPath, "r") as f:
            self.imagePath = f.read().splitlines()
        self.imageSize = imageSize
        self.mod = mod
        self.trans = transforms.Compose([
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.1),
    		transforms.ToTensor()])

    def __getitem__(self, idx):
        image_path = self.imagePath[idx]
        mask_path = image_path.replace("image", "mask").replace(".jpg", ".png")
        #print(image_path, mask_path)
        #img = cv2.resize(cv2.imread(image_path), self.imageSize)/255


        # mask = cv2.resize(cv2.imread(mask_path), self.imageSize)
        mask = np.array(Image.open(mask_path))
        # print(np.max(mask))
        if self.mod == "train":
            img = Image.open(image_path).convert('RGB').resize(self.imageSize)
            img = self.trans(img)
        else:
            img = cv2.resize(cv2.imread(image_path), self.imageSize) / 255
            img = img.transpose((2, 0, 1))
        #if mask.shape[2] != 1:
        #    mask = mask[:, :, 0:1]

        #img = img.transpose((2, 0, 1))
        #mask = mask.transpose((2, 0, 1))  #PraNet_train_V2
        #mask[mask > 0] = 1
        #mask = mask[:, :, 0]
        return img, mask


    def __len__(self):
        return len(self.imagePath)

class ZDM_Lung_Heart_DataLoad(Dataset):
    def __init__(self, dataPath, imageSize):
        self.TestPath = glob.glob(dataPath + '/*')
        self.imageSize = imageSize

    def __getitem__(self, idx):
        image_path = self.TestPath[idx]
        img = cv2.resize(cv2.imread(image_path), self.imageSize)/255
        img = img.transpose((2, 0, 1))
        return img, image_path

    def __len__(self):
        return len(self.TestPath)


class PredictDataLoad(Dataset):
    def __init__(self, dataPath, imageSize):
        self.TestPath = glob.glob(os.path.join(dataPath, 'image_val/*.png'))
        self.imageSize = imageSize

    def __getitem__(self, idx):
        image_path = self.TestPath[idx]
        img = cv2.resize(cv2.imread(image_path), self.imageSize)/255
        img = img.transpose((2, 0, 1))
        return img, image_path


    def __len__(self):
        return len(self.TestPath)

if __name__ == '__main__':
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    batch_size = 1
    imagesize = (128, 128)
    train_path = r'train_table.csv'
    val_path = r'val_table.csv'
    test_Path = r"Test"

    train_loader = GetLoadData(train_path, imagesize, mode="train")
    train_loader = DataLoader(train_loader,  batch_size=batch_size, shuffle=True, pin_memory=True)

    val_loader = GetLoadData(val_path, imagesize, mode="val")
    val_loader = DataLoader(val_loader, batch_size=batch_size, shuffle=True, pin_memory=True)

    test_loader = TestDataLoad(test_Path, imagesize)
    test_loader = DataLoader(test_loader, batch_size=batch_size, shuffle=True, pin_memory=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for image, label in test_loader:

        image = image.to(device=device, dtype=torch.float32).cpu().numpy()[0].transpose((1, 2, 0))   # transpose是numpy的对象
        label = label.to(device=device, dtype=torch.float32).cpu().numpy()[0].transpose((1, 2, 0))
        print(image.max())
        ax1 = plt.subplot(1, 2, 1)
        plt.sca(ax1)
        plt.imshow(image)

        ax2 = plt.subplot(1, 2, 2)
        plt.sca(ax2)
        plt.imshow(label[:, :, 0])

        plt.show()