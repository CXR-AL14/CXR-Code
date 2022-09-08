import os
from concurrent.futures.thread import ThreadPoolExecutor
import PIL
import pydicom
import SimpleITK as sitk
import numpy as np
import PIL.ImageOps


class Dicom:
    def __init__(self, filePath):
        dcm = pydicom.read_file(filePath, force=True)
        try:
            self.PatientName = dcm.PatientName
        except:
            self.PatientName = None
        try:
            self.PatientID = dcm.PatientID
        except:
            self.PatientID = None
        try:
            self.PatientBirthDate = dcm.PatientBirthDate
        except:
            self.PatientBirthDate = None
        try:
            if dcm.PatientSex == "M":
                self.PatientSex = "男"
            elif dcm.PatientSex == "F":
                self.PatientSex = "女"
        except:
            self.PatientSex = "未知"
        try:
            self.PatientAge = dcm.PatientAge
        except:
            self.PatientAge = -1
        try:
            self.StudyDate = dcm.StudyDate
            self.StudyTime = dcm.StudyTime.split(".")[0]
            self.StudyDateTime = self.StudyDate + self.StudyTime
        except:
            self.StudyDateTime = None
        try:
            self.ViewPosition = dcm.ViewPosition
        except:
            self.ViewPosition = "未知"
        try:
            self.ImagerPixelSpacing = dcm.PixelSpacing
        except:
            self.ImagerPixelSpacing = None
        try:
            self.WindowCenter = dcm.WindowCenter
        except:
            self.WindowCenter = 2024
        try:
            self.WindowWidth = dcm.WindowWidth
        except:
            self.WindowWidth = 1024
        try:
            img_data = dcm.PixelData
            self.img_data = np.frombuffer(img_data, np.uint16).reshape(dcm.Rows, dcm.Columns)



        except:
            self.img_data = None

    def transform_Xraydata(self, img_data, windowWidth, windowCenter, normal=False):
        """
        注意，这个函数的self.image一定得是float类型的，否则就无效！
        return: trucated image according to window center and window width
        """
        minWindow = float(windowCenter) - 0.5 * float(windowWidth)
        newimg = (img_data - minWindow) / float(windowWidth)
        newimg[newimg < 0] = 0
        newimg[newimg > 1] = 1
        if not normal:
            newimg = (newimg * 255).astype('uint8')
        return newimg

    def dcm2jpg(self, normal=False):
        try:
            minWindow = float(self.WindowCenter) - 0.5 * float(self.WindowWidth)
            newimg = (self.img_data - minWindow) / float(self.WindowWidth)
            newimg[newimg < 0] = 0
            newimg[newimg > 1] = 1
            if not normal:
                newimg = (newimg * 255).astype('uint8')
            return newimg
        except:
            max = self.img_data.max()
            newimg = (self.img_data / max) * 255
            return newimg


def tojpg(path):
    dcm = Dicom(path)
    if dcm.ViewPosition == "PA":
        img_temp = dcm.dcm2jpg()
        img = PIL.Image.fromarray(img_temp)
        img.save("G:/217/IMG/" + dcm.PatientID + "_" + dcm.StudyDateTime + ".jpg")


if __name__ == '__main__':
    import tqdm
    import glob
    from PIL import Image

    for image_path in glob.glob(r"G:\ExternalValidationSet\CTR\fd\inver/*.jpg"):
        image = Image.open(image_path)
        img = PIL.ImageOps.invert(image)
        img.save(image_path)
    exit(0)

    # for root, dirs, files in os.walk("G:\segment\Test_Dcm2"):
    #     for file in files:
    #         dcm = Dicom(os.path.join(root, file))
    #         print(dcm.ImagerPixelSpacing)
    # exit(0)

    dst_dir = r"G:\ExternalValidationSet\CTR\xs"
    for root, dirs, files in os.walk(r"G:\ExternalValidationSet\CTR\习水"):
        for file in tqdm.tqdm(files):
            dcm = Dicom(os.path.join(root, file))
            img_temp = dcm.dcm2jpg()
            img = PIL.Image.fromarray(img_temp)
            img = PIL.ImageOps.invert(img)
            try:
                img.save(os.path.join(dst_dir, file.replace("dcm", "jpg")))
            except:
                img.save(os.path.join(dst_dir, file.replace("DCM", "jpg")))
        exit(0)
        dcm = Dicom(
            r"G:\wai\fulin\2\2\ANKE1_1_1.2.156.112536.2.560.7050122168178.13907970428.22_20210705141844.DCM_0001.dcm")
        img_temp = dcm.dcm2jpg(normal=False)
        img = PIL.Image.fromarray(img_temp)
        # img = PIL.ImageOps.invert(img)
        img.show()
        print(dcm.__dict__)
        exit(0)
        i = 0
        for k in [29]:
            print(k)
            file_path = []
            # for root, dirs, files in os.walk("H:/dicom/%d" % k):
            for root, dirs, files in os.walk("G:/last/124"):
                for file in files:
                    if file == "IMG00000.DCM":
                        file_path.append(os.path.join(root, file))
            pool = ThreadPoolExecutor(20)
            print(file_path)
            for path in file_path:
                pool.submit(tojpg, path)
            pool.shutdown(True)

        #             #     i += 1
        #             #     print(i)
