## YOLOX：You Only Look Once X to train CXR-AL14 dataset
## Requirement
CentOS 7.6, 8*GTX 2080 Ti, Docker (PyTorch 1.10.1, CUDA 10.2 + CuDNN 8.0, Python 3.6.9)

## Training Steps
1. ##### Dataset preparation  

  This article uses VOC format for training. Before training, you need to make your own dataset.    
  Before training, put the label file in the Annotation under the CXR-AL14-2022 folder under the CXR-AL14 folder.   
  Before training, put the picture file in the JPEGImages under the CXR-AL14-2022 folder under the CXR-AL14 folder.   

2. ##### Dataset processing  

  After placing the dataset, we need to use voc_ Annotation.py Get 2022 for training_ train.txt and 2022_ val.txt.   
  Modify the parameters in voc_ annotation.py.  The first training can only modify classes_ path，classes_ path is used to point to the txt corresponding to the detection category.   
  When training your own dataset, you can create a cxr_ classes.txt, where you write the categories you need to distinguish. 
  model_ data/cxr_ The contents of the classs.txt file are:
```python
Atelectasis
Calcification
...
```
​        Modify voc_annotation.py in classes_path, to make it correspond to cxr_classs.txt, and run voc_ annotation.py.

3. ##### Start Network Training

  There are many training parameters, all in the train.py. The most important part is still the classes_path in the train.py.  
  classes_ path is used to point to the txt corresponding to the detection category, and is consistent with the annotation.py txt. 
  After modifying classes_path, you can run train.py to start training. After training multiple epochs, the weights will be generated in the logs folder.  

4. ##### Training result prediction

  Two files (yolo.py and predict.py) are required for training result prediction. Modify model_path and classes_path in yolo.py. 

  After modification, you can run predict.py to detect. After running, enter the image path to detect.

## Predict steps
1. ##### According to the above training steps.  

2. ##### In the yolo.py file, modify model_path and classes_path in the following sections to make them correspond to the trained file.

```python
_defaults = {

    "model_path"        : 'model_data/yolox_l.pth',
    
    "classes_path"      : 'model_data/xray_classes.txt',

    "input_shape"       : [1280, 1280],

    "phi"               : 'l',
 
    "confidence"        : 0.001,

    "nms_iou"           : 0.1,

    "letterbox_image"   : True,

    "cuda"              : True,
}
```
3. ##### Run predict.py and enter
```python
img/deom.jpg  
```
## Evaluation steps
1. ##### This paper uses VOC format for evaluation.  

2. ##### If you have run voc before training_ annotation. py file. The code will automatically divide the data set into training set, verification set and test set. If you want to modify the scale of the test set, you can modify the trainval_percentl in the voc_annotation.py.

3. ##### After voc_ annotation.py divides the test set, go to get_ map. py file modifying classes_ path，classes_ path is used to point to the txt corresponding to the detection category, which is the same as the txt during training. The evaluation data set must be modified.

4. ##### In the yolo.py file, modify model_path and classes_path.

5. ##### Run get_ map.py will get the evaluation results, which will be saved in the map_ out folder.

## Reference
https://github.com/bubbliiiing/yolox-pytorch
