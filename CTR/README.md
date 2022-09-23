## CTR calculation algorithm
## Requirement
CentOS 7.6, 8*GTX 2080 Ti, Docker (PyTorch 1.10.1, CUDA 10.2 + CuDNN 8.0, Python 3.6.9)

## Train.py

 Run  train.py  to train a segmentation model of heart and double lungs.

## CTR.py

Run  CTR.py  to calculate cardiothoracic ratio.

The detailed steps are as follows:

Step 1: The closed morphology operation was applied to the mask to eliminate holes and burrs.

Step 2: By using the connected area labelling approach, the two largest connected areas of the lung and the largest connected area of the heart were preserved in the mask.

Step 3: The Canny operator was applied to the mask to obtain the single-pixel boundary between the lung and the heart.

Step 4: Traversing upwards from the lowest point of the right lung boundary using a number of horizontal lines, at least two intersections were encountered between each horizontal line and the boundary. The first two intersections were determined from left to right, and the distance between these two intersections was calculated. As the upwards traversal process proceeded, a number of difference values between two adjacent distances could be obtained. Note that the horizontal line that determined the maximum difference value can be regarded as a tangent line passing through the right diaphragm.

Step 5: The width of the chest (**L**) could be calculated by the two farthest intersections determined by the above tangent line and the boundary of both lungs.

Step 6: Find the left and right points that were farthest from the midline on the boundary of the heart, the distances from the above two points to the midline were calculated and summed (**L**1+**L**2).

Step 7: The CTR could be easily calculated according to the formula (**L**1+**L**2)/**L**.





