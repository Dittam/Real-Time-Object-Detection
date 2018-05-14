# Real-time object detection MobileNet-SSD
Real-time object detection using a convolutional neural network. Implemented with the MobileNet-SSD model using TensoFlow API developed by Google.
<p align="center">
<img src="https://github.com/Dittam/Real-Time-Object-Detection/blob/master/screenshots/main.JPG" width="358" height="337">
</p>


## File Info

|          File Name          |                                        Description                                       |
|:--------------------:|:------------------------------------------------------------------------------------:|
| extraFunctions.py           | Script containing helper functions for video stream, fps and multithreading TensorFlow graphs                                                                       |
| objectDetection.py       | Script that runs object detection using webcam input. Added multithreading support to split work btwn CPU & GPU                                              |
| objectDetectionCPU.py       | Script that runs object detection using webcam input. Run this if you dont have a GPU or TensorFlow-GPU isn't installed. WARNING: may be slow due to lack of multithreading &  GPU support               |
| TensorFlowDetectionAPI folder | Contains the mobileNet-SSD model developed by Google. Retrained on the COCO 2017 dataset                        |


## MobileNet-SSD:
MobileNet is a convolutional neural network architecture designed by Google for computationally inexpensive image classification and features extraction. However MobileNet alone cannot localize objects within an image.

SSD stands for single shot multibox detection. An algorithm designed to localize objects via bounding boxes within an image. SSD draws hand chosen default bounding boxes known as 'priors' onto the image then uses several metrics such as intersection-over-union to determine the best bounding boxes encompassing objects, drastically reducing computational cost compared to the brute force sliding window approach. The boxes are then classified using mobileNet. 

MobileNet-SSD removes the last fully connected layers in Mobilenet, keeping only the features extraction layers. The SSD algorithm is then added to the end of features extraction layers. Allowing the network to classify and localize objects in a single pass. 

### MobileNet-SSD Architecture
<p align="left">
<img src="https://github.com/Dittam/Real-Time-Object-Detection/blob/master/screenshots/architecture.png" width="500" height="209">
</p> 



## Dependencies:
* Python 3+
* TensorFlow (gpu version optional)
* Matplotlib
* Numpy
* openCV
* Cython
