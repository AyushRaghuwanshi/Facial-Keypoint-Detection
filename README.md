# Computer Vision Nanodegree
## Project:Facial Keypoint Detection
In this project, we build a facial keypoint detection system that takes in any image with faces, and predicts the location of 68 distinguishing keypoints on each face.

Facial keypoints include points around the eyes, nose, and mouth on a face and are used in many applications. These applications include: facial tracking, facial pose recognition, facial filters, and emotion recognition.

Some examples of these keypoints are pictured below.

![](readme-image/image-description.png)

## Files
- Notebook 1: Loading and visualizing the facial keypoint data
- Notebook 2: Defining and Training a Convolutional Neural Network (CNN) to predict facial keypoints
- Notebook 3: Facial keypoint detection using haar cascades and the trained CNN
- Notebook 4: Fun filters and keypoints uses
- models.py: Define the neural network architectures
- data_load.py: Data transforms classes

## Model architecture
### Net(<br>
  (conv1): Conv2d(1, 64, kernel_size=(5, 5), stride=(1, 1))<br>
  (conv2): Conv2d(64, 128, kernel_size=(4, 4), stride=(2, 2))<br>
  (conv3): Conv2d(128, 256, kernel_size=(4, 4), stride=(2, 2))<br>
  (conv4): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1))<br>
  (pool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)<br>
  (dropout1): Dropout(p=0.1)<br>
  (dropout2): Dropout(p=0.2)<br>
  (dropout3): Dropout(p=0.3)<br>
  (dropout4): Dropout(p=0.4)<br>
  (dropout5): Dropout(p=0.4)<br>
  (dropout6): Dropout(p=0.4)<br>
  (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)<br>
  (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)<br>
  (bn3): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)<br>
  (bn4): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)<br>
  (bn5): BatchNorm1d(1600, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)<br>
  (bn6): BatchNorm1d(800, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)<br>
  (fc1): Linear(in_features=2048, out_features=1600, bias=True)<br>
  (fc2): Linear(in_features=1600, out_features=800, bias=True)<br>
  (fc3): Linear(in_features=800, out_features=136, bias=True)<br>
### )
