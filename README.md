ml_lab4_2 report
===============
**310553003張皓鈞**

---

**Table of Contents**
- Introduction
- Experiment Stages
  - Details of my model (Resnet) 
  - Details of my Dataloader
  - Describing my evaluation matrix through the confusion matrix
- Experimental Results
  - Highest testing accuracy
    - screenshot
    - Stuff I want to present
  - Comparison Figures
    - Plotting Comparison Figures (Res 18/50, with/without pretraining)
- Disscussion

---

## Introduction

  Lab 4-2 is about diabetic retinopathy analysis, using a customized dataloader by us to load data into a resnet network, which is implemented by Pytorch framework.
After training the networks (details provided later), evaluation said network by confusion matrices.
Will provide some brief introduction on the following topic: Diabetic Retinopathy & Dataset (input/output), Resnet Network & Transfer Learning (method), Pytorch and Dataloader (implementation)
### Diabetic Retinopathy & Dataset (input/output)
Diabetic retinopathy is a complication of diabetes, caused by high blood sugar levels damaging the back of the eye (retina). It can cause blindness if left undiagnosed and untreated.

https://www.nhs.uk/conditions/diabetic-retinopathy/#:~:text=Diabetic%20retinopathy%20is%20a%20complication,it%20could%20threaten%20your%20sight.

 large set of high-resolution retina images (jpeg) taken under a variety of imaging conditions. A left and right field is provided for every subject. Images are labeled with a subject id as well as either left or right (e.g. 1_left.jpeg is the left eye of patient id 1).

A clinician has rated the presence of diabetic retinopathy in each image on a scale of 0 to 4, according to the following scale:


|Label|Meaning|
  |:-:|:-:|
  |0|**`No DR`**|
  |1|**`Mild`**|
  |2|**`Moderate`**|
  |3| **`Severe`**|
  |4| **`Proliferative DR`**|


https://www.kaggle.com/c/diabetic-retinopathy-detection#description

### Resnet Network & Transfer Learning (method)
A residual neural network (ResNet) is an artificial neural network (ANN). Residual neural networks utilize skip connections, or shortcuts to jump over some layers.
There are two main reasons to add skip connections: to avoid the problem of vanishing gradients, or to mitigate the Degradation (accuracy saturation) problem; where adding more layers to a suitably deep model leads to higher training error. Resnet 50 and 18 will be compared in the following results.

We will also be comparing the effects of whether or not transfer learning is used. Transfer learning is a machine learning method where we reuse a pre-trained model as the starting point for a model on a new task. Focuses on storing knowledge gained while solving one problem and applying it to a different but related problem, like using network that classifies cats to another one that classifies trucks. 

https://en.wikipedia.org/wiki/Residual_neural_network
### Pytorch & Dataloader (implementation)
Code for processing data samples can get messy and hard to maintain; we ideally want our dataset code to be decoupled from our model training code for better readability and modularity. PyTorch provides two data primitives: `torch.utils.data.DataLoader` and `torch.utils.data.Dataset` that allow you to use pre-loaded datasets as well as your own data. 

Dataset stores the samples and their corresponding labels, and DataLoader wraps an iterable around the Dataset to enable easy access to the samples.

## Experiment Stages
Implementation details will also be included here
### Details of my model (Resnet)
```
from torchvision import models
import torch.nn as nn
model = models.resnet50(pretrained=True)
model.fc = nn.Sequential(nn.Linear(2048,256),
                         nn.ReLU(inplace=True),
                         nn.Linear(256,128),
                         nn.ReLU(inplace=True),
                         nn.Linear(128,64),
                         nn.ReLU(inplace=True),
                         nn.Linear(64,5),    
                    )
print(model)
model = model.to(device)
```
`model = models.resnet50(pretrained=True)` Downloads pretrained (pretrained=True) model pretrained on Imagenet dataset
`model.fc = nn.Sequential(nn.Linear(2048,256),...` reinitialized specific layers, replace the Final layer of pretrained resnet to fit output of 5 labels, (2048 will be 512 for resnet18)
`model = model.to(device)` move model to device (GPU)
![alt text](https://github.com/changb1/ml_lab4_2/blob/main/%E8%9E%A2%E5%B9%95%E6%93%B7%E5%8F%96%E7%95%AB%E9%9D%A2%202022-04-26%20161706.png "Res Architecture")
![alt text](https://github.com/changb1/ml_lab4_2/blob/main/%E8%9E%A2%E5%B9%95%E6%93%B7%E5%8F%96%E7%95%AB%E9%9D%A2%202022-04-26%20161849.png "Adjusted layers")

### Details of my Dataloader
### Describing my evaluation matrix through the confusion matrix
## Experimental Results
### Highest testing accuracy
#### Screenshots
#### Stuff I want to present
Some thoughts: No DR to Proliferative DR classification, rather than being independent classifier such as 'dog','sheep','cat', that is 


https://www.intellspot.com/data-types/
### Comparison Figures
#### Plotting Comparison Figures (Res 18/50, with/without pretraining)
## Disscussion
