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

Resnet Architecture:
![alt text](https://github.com/changb1/ml_lab4_2/blob/main/%E8%9E%A2%E5%B9%95%E6%93%B7%E5%8F%96%E7%95%AB%E9%9D%A2%202022-04-26%20161706.png "Res Architecture")
Final layer Adjustments, seen by printing the model:
![alt text](https://github.com/changb1/ml_lab4_2/blob/main/%E8%9E%A2%E5%B9%95%E6%93%B7%E5%8F%96%E7%95%AB%E9%9D%A2%202022-04-26%20161849.png "Adjusted layers")

### Details of my Dataloader
```
import skimage.io as sk
import pandas as pd
from torch.utils import data
import numpy as np

from torchvision import transforms
from PIL import Image
image_transform = transforms.Compose([transforms.Resize([512,512]),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]) 
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #Use GPU if it's available or else use CPU.
print(device) #Prints the device we're using
print(torch.cuda.get_device_name(0))
def getData(mode):
    if mode == 'train':
        img = pd.read_csv('train_img.csv')
        label = pd.read_csv('train_label.csv')
        return np.squeeze(img.values), np.squeeze(label.values)
    else:
        img = pd.read_csv('test_img.csv')
        label = pd.read_csv('test_label.csv')
        return np.squeeze(img.values), np.squeeze(label.values)

class RetinopathyLoader(data.Dataset):
    def __init__(self, root,image_transform, mode):
        self.root = root
        self.img_name, self.label = getData(mode)
        self.image_transform = image_transform
        self.mode = mode
        print("> Found %d images..." % (len(self.img_name)))

    def __len__(self):
        return len(self.img_name)

    def __getitem__(self, index):
        path = self.root + '/'+ self.img_name[index] + '.jpeg'
        img = Image.open(f'{path}')
        gt_label = self.label[index]
        if self.image_transform :
            img = self.image_transform(img)
        return img, gt_label
```
 - RetinopathyLoader
   - init : intilize RetinopathyLoader 
     - root (string): Root path of the dataset. 
     - mode : Indicate procedure status(training or testing)
     - self.img_name (string list): String list that store all image names.
     - self.label (int or float list): Numerical list that store all ground truth label values.
   - len : return the size of dataset
   - getitem : 
     - `path = self.root + '/'+ self.img_name[index] + '.jpeg'`step1-1: Get the image path 
     - `img = Image.open(f'{path}')`step1-2: load image (used PIL Image here)
     - `gt_label = self.label[index]`step2: Get the ground truth label from self.label 
     - `if self.image_transform : img = self.image_transform(img)`step3: Transform the .jpeg rgb images during the training phase, details in image_transform sector of the report
     - `return img, gt_label`step4: Return processed image and label
   - code usage example for feeding training data, change mode to test for testing data
     - `train_set = RetinopathyLoader(root='./data',mode='train',image_transform=image_transform)`
     - `train_loader = data.DataLoader(dataset=train_set, batch_size=4, shuffle=True)`

 - getData : fetch img_name and label for RetinopathyLoader
 - image_transform

`from torchvision import transforms` using transforms from torchvision
```
image_transform = transforms.Compose([transforms.Resize([512,512]),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
```
  `Resize()`: resize image
  
  `ToTensor()`: Convert a `PIL Image`(PIL in this case) or `numpy.ndarray` to tensor
  
  `Normalize()` a tensor image with mean and standard deviation Given mean: `(mean[1],...,mean[n])` and std: `(std[1],..,std[n])` for n channels
example of a transformed image(test sample), plot results by pyplot:
```
import matplotlib.pyplot as plt
train_features, train_labels = next(iter(train_loader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")
img = train_features[0].squeeze()
label = train_labels[0]
im=transforms.ToPILImage()(img).convert("RGB")
display(im)
print(im)
print(im.size)
```
![alt text](https://github.com/changb1/ml_lab4_2/blob/main/%E8%9E%A2%E5%B9%95%E6%93%B7%E5%8F%96%E7%95%AB%E9%9D%A2%202022-04-26%20161706.png "Res Architecture")

### Describing my evaluation matrix through the confusion matrix
using seaborn heatmap to generate confusion matrix
```
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
nb_classes = 5
confusion_matrix = np.zeros((nb_classes, nb_classes))
with torch.no_grad():
    for i, (inputs, classes) in enumerate(test_loader):
        inputs = inputs.to(device)
        classes = classes.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        for t, p in zip(classes.view(-1), preds.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1

plt.figure(figsize=(15,10))

class_names = ['0','1','2','3','4']
df_cm = pd.DataFrame(confusion_matrix, index=class_names, columns=class_names).astype(int)
heatmap = sns.heatmap(df_cm, annot=True, fmt="d")

heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right',fontsize=15)
heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right',fontsize=15)
plt.ylabel('True label')
plt.xlabel('Predicted label')
```
Enumerates through test_loader, feeds results into model and counting up the total one by one in confuction matrix, plot results by pyplot
```
with torch.no_grad():
    for i, (inputs, classes) in enumerate(test_loader):
        inputs = inputs.to(device)
        classes = classes.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        for t, p in zip(classes.view(-1), preds.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
```
Plotted results:
![alt text](https://github.com/changb1/ml_lab4_2/blob/main/%E8%9E%A2%E5%B9%95%E6%93%B7%E5%8F%96%E7%95%AB%E9%9D%A2%202022-04-26%20161706.png "Res Architecture")



A version adjusted to show distribtion by percentage
```
plt.figure(figsize=(15,10))

df_cm=df_cm/df_cm.to_numpy().sum()
heatmap = sns.heatmap(df_cm, annot=True)

heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right',fontsize=15)
heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right',fontsize=15)
plt.ylabel('True label')
plt.xlabel('Predicted label')
```
Plotted results:
![alt text](https://github.com/changb1/ml_lab4_2/blob/main/%E8%9E%A2%E5%B9%95%E6%93%B7%E5%8F%96%E7%95%AB%E9%9D%A2%202022-04-26%20161706.png "Res Architecture")

## Experimental Results
### Highest testing accuracy
#### Screenshots
#### Stuff I want to present
Some thoughts: No DR to Proliferative DR classification, rather than being independent classifier such as 'dog','sheep','cat', that is 


https://www.intellspot.com/data-types/
### Comparison Figures
#### Plotting Comparison Figures (Res 18/50, with/without pretraining)
## Disscussion
