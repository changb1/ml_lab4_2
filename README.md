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
Will provide some brief introduction on the following topic: Diabetic Retinopathy & Dataset, Resnet Network, Pytorch and Dataloader
### Diabetic Retinopathy & Dataset
Diabetic retinopathy is a complication of diabetes, caused by high blood sugar levels damaging the back of the eye (retina). It can cause blindness if left undiagnosed and untreated.

https://www.nhs.uk/conditions/diabetic-retinopathy/#:~:text=Diabetic%20retinopathy%20is%20a%20complication,it%20could%20threaten%20your%20sight.

 large set of high-resolution retina images taken under a variety of imaging conditions. A left and right field is provided for every subject. Images are labeled with a subject id as well as either left or right (e.g. 1_left.jpeg is the left eye of patient id 1).

A clinician has rated the presence of diabetic retinopathy in each image on a scale of 0 to 4, according to the following scale:


|Label|Meaning|
  |:-:|:-:|
  |0|**`No DR`**|
  |1|**`Mild`**|
  |2|**`Moderate`**|
  |3| **`Severe`**|
  |4| **`Proliferative DR`**|


https://www.kaggle.com/c/diabetic-retinopathy-detection#description

### Resnet Network
### Pytorch & Dataloader

## Experiment Stages
### Details of my model (Resnet) 
### Details of my Dataloader
### Describing my evaluation matrix through the confusion matrix
## Experimental Results
### Highest testing accuracy
#### Screenshots
#### Stuff I want to present
### Comparison Figures
#### Plotting Comparison Figures (Res 18/50, with/without pretraining)
## Disscussion
