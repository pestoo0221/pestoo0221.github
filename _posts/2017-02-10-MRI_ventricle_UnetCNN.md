---
layout: post
title: Using Unet Deep Convolutional Neural Networks to Segment Ventricle from MR Image (TENSORFLOW) 
excerpt_separator: <!--more-->
---
<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"></script>

![]({{ site.baseurl }}/images/True_Unet_MRI_deco.png)
<!--more-->

**Overview**

From last post, I've performed encoder-decoder convolutional neuronetwork (ED-CNN) to segment the ventricle from the MR image. In this post, I want to see whether using a Unet [1] would help to improve the result. 

**Data set** (**Same as the previous post**)
* MRI DICOM 3D image from 118 subjects. 
* This is a 2D CNN implementation. For each subject's 3D image, I extracted ~34 2D images on the sagittal view where ventricle is present. In total I have 4032 images.
* All the images were preprocessed by Freesurfer for image intensity normalization and to label the ventricles. (so my label information is based on existing algorithm, not manual label. But this is just a demonstration that how well this CNN works) 
* For each image, the original size is 256 x 256. It is too big for my computer to run the deep learning. So I rescaled the images to 96 x 96. Image intesity is scaled to 0~1 for a better performance. Label information is either 0 (not ventricle) or 1 (ventricle). 

**Model**

We used a Unet architechture published last year [1]. We use negative dice coefficient as the loss to train the model. So if the loss is -1, the match will be perfect match. The dice coefficient is calculated as the intersection of true label image and the predicted image, divided by the union of the two: $$ Dice = \frac{Mask_{true} \bigcap Mask_{pred} }{Mask_{true} \bigcup Mask_{pred}} $$. 

**Training result comparison**

1. **Unet**: With 3200 training images (832 testing images), I ran 100 image per batch, and 50 Epochs, it took 5 minutes per epoch with 8 CPU. Dice is higher than 0.46 and you can see that the Dice for training is still in a decreasing trend. 

2. **ED-CNN**: With 4000 training images (32 testing images), I ran 100 image per batch, and 50 Epochs, it took 5 minutes per epoch with 8 CPU. Dice is around 0.45 but it looks that the loss is getting stable after the 35th epoch.

**Testing result comparison**
1. **Unet**: Dice close to 0.45

2. **ED-CNN**: Dice around 0.45
 
TensorBoard shows the training loss and testing loss after each epoch training for **Unet**:

![]({{ site.baseurl }}/images/Unet_MRI_loss.png)

TensorBoard shows the training loss and testing loss after each epoch training for **ED-CNN**:

![]({{ site.baseurl }}/images/Unet_MRI_fakeU_loss.png)


TensorBoard shows the graph of the model for **Unet**:

![]({{ site.baseurl }}/images/Unet_MRI_graph1.png)

TensorBoard shows the graph of the model for **ED-CNN**:

![]({{ site.baseurl }}/images/Unet_MRI_fakeU_graph.png)


The figure below shows the predicted mask versus the "true" mask for both **Unet** and **ED-CNN**:

![]({{ site.baseurl }}/images/Unet_MRI_pred.png)


**Note**

1. Both Unet and and ED-CNN show similar result, with predicted labels looking better than my "true" labels.  

2. The results look similar, although Unet may have slightly higher Dice if I run more epochs. Another factor to consider is that I gave different number of training and testing for this comparsion (Unet: 3000:832 vs. 4000:32). This was to show that even with more testing samples the dice would not be so different from the version with fewer testing samples (I mean magically jumping from 0 to 0.5 or 1). But it also raise a question that my comparison for the Unet and ED-CNN is not a fair comparison. I declare that this is not a scientific proof, and it is just for us to get a taste of how the two networks perform under a similar condition.   

**Code**

Source code could be found on [my github](https://github.com/pestoo0221/tensorflow_MRI_ventricle_CNN).

### Reference

1. Olaf Ronneberger, Philipp Fischer, Thomas Brox, "U-Net: Convolutional Networks for Biomedical Image Segmentation", 2015.

2. Alexander Kalinovsky, Vassili Kovalev, "Lung Image Segmentation Using Deep Learning Methods and Convolutional Neural Networks" 2016

