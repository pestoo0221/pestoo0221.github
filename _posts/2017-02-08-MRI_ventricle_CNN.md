---
layout: post
title: Using Encode-Decoder Deep Convolutional Neural Networks to Segment Ventricle from MR Image (TENSORFLOW) 
excerpt_separator: <!--more-->
---
<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"></script>

![]({{ site.baseurl }}/images/Unet_MRI_fakeU_deco.png)
<!--more-->

**Overview**

Convolutional neural networks (CNN) have been widely used for image recognition. In this post, I am going to show you the application of CNN in extracing ventricles from structural MR images. 

**Data set** 

* MRI DICOM 3D image from 118 subjects. 
* This is a 2D CNN implementation. For each subject's 3D image, I extracted ~34 2D images on the sagittal view where ventricle is present. In total I have 4032 images.
* All the images were preprocessed by Freesurfer for image intensity normalization and to label the ventricles. (so my label information is based on existing algorithm, not manual label. But this is just a demonstration that how well this CNN works) 
* For each image, the original size is 256 x 256. It is too big for my computer to run the deep learning. So I rescaled the images to 96 x 96. Image intesity is scaled to 0~1 for a better performance. Label information is either 0 (not ventricle) or 1 (ventricle). 

![]({{ site.baseurl }}/images/Unet_MRI_fakeU_input.png)

The figure above shows the input image and its label information for the ventricle. Columns 1 & 2 show the original image and the ventricle mask. Columns 3 & 4 show the image and masks after rescale to 96x96. You can see that the label information is not that smooth from the original mask (well, this is not manual label), and it looks worse in the rescaled image. Let's see whether we can get some nice results based on what we have :)

**Model**

We used an archetecture similar to the Unet architechture published last year [1], but it is a simpeler type, encode-decoder CNN. We use negative dice coefficient as the loss to train the model. So if the loss is -1, the match will be perfect match. The dice coefficient is calculated as the intersection of true label image and the predicted image, divided by the union of the two: $$ Dice = \frac{Mask_{true} \bigcap Mask_{pred} }{Mask_{true} \bigcup Mask_{pred}} $$. 

This implements with dice coefficient of 0.45 for testing images. With 4000 training images, I ran 100 image per batch, and 50 Epochs, it took 5 minutes per epoch with 8 CPU. But you will see that the loss is pretty stable after the 30th epoch. 

TensorBoard shows the training loss and testing loss after each epoch training:

![]({{ site.baseurl }}/images/Unet_MRI_fakeU_loss.png)


TensorBoard shows the graph of the model:

![]({{ site.baseurl }}/images/Unet_MRI_fakeU_graph.png)

TensorBoard shows the predicted mask versus the "true" mask:

![]({{ site.baseurl }}/images/Unet_MRI_fakeU_pred.png)


**Note**

1. I'm not sure whether it is just me or not, I think the predicted mask is better than the "true" mask, although the dice coefficient is only 0.45. This could be due to the fact that my "true" mask is actually not very good in the sense of continuity/smoothness. It  has holes inside of the mask which should not happen. While the predicted mask is continuous, the dice coefficient can't reach a very high value. In [2], they used a similar model and got dice coefficient of ~ 0.97 for both testing ang training images, but they have trained it with 5000 iterations and 85 epochs.   

2. You may notice that I have 4000 training images and only 32 testing images. I did this because: 1) I want to have many training samples as possible. 2) the dice coefficient is calculated based on each sample image, and the final dice coefficient is based on the average across all testing samples. The final dice coefficient would not be as affected as when we calculate the accruacy for examples like music classification. E.g. For music classification (my precious post), one sample will give out an answer of 0 or 1 for wrong or right classification. So with 1 sample, the accuracy would either be 0 or 100%. With two samples, it will jump between 0, 50% or 100%. This change is huge, and with more samples, the accuracy would become more stable. Dice coefficient is another case. For one subject, the dice coefficient already considers all the wrongly / correctly classified pixels in that image. Extra samples should give similar result as the one testing sample. 

3. Conditional random field (CRF) could help to improve the smoothness of the boundary, but I did not try it here. I may try it out in another experiment. 

4. While the archetecture here is similar to the Unet architechture published last year [1], it is actually just encoder - decoder architecture, it reduces the image size (contract part) and then upsamles it back to the original size (expand part). I did not merge the information from the contract part to the expand part so that the contract part's information could be re-used in the expand part. Will try it in the next post.  

**Code**

Source code could be found on [my github](https://github.com/pestoo0221/tensorflow_MRI_ventricle_CNN).

### Reference

1. Olaf Ronneberger, Philipp Fischer, Thomas Brox, "U-Net: Convolutional Networks for Biomedical Image Segmentation", 2015.

2. Alexander Kalinovsky, Vassili Kovalev, "Lung Image Segmentation Using Deep Learning Methods and Convolutional Neural Networks" 2016

