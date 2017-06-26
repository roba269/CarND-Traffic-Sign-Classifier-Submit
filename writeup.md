# **Traffic Sign Recognition** 
---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[new_images]: ./images/new_images.png "new images"
[categories]: ./images/categories.png "Samle images for each category"
[distribution]: ./images/distribution.png "Count for each cateogory"
[rotation]: ./images/rotation.png "Ratated Images"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/roba269/CarND-Traffic-Sign-Classifier-Submit/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate summary statistics of the traffic signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Firstly I picked one example from each category, to get an intuitive idea about what these images look like:

![categories][categories]

Then I drew a bar chart about the number of images for each category:

![distribution][distribution]

We can see it's not evenly distributed. So possibly it will be helpful to adding more augmented data to make it an even distribution.

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

Firstly, I converted the images to grayscale by averaging the three channels, and normalized the images by simply `(X - 128) / 128`.

Since some categories contain very few images, I think it's a good idea to do some image augmentation. The augementation methods I tried was left-right flipping and rotating by a small random angle. Turns out that flipping doesn't help. I think the reason is if some signs are flipped, the meanings are really changed, for example, left-turn signal vs. right-turn signal. On the otherhand, add more slightly rotated images did help a lot.

See below for rotated examples: 

![rotation][rotation]

I generated 3 additional rotated images for each original image, so the size of training set changed from 34799 to 139196. The distribution kept same. (But I think an evenly distribution may be better.)

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model is as following:

| Layer         		      |     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		      | 32x32x1 Greyscale image   							| 
| Convolution 5x5     	 | 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					             |												|
| Max pooling	      	   | 2x2 stride, valid padding, outputs 14x14x6 				|
| Convolution 5x5	      | 1x1 stride, valid padding, outputs 10x10x16    |
| RELU					             |												|
| Max pooling	      	   | 2x2 stride, valid padding, outputs 5x5x16  				|
| Fully connected		     | 400 => 120       |
| RELU					             |												|
| Dropout               | with 50% keep | 
| Fully connected		     | 120 => 84       |
| RELU					             |												|
| Dropout               | with 50% keep | 
| Fully connected		     | 84 => 43       |
| Softmax				|        									|

Initally I didn't add the dropout. Then I noticed there was an overfitting - very high accuracy on training set but not so good on validation set. After adding dropout after fc layer to mitigate the overfitting, things got much better.

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used learning rate 0.001, batch size 128, and the number of epochs is 20. I used AdamOptimizer. Most of these parameters are borrowed from the LeNet example in the course. I increased the number of epochs from 10 to 20, because I found the accurracy is still increasing significantly around epoch 10. 

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.994
* validation set accuracy of 0.974
* test set accuracy of 0.950

I used the LeNet architecture from the course matrials before this project, because I think the image size (32x32) fits very well, and the complexity of both task (digit recognition vs traffic sign classifaction) is comparable.

As mentioned above, I noticed the original model got almost 100% accurary on training set, but stuck on 80+% on validation set after a few epochs, which may indicates an overfitting. So I added two dropout layers after the full-connected layers, and it worked very well.

Originally I tried 10 epochs, but then I found the accuracy still kept increasing at the 10th epoch. So I extended it to 20 epochs to get a better final result.  

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web (after resizing):

![new images][new_images]

I thought they are all not very difficult, no obstacle, no blur, etc. But interestingly the model misclassified the "60 km/h" sign.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 60 km/h      		| 30 km/h   									| 
| Right-of-way     			| Right-of-way 										|
| Stop					| Stop											|
| Bumpy Road	      		| Bumpy Road					 				|
| Children	Crossing		| Children Crossing      							|

The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. The model mistakenly consider the 60 km/h sign as 30 km/h. I think we can dive into it, analyzing if the model is really doing worse on the speed number sign on validation set. If so, we can try to add more data for these categories.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 14th cell of the Ipython notebook.

For all the images, the model is pretty sure. I listed the top 5 categories for each image:

```
Image 0:
 0.99981: Speed limit (30km/h)
 0.00013: Right-of-way at the next intersection
 0.00004: Priority road
 0.00001: Roundabout mandatory
 0.00001: Speed limit (50km/h)
Image 1:
 1.00000: Right-of-way at the next intersection
 0.00000: Pedestrians
 0.00000: General caution
 0.00000: Priority road
 0.00000: End of no passing by vehicles over 3.5 metric tons
Image 2:
 0.95379: Stop
 0.04591: Turn left ahead
 0.00014: Speed limit (60km/h)
 0.00009: Ahead only
 0.00002: Keep right
Image 3:
 0.99999: Bumpy road
 0.00001: Bicycles crossing
 0.00000: Traffic signals
 0.00000: Road work
 0.00000: Children crossing
Image 4:
 0.99981: Children crossing
 0.00019: Beware of ice/snow
 0.00000: Right-of-way at the next intersection
 0.00000: Slippery road
 0.00000: Dangerous curve to the right
```

It's kind of surprising that the model gave a wrong prediction for the first image confidently, which may suggest this model is really doing not good on the speed number.


