# **Traffic Sign Recognition** 

## Writeup

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

### Writeup

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/mikaspar/CarND_Term1_Traffic_Sign_Classifier/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799 Images
* The size of the validation set is 4410 Images
* The size of test set is 12630 Images
* The shape of a traffic sign image is 32x32 RGB
* The number of unique classes/labels in the data set is 43 Classes

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...

![alt text]( https://github.com/mikaspar/CarND_Term1_Traffic_Sign_Classifier/blob/master/examples/Images_Training_Set.png)

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to improve the equity of the label frequency in the histogram. That helps the random function to choose the classes randomly and not to overfit the model for classes with high frequency in the training set.

Moreover I applied a random rotation (-7,7 Grades), zoom (.8,1.1) a shear(0.3) on these supplementary images in the training set.

On all pictures of the enhanced training set I applied grayscaling and normalization (0,255) -> (-1,1)  



#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, Valid padding, outputs 28x28x8 	|
| RELU					|												|
| Dropout 0.8 for Training, 1.0 Evaluation
| Max pooling	      	| 2x2 stride,  outputs 14x14x8 				|
| Convolution 5x5	    | 1x1 stride, Valid padding, outputs 10x10x32      									|
| RELU
| Dropout 0.8 for Training, 1.0 Evaluation
| Max pooling	      	| 2x2 stride,  outputs 5x5x32 	
| Fully connected		| 800 -> 120        									|
| RELU
| Dropout 0.55 for Training, 1.0 Evaluation
| Fully connected		| 120 -> 84
| RELU
| Dropout 0.55 for Training, 1.0 Evaluation
| Fully connected		| 84 -> 43
| Softmax				|    43 - Probabilities for each image     									|

 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used the internal GPU Nvidia M2000M. To keep the computing time at acceptable level and not to overload the GPU I used batch size of 128 samples with 50 epochs. 

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.99
* validation set accuracy of 0.97
* test set accuracy of 0.95

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen? The basis structure of the CovNet was provided in the LeNet Lab.
* What were some problems with the initial architecture?
Missing Dropout layers in the LeNet structure lead to overfitting.  Deeper convolution filters compared to the original LeNEt structure can better recognize the structure of the traffic signs. Original LeNet structure was fitted on identification of numbers. With this architecture 99 per cent training accuracy and 97 per cent validation accuracy were reached. This would indicate a slight overfitting of the model to the training set. Further augumentation, and noise application could be another options to improve the validation accuracy.
* 
* Which parameters were tuned? How were they adjusted and why?
0.55 - keep_prob was used in the training for fully connected layers, 0.8 - keep_prob_conv for the convolution layers.
0.0008 rate was used. A much higher rate would prevent the model from converging. A much lower rate would be very costly concerning the computational time.

If a well known architecture was chosen:
* What architecture was chosen? LeNet. Provided in the LeNet Lab.
* Why did you believe it would be relevant to the traffic sign application? Well, it is the first CNN structure that I am trying to understand :)
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
The model shows slight overfitting since the training set accuracy is 0.99 and the validation set accuracy is 0.97. The test set accuracy reached 0.95, which meats the criteria of the project.
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text](https://github.com/mikaspar/CarND_Term1_Traffic_Sign_Classifier/blob/master/examples/5_Germans.png)

The model has no problems to classify the dowloaded images from the web.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

![alt text](https://github.com/mikaspar/CarND_Term1_Traffic_Sign_Classifier/blob/master/examples/5_Guesses.png)

The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 99%. Predicition provides in all 5 images certainty close to 1.0, that the image is of one class. This classification is in all 5 cases correct.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)


For the 5th image, the model is very sure that this is a maximum speed 100 km/h sign  (probability of 0.999698), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .999698         			| 100 km/h   									| 
| .000294     				| 120 km/h 										|
| .000004					| 30 km/h									|
| .000002	      			| 70	km/h			 				|
| .000002				    | 80 km/h      							|



### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


