# **Traffic Sign Recognition** 

## Project Writeup

---

** Build a Traffic Sign Recognition Project **

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./resources/raw_examples.png "Random Sample Data"
[image2]: ./resources/grayscale.png "Conversion to Grayscale"
[image3]: ./resources/processed_image.png "Preprocessed Imagef"
[image4]: ./resources/image_dist.png "Distribution of images types"
[image5]: ./resources/training_progress.png "Training Progress"

## Rubric Points 
[[spec]](https://review.udacity.com/#!/rubrics/481/view)

### Writeup / README

#### 1. The project submission includes all required files 

All project files are archived on github
[here](http://github.com/flaschmurphy/sdc_term_1_proj_2)

- Notebook file with all questions answered and all code cells executed: 
  [link](http://github.com/flaschmurphy/sdc_term_1_proj_2/blob/master/Traffic_Sign_Classifier.ipynb)

- Project writeup: [link](http://github.com/flaschmurphy/sdc_term_1_proj_2/blob/master/writeup.ipynb)

- An HTML or PDF export of the project notebook with the name report.html or report.pdf: 
  [link](http://github.com/flaschmurphy/sdc_term_1_proj_2/blob/master/report.pdf)

- Any additional datasets or images used:
  [link](https://github.com/flaschmurphy/sdc_term_1_proj_2/tree/master/data/other_data)

- Your writeup report as a markdown or pdf file:
  [link](https://github.com/flaschmurphy/sdc_term_1_proj_2/blob/master/writeup.md) (or this file)


### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. 

I used pandas, numpy and matplotlib to explore and analyze the dataset. Below
are some findings.

- Number of training examples is: 34799
- Number of validation examples is: 4410
- Number of testing examples is: 12630
- Image data shape is: (32, 32, 1)
- Number of unique classes is: 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. First off, here is
a random selection of 15 images from the original dataset. As can be seen they
are not very high resolution, and have very varied levels of brightness. 

![alt text][image1]

**Further observations as follows**

- The classes are distributed in equal proportions in the training and
  validation sets. See histogram below which demonstrates this

  ![alt_text][image4]

- The most common classes are approx. 10 times more populous than the least common.

- All the categories are valid and none are blank/NaN, which is good (no need
  to preprocess to cover missing data)

- Some images are very dark (i.e. the pictures were taken in very low light)
  therefore there is a broad range in the brightness. This is a good motivation
  for applying grayscale. For example, images 11830, 18473, 15141 and 17847.

- The images are also quite low resolution and some of them are very hard to
  interpret for a human. Examples: 24452, 13415. For those two images, they
  look like speed limit signs, but the numbers are impossible to read.

- According to http://benchmark.ini.rub.de, the images contain a border of at
  least 10% of the image size, "at least 5 pixels". Additionally, "images are
  not necessarily squared... the actual traffic sign is not necessarily centered
  within the image... this is true for images that were close to the image
  border in the full camera image"

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. 

To preprocess the data I normalized it by subtracting and the dividing by 128.

To help with the brightness issue I used OpenCV to convert the images to
grayscale. Here is an example of a fully preprocessed image:

  ![alt_text][image3]

I also explored the option of generating additional data since the overall
accuracy I was able to achieve didn't get above 95%. One reason for this is
probably that there are many more samples of some images than others in the
data so the model will train 'better' on the more numerous samples but have
less opportunity to train on others. Therefore it would make sense to try to
generate additional images for the sparse ones to end up with an even
distribution. However an additional step would likely be needed to randomize
the generated images e.g. by skewing them and varying their brightness, adding
additional random noise etc, so that the model doesn't have an opportunity to
overfit to the 'fake' data. Due to time constraints, so far I was unable to go
beyond the 1st step of generating the replica images. Because I hadn't followed
up with the other steps, the validation accuracy was very low, approx 30% The
code is archived in the git branch 'with_image_replication'. Using the adagrad
optimizer in the model (see section 3 below) helps with the sparse data as it
performs larger updates for infrequent parameters and smaller updates for frequent
parameters.

#### 2. Describe what your final model architecture looks like 

My final model consisted of the following layers:

| Layer             |      |     Description                              | 
|:------------------|------|---------------------------------------------:| 
| Input             |      | 32x32x3 RGB image                            | 
| Convolution 3x3   |      | 1x1 stride, 32x32x1 to 30x30x6               |
| Max pooling       |      | 2x2 kernel, 30x30x6 to 15x15x6               |
| Convolution 3x3   |      | 1x1 stride, 15x15x6 to 13x13x16              |
| Max pooling       |      | 2x2 kernel, 13x13x16 to 6x6x16               |
| Convolution 1x1   |      | 1x1 stride, 6x6x16 to 6x6x32                 |
| Elu               |      | Activation                                   |
| Fully Connected   |      | With bias, output size 120                   |
| Dropout           |      |                                              |
| Elu               |      | Activation                                   |
| Fully Connected   |      | With bias, output size 84                    |
| Dropout           |      |                                              |
| Fully Connected   |      | With bias, output size 43                    |
|-------------------|------|----------------------------------------------| 
 

#### 3. Describe how you trained your model

In training my model I used a large batch size as I was using an Amazon VM with
plenty of memory and compute power. It might be that the model would improve if
run for more epochs, but 200 was enough to reach 95%. This is visible in the
graph below which shows the validation accuracy plotted against the training
accuracy over the 200 epochs.

![alt_text][image5]

After experimenting with many values, I settled on a learning rate of 0.005 and
a keep probability of 0.7. Learning rates higher than this caused the model to
destabalize, while lower values took too long to train.

The optimizer used was the AdamOptimizer which has the benefits of including
momentum and learning rate decay built in. It also handles sparce data well.

####4. Describe the approach taken for finding a solution and getting the
validation set accuracy to be at least 0.93. Include in the discussion the
results on the training, validation and test sets and where in the code these
were calculated. Your approach may have been an iterative process, in which
case, outline the steps you took to get to the final solution and why you chose
those steps. Perhaps your solution involved an already well known
implementation or architecture. In this case, discuss why you think the
architecture is suitable for the current problem.

My final model results were:
* training set accuracy of ?
* validation set accuracy of ? 
* test set accuracy of ?

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image                    |     Prediction                                | 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign              | Stop sign                                       | 
| U-turn                 | U-turn                                         |
| Yield                    | Yield                                            |
| 100 km/h                  | Bumpy Road                                     |
| Slippery Road            | Slippery Road                                  |


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability             |     Prediction                                | 
|:---------------------:|:---------------------------------------------:| 
| .60                     | Stop sign                                       | 
| .20                     | U-turn                                         |
| .05                    | Yield                                            |
| .04                      | Bumpy Road                                     |
| .01                    | Slippery Road                                  |


For the second image ... 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


