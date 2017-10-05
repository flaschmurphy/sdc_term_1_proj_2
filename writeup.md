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

[image6]: ./data/other_images/a_speed80_5.png "Speed 80"
[image7]: ./data/other_images/a_stop_14.png "Stop"
[image8]: ./data/other_images/resized_stop3_14.png "Stop"
[image9]: ./data/other_images/yield_15.png "Yield"
[image10]: ./data/other_images/resized_pedestrians_15.png "Pedestrians"

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

- According to http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset,
  the images contain a border of at least 10% of the image size, "at least
  5 pixels". Additionally, "images are not necessarily squared... the actual
  traffic sign is not necessarily centered within the image... this is true for
  images that were close to the image border in the full camera image"

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. 

To preprocess the data I normalized it by subtracting 128 and dividing by 128.
As the range of values in the images is from 0 to 255, subtracting and dividing
by 128 is a simple way to normalize.

To help with the brightness issue I used OpenCV to convert the images to
grayscale. Here is an example of a fully preprocessed image:

  ![alt_text][image3]

I also explored the option of generating additional data since the overall
accuracy I was able to achieve didn't get above 96.5%. One reason for this is
probably that there are many more samples of some images than others in the
data so the model will train 'better' on the more numerous samples but have
less opportunity to train on others. Therefore it would make sense to try to
generate additional images for the sparse ones to end up with an even
distribution. However an additional step would likely be needed to randomize
the generated images e.g. by skewing them and varying their brightness, adding
additional random noise etc, so that the model doesn't have an opportunity to
over fit to the 'fake' data. Due to time constraints, so far I was unable to go
beyond the 1st step of generating the replica images. Because I hadn't followed
up with the other steps, the validation accuracy was very low however.

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
| Elu               |      | Exponential RELU Activation                  |
| Fully Connected   |      | With bias, output size 120                   |
| Dropout           |      |                                              |
| Elu               |      | Exponential RELU Activation                  |
| Fully Connected   |      | With bias, output size 84                    |
| Dropout           |      |                                              |
| Fully Connected   |      | With bias, output size 43                    |
|-------------------|------|----------------------------------------------| 
 
For the optimizer I used the AdamOptimizer which provides learning rate
momentum out of the box at the possible cost of additional computation. Another
benefit of the Adam optimizer is that it is relatively stable to changes in
other hyperparameters.

#### 3. Describe how you trained your model

In training my model I used a large batch size as I was using an Amazon VM with
plenty of memory and compute power. After trying out various high and low
values for the number of epochs, I trained the model for this report on 100
epochs. This number (together with the other hyperparameters) achieved
a training accuracy of 99% with a validation accuracy of up to 96.5% depending
on the initial random values of the weights. Going beyond 100 epochs didn't
seem to produce any significant benefit and was leading to over fitting on the
training. This is visible in the graph below which shows the validation
accuracy plotted against the training accuracy over all epochs.

![alt_text][image5]

After experimenting with many values, I settled on a learning rate of 0.0025 and
a keep probability of 0.5. Learning rates higher than this caused the model to
destabilize, while lower values took too long to train. Increasing the keep
probability was leading to over fitting.

#### 4. Describe the approach taken for finding a solution 

To achieve a minimum of 93% accuracy, I started with the example of the LeNet
lab exercise from the previous lesson and modified dimensions to match the
input data. LeNet is a famous architecture for image classification that
leverages convolutions, pooling and fully connected layers. CNNs (Convolutional
Neural Networks) were inspired by research on the visual cortex by Hubel and
Wiesel in the 1960s. The word 'convolution' is a mathematical term that refers
to the operation performed instead of general matrix multiplication.
Convolutions work well on data that has a grid like topology such as images or
time series data. Convolutional networks have a few key benefits when processing
such data: sparce interactions, parameter sharing and equivariance. These
properties enable convolutional networks to deal with objects that may vary in
location on each image, while being computationally and spatially efficient.

To achieve the target accuracy, the most significant step was to implement
working preprocessing (although quite basic). As mentioned above,
I experimented with generating additional sample data for the less populous
classes, but discovered it is not enough to simply replicate the input data
many times. To enhance validation & test accuracy on the less prevalent
classes, it would be necessary to have a comparable number of unique samples
for all input classes. This would mean replicating work done by others to
mutate the input images (e.g. add random brightness, skew, rotate, etc) and add
this data to the training set. Interestingly, simply replicating infrequent
images so that they become more frequent in the training data is not helpful
(and actually produced worse results in my experiments). Since it was possible
to achieve 93% using only Gray scaling and normalization due to time
considerations, I didn't go further than that in the end.

To increase validation accuracy, I decided to deepen the model by
adding a 3rd conv layer. Once I had the dimensions correct, I observed that the
performance of the validation accuracy had improved but was still not high
enough. I then switched from RELU to ELU activations, which have the benefit of
including negative values which enables them to push mean unit activations
closer to zero. These modifications had a positive effect on training and
validation accuracy and enabled the overall training time to be reduced
significantly. 

Drop out is a technique A major stumbling block was the keep probability. A bug
in my code was causing the training step to correctly feed a value of 0.5 for
`keep_prob`, but when validating it was unfortunately also feeding in 0.5 and
not 1.0. This meant that the results produced by the model were somewhat random
which took me a long time to diagnose.

My final model results were:
* training set accuracy of 0.999
* validation set accuracy of 0.963
* test set accuracy of 0.95


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report

Here are German traffic signs that I found on the web (note: they have been
resized to 32x32 here to make them easier to display):

![image6] ![image7] ![image8] 
![image9] ![image10]

#### 2. Discuss the model's predictions on these new traffic signs 
*and compare the results to predicting on the test set. At a minimum, discuss
what the predictions were, the accuracy on these new predictions, and compare
the accuracy to the accuracy on the test set* 

The images I downloaded look significantly different to the training images.
For example they are higher resolution, higher brightness and intensity of
color. For these reasons the images are in fact not easy for the model to
classify/generalize to. 

To try to help with that I experimented with various preprocessing steps
manually, including resizing the images to be 32x32 pixels, adding Gaussian
blur and other options. In the end I was not able to get the prediction
accuracy for the downloaded images to go above 15%, which is disappointing
given the high accuracy on the test images. Only two of the images were
successfully predicted as a stop sign and a dangerous curve sign. There seems
to be a heavy bias towards class ID #34 which is the 'turn left ahead' sign.
It's not clear to me why this is the case, but it is an interesting result
especially given that the turn left sign is not particularly dominant in the
training data (only 360 samples out of ~35k).

I expect that with further experimentation and research into preprocessing
options it would be possible to bring this value higher.  I conclude that in
a real-life system, a significant portion of the engineering effort would need
to be spent ensuring that all images being sent to the model for predictions
have been sanitized properly and probably also that the model is trained on
a higher number of samples, with a greater variance in their structure and
content.

Here are the results of the prediction:

| Image                          | Actual   | Prediction                                    | 
|:-------------------------------|:--------:|----------------------------------------------:| 
| speed80_5.png                  |  5       | 16 (Vehicles over 3.5 metric tons prohibited) |
| yield_15.png                   | 15       | 34 (Turn left ahead)                          |
| keep_right_38.png              | 38       | 34 (Turn left ahead)                          |
| dangerous_curve_20.png         | 20       | 20 (Dangerous curve to the right)             |
| resized_pedestrians_15.png     | 15       | 34 (Turn left ahead)                          |
| a_stop_14.png                  | 14       | 14 (Stop)                                     |
| a_stop2_14.png                 | 14       | 34 (Turn left ahead)                          |
| a_slipperyroad_23.png          | 23       | 0 (Speed limit (20km/h))                      |
| resized_speed80_5.png          |  5       | 16 (Vehicles over 3.5 metric tons prohibited) |
| pedestrians_15.png             | 15       | 34 (Turn left ahead)                          |
| no_entry_17.png                | 17       | 34 (Turn left ahead)                          |
| a_speed80_5.png                |  5       | 14 (Stop)                                     |
| resized_no_entry_17.png        | 17       | 34 (Turn left ahead)                          |
| a_traffic_lights_26.png        | 26       | 34 (Turn left ahead)                          |
| resized_dangerous_curve_20.png | 20       | 20 (Dangerous curve to the right)             |
| resized_stop3_14.png           | 14       | 35 (Ahead only)                               |
| a_speed50_2.png                |  2       | 34 (Turn left ahead)                          |
| stop3_14.png                   | 14       | 35 (Ahead only)                               |
| resized_keep_right_38.png      | 38       | 34 (Turn left ahead)                          |


#### 3. Describe how certain the model is when predicting on each of the five new images 
*by looking at the softmax probabilities for each prediction. Provide the top 5
softmax probabilities for each image along with the sign type of each
probability.*

Below are the top 5 predictions for each of my images. The code to generate
this table can be seen in the 2nd last code cell of the notebook in the git
repo (the last one being the optional exercises).  The key code line is:

`topk = sess.run(tf.nn.top_k(tf.nn.softmax(scores), k=5))`

If all images were to end up with equal probabilities for all possible
predictions, the value for each of the slots would be 1/43 = 0.0233. Most of
the predictions are not very far away from this value meaning the model is not
very certain of it's predictions. Even where the predictions were correct, the
certainty was very tentative. For example, 'a_stop2_14.png' was correctly
predicted, but with a probability of just 0.0244 whereas the 2nd probability
was 0.0243! The predictions for the two 'dangerous curve' images was slightly
better, but still just 0.0282 in 1st place vs. 0.0270, and 0.0281 vs. 0.0274. 


##### Top 5 Softmax for resized_speed80_5.png:

| Probability | Prediction                                    |
|:------------|:----------------------------------------------|
|      0.0274 | Vehicles over 3.5 metric tons prohibited (16) |
|      0.0270 | No vehicles                              (15) |
|      0.0265 | Right-of-way at the next intersection    (11) |
|      0.0262 | Priority road                            (12) |
|      0.0261 | Ahead only                               (35) |



##### Top 5 Softmax for resized_no_entry_17.png:

| Probability | Prediction                                    |
|:------------|:----------------------------------------------|
|      0.0287 | Turn left ahead                          (34) |
|      0.0283 | Ahead only                               (35) |
|      0.0266 | Speed limit (60km/h)                     ( 3) |
|      0.0265 | Vehicles over 3.5 metric tons prohibited (16) |
|      0.0264 | Bicycles crossing                        (29) |



##### Top 5 Softmax for a_stop2_14.png:

| Probability | Prediction                                    |
|:------------|:----------------------------------------------|
|      0.0294 | Turn left ahead                          (34) |
|      0.0281 | Speed limit (60km/h)                     ( 3) |
|      0.0271 | Ahead only                               (35) |
|      0.0269 | No vehicles                              (15) |
|      0.0264 | General caution                          (18) |



##### Top 5 Softmax for a_speed50_2.png:

| Probability | Prediction                                    |
|:------------|:----------------------------------------------|
|      0.0247 | Turn left ahead                          (34) |
|      0.0241 | Ahead only                               (35) |
|      0.0240 | Stop                                     (14) |
|      0.0239 | Speed limit (60km/h)                     ( 3) |
|      0.0239 | Yield                                    (13) |



##### Top 5 Softmax for dangerous_curve_20.png:

| Probability | Prediction                                    |
|:------------|:----------------------------------------------|
|      0.0282 | Dangerous curve to the right             (20) |
|      0.0270 | Right-of-way at the next intersection    (11) |
|      0.0268 | No entry                                 (17) |
|      0.0267 | Stop                                     (14) |
|      0.0260 | Speed limit (60km/h)                     ( 3) |



##### Top 5 Softmax for resized_dangerous_curve_20.png:

| Probability | Prediction                                    |
|:------------|:----------------------------------------------|
|      0.0281 | Dangerous curve to the right             (20) |
|      0.0274 | Right-of-way at the next intersection    (11) |
|      0.0269 | Stop                                     (14) |
|      0.0266 | No entry                                 (17) |
|      0.0259 | Speed limit (60km/h)                     ( 3) |



##### Top 5 Softmax for a_speed80_5.png:

| Probability | Prediction                                    |
|:------------|:----------------------------------------------|
|      0.0244 | Stop                                     (14) |
|      0.0243 | Road narrows on the right                (24) |
|      0.0243 | No entry                                 (17) |
|      0.0242 | Turn left ahead                          (34) |
|      0.0241 | Turn right ahead                         (33) |



##### Top 5 Softmax for yield_15.png:

| Probability | Prediction                                    |
|:------------|:----------------------------------------------|
|      0.0257 | Turn left ahead                          (34) |
|      0.0254 | Ahead only                               (35) |
|      0.0251 | Vehicles over 3.5 metric tons prohibited (16) |
|      0.0247 | Speed limit (60km/h)                     ( 3) |
|      0.0245 | General caution                          (18) |



##### Top 5 Softmax for resized_keep_right_38.png:

| Probability | Prediction                                    |
|:------------|:----------------------------------------------|
|      0.0277 | Turn left ahead                          (34) |
|      0.0266 | Vehicles over 3.5 metric tons prohibited (16) |
|      0.0258 | Bicycles crossing                        (29) |
|      0.0258 | Speed limit (60km/h)                     ( 3) |
|      0.0258 | No entry                                 (17) |



##### Top 5 Softmax for a_slipperyroad_23.png:

| Probability | Prediction                                    |
|:------------|:----------------------------------------------|
|      0.0291 | Speed limit (20km/h)                     ( 0) |
|      0.0267 | Road narrows on the right                (24) |
|      0.0262 | Stop                                     (14) |
|      0.0260 | Right-of-way at the next intersection    (11) |
|      0.0259 | Yield                                    (13) |



##### Top 5 Softmax for resized_stop3_14.png:

| Probability | Prediction                                    |
|:------------|:----------------------------------------------|
|      0.0280 | Ahead only                               (35) |
|      0.0275 | Turn left ahead                          (34) |
|      0.0273 | Speed limit (60km/h)                     ( 3) |
|      0.0272 | Stop                                     (14) |
|      0.0270 | General caution                          (18) |



##### Top 5 Softmax for no_entry_17.png:

| Probability | Prediction                                    |
|:------------|:----------------------------------------------|
|      0.0288 | Turn left ahead                          (34) |
|      0.0283 | Ahead only                               (35) |
|      0.0266 | Speed limit (60km/h)                     ( 3) |
|      0.0263 | Vehicles over 3.5 metric tons prohibited (16) |
|      0.0263 | Bicycles crossing                        (29) |



##### Top 5 Softmax for resized_pedestrians_15.png:

| Probability | Prediction                                    |
|:------------|:----------------------------------------------|
|      0.0281 | Turn left ahead                          (34) |
|      0.0276 | Ahead only                               (35) |
|      0.0270 | General caution                          (18) |
|      0.0268 | Speed limit (20km/h)                     ( 0) |
|      0.0261 | Speed limit (60km/h)                     ( 3) |



##### Top 5 Softmax for pedestrians_15.png:

| Probability | Prediction                                    |
|:------------|:----------------------------------------------|
|      0.0276 | Turn left ahead                          (34) |
|      0.0273 | Ahead only                               (35) |
|      0.0267 | General caution                          (18) |
|      0.0267 | Speed limit (20km/h)                     ( 0) |
|      0.0263 | No vehicles                              (15) |



##### Top 5 Softmax for a_stop_14.png:

| Probability | Prediction                                    |
|:------------|:----------------------------------------------|
|      0.0275 | Stop                                     (14) |
|      0.0271 | No vehicles                              (15) |
|      0.0271 | General caution                          (18) |
|      0.0268 | Turn left ahead                          (34) |
|      0.0267 | Ahead only                               (35) |



##### Top 5 Softmax for stop3_14.png:

| Probability | Prediction                                    |
|:------------|:----------------------------------------------|
|      0.0280 | Ahead only                               (35) |
|      0.0273 | Speed limit (60km/h)                     ( 3) |
|      0.0273 | Stop                                     (14) |
|      0.0272 | Turn left ahead                          (34) |
|      0.0271 | General caution                          (18) |



##### Top 5 Softmax for a_traffic_lights_26.png:

| Probability | Prediction                                    |
|:------------|:----------------------------------------------|
|      0.0281 | Turn left ahead                          (34) |
|      0.0274 | Ahead only                               (35) |
|      0.0269 | General caution                          (18) |
|      0.0262 | Speed limit (60km/h)                     ( 3) |
|      0.0261 | Bicycles crossing                        (29) |



##### Top 5 Softmax for keep_right_38.png:

| Probability | Prediction                                    |
|:------------|:----------------------------------------------|
|      0.0270 | Turn left ahead                          (34) |
|      0.0266 | Vehicles over 3.5 metric tons prohibited (16) |
|      0.0263 | Bicycles crossing                        (29) |
|      0.0256 | No entry                                 (17) |
|      0.0255 | Ahead only                               (35) |



##### Top 5 Softmax for speed80_5.png:

| Probability | Prediction                                    |
|:------------|:----------------------------------------------|
|      0.0273 | Vehicles over 3.5 metric tons prohibited (16) |
|      0.0270 | No vehicles                              (15) |
|      0.0264 | Right-of-way at the next intersection    (11) |
|      0.0261 | Priority road                            (12) |
|      0.0260 | Yield                                    (13) |







