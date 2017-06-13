#**Behavioral Cloning** 

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/raw_samples-histogram.png "Raw data histogram"
[image2]: ./examples/center_2016_12_01_13_38_26_805.jpg "High steering 1"
[image3]: ./examples/center_2016_12_01_13_38_42_894.jpg "High steering 2"
[image4]: ./examples/filtered_samples-histogram.png "Histogram after filter"
[image5]: ./examples/augment_brightness_center_2016_12_01_13_31_15_411.jpg "Brightness 1"
[image6]: ./examples/augment_brightness_center_2016_12_01_13_32_39_212.jpg "Brightness 2"
[image7]: ./examples/augment_brightness_center_2016_12_01_13_32_42_446.jpg "Brightness 3"
[image8]: ./examples/augment_shadow_right_2016_12_01_13_33_54_476.jpg "Shadow 1"
[image9]: ./examples/augment_shadow_right_2016_12_01_13_33_55_184.jpg "Shadow 2"
[image10]: ./examples/augment_shadow_right_2016_12_01_13_33_55_285.jpg "Shadow 3"
[image11]: ./examples/augment_shift_center_2016_12_01_13_35_21_674.jpg "Shift 1"
[image12]: ./examples/augment_shift_center_2016_12_01_13_35_21_776.jpg "Shift 2"
[image13]: ./examples/augment_shift_center_2016_12_01_13_35_24_534.jpg "Shift 3"
[image14]: ./examples/center_2016_12_01_13_44_57_783.jpg "Un-flipped Image"
[image15]: ./examples/augment_flipped_center_2016_12_01_13_44_57_783.jpg "Flipped Image"
[image16]: ./examples/with_flipped_samples-histogram.png "Final Histogram"


#### 1. Files

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* myutils.py containing the code for data loading and augmentation
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The **model.py** file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.
I used checkpoint code to save the weights and biases of of each epoch run.  This code is based on techniques described in http://machinelearningmastery.com/check-point-deep-learning-models-keras/

The **myutils.py** file contains the methods for loading and augmenting data.  It also contains methods to help visualize the data.


### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

I started with with a simple model with 2 convolutions and a fully-connected layer, just to test that my code is working.   Once everything started to fit in together, I finally adapted the model used by NVIDIA as described in this paper  - https://arxiv.org/abs/1604.07316.

My model consists of a convolution neural network having 3 convolution layers with 5x5 filters and another 2 convolution layers with 3x3 filters. This is followed by 3 fully-connected layers and a final layer for the steering angle predicted value (model.py lines 60-92).

The model includes ELU layers to introduce nonlinearity (code line 66, 68, 70, 73, 75), and the data is normalized in the model using a Keras lambda layer (code line 62). 

As suggested in the lessons, I added a Cropping2D layer that is useful for choosing an area of interest that excludes the sky and/or the hood of the car (code line 63). 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers with a rate of 0.3 in order to reduce overfitting (model.py lines 79, 81). 

The model was trained and validated on the Udacity-provided data and data I generated as part of augmentation. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer.  The initial learning rate was set to 0.001 (model.py line 23, 88).

#### 4. Appropriate training data

Training data used is that from Udacity plus additional data created in augmentation stage. 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

As mentioned earlier, I adapted the model used by NVIDIA.

With my current setup, this model took time for training. I decided to remove the 1164-neuron Dense layer to reduce parameters and hopefully reduce training time.   This change did not negatively affect the performance of my model so I settled with this final model.   


#### 2. Final Model Architecture

The final model architecture (model.py lines 60-92) consisted of a convolution neural network with the following layers and layer sizes ...

Layer (type)                     Output Shape          Param #     Connected to                     

====================================================================================================

lambda_1 (Lambda)                (None, 160, 320, 3)   0           lambda_input_1[0][0]             

____________________________________________________________________________________________________

cropping2d_1 (Cropping2D)        (None, 75, 320, 3)    0           lambda_1[0][0]                   

____________________________________________________________________________________________________

convolution2d_1 (Convolution2D)  (None, 36, 158, 24)   1824        cropping2d_1[0][0]               

____________________________________________________________________________________________________

elu_1 (ELU)                      (None, 36, 158, 24)   0           convolution2d_1[0][0]            

____________________________________________________________________________________________________

convolution2d_2 (Convolution2D)  (None, 16, 77, 36)    21636       elu_1[0][0]                      

____________________________________________________________________________________________________

elu_2 (ELU)                      (None, 16, 77, 36)    0           convolution2d_2[0][0]            

____________________________________________________________________________________________________

convolution2d_3 (Convolution2D)  (None, 6, 37, 48)     43248       elu_2[0][0]                      

____________________________________________________________________________________________________

elu_3 (ELU)                      (None, 6, 37, 48)     0           convolution2d_3[0][0]            

____________________________________________________________________________________________________

convolution2d_4 (Convolution2D)  (None, 4, 35, 64)     27712       elu_3[0][0]                      

____________________________________________________________________________________________________

elu_4 (ELU)                      (None, 4, 35, 64)     0           convolution2d_4[0][0]            

____________________________________________________________________________________________________

convolution2d_5 (Convolution2D)  (None, 2, 33, 64)     36928       elu_4[0][0]                      

____________________________________________________________________________________________________

elu_5 (ELU)                      (None, 2, 33, 64)     0           convolution2d_5[0][0]            

____________________________________________________________________________________________________

flatten_1 (Flatten)              (None, 4224)          0           elu_5[0][0]                      

____________________________________________________________________________________________________

dense_1 (Dense)                  (None, 100)           422500      flatten_1[0][0]                  

____________________________________________________________________________________________________

dropout_1 (Dropout)              (None, 100)           0           dense_1[0][0]                    

____________________________________________________________________________________________________

dense_2 (Dense)                  (None, 50)            5050        dropout_1[0][0]                  

____________________________________________________________________________________________________

dropout_2 (Dropout)              (None, 50)            0           dense_2[0][0]                    

____________________________________________________________________________________________________

dense_3 (Dense)                  (None, 10)            510         dropout_2[0][0]                  

____________________________________________________________________________________________________

dense_4 (Dense)                  (None, 1)             11          dense_3[0][0]                    

====================================================================================================

Total params: 559,419

Trainable params: 559,419

Non-trainable params: 0


A visualization of the architecture can be found here: https://arxiv.org/pdf/1604.07316.pdf


####3. Creation of the Training Set & Training Process

I ended up just using the Udacity training data as my baseline.   But looking at the distribution of the steering angles, it looks like there is an abundance of zero or near-zero angles.  This means that there are more data driving in straight than curved lanes.  This is not ideal since our model learning could end up having a bias on straight line driving.  We need to filter this data.  Below is the distribution of the raw data.

![alt text][image1]

I was also curious what part of the track generated higher steering angles.   Here are samples of steering angles above 0.9.

![alt text][image2] 
![alt text][image3]


The image on the bridge looks like an anomaly as this is a straight line.  This could have been a result of the driver making a quick steering adjustment in this part of the track.  To eliminate these cases, I decided to also filter out steering angles greater than 0.9.  Here is the distribution after the filter. 

![alt text][image4]

I tried the model with the data after filtering.  For this first attempt, the car did not complete the track and would veer off the drivable portion of the track.   I attribute this to the reduced number of data available for the model to learn from.  Clearly, I would have to add more by augmenting.

**Use Left and Right camera images**   
The simulator captures images from three cameras mounted on the car: a center, right and left camera.  I used the left and right images with a steering correction of 0.28.

            correction = 0.28
            steering_left = steering_center + correction
            steering_right = steering_center - correction

**Add Brightness, shadows, horizontal shift, and flipped images**
I utilized the augmentation ideas and code from Vivek Yadav's work.   However, I only performed these augmentation for images with steering angles greater than 0.20. The purpose of this is to add more representation from higher angled steering to improve the distribution of the data. Below are sample images:

### Adding random brightness:
![alt text][image5]
![alt text][image6]
![alt text][image7]

### Adding random shadow:
![alt text][image8]
![alt text][image9]
![alt text][image10]

### Adding horizontal shift
![alt text][image11]
![alt text][image12]
![alt text][image13]

### Flipping

Unflipped image
![alt text][image14]

Flipped image
![alt text][image15]

After performing these augmentation, we ended up with 12779 samples.  After adding the flipped images, our data distribution looks like this.   It is still not the ideal balance of data I was going for but looks like these can work.
![alt text][image16]

I used a 80-20 split on the data, 20% being used for validation.

Working with images takes a huge chunk of memory to be used especially if the images are loaded all at the same time.  To avoid hitting memory limitations, I used Keras generators to provide just-in-time loading of the image files that are fed to the model during training.   I used a batch size of 32 samples.

I set the number of epochs to 6.  Surprisingly, the best performance came out of epoch number 2.  With this model, the car is able complete Track1 successfully.



