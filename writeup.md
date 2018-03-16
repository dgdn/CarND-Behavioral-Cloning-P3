# **Behavioral Cloning**

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:

* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points

### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.

---

### Summary

#### 1. An appropriate model architecture has been employed

I choose the PilotNet as basis, and adjust slightly by add batch normalization after each convolutional and fully connected layer.

The model includes RELU layers to introduce nonlinearity (code line 20), and the data is normalized in the model using a Keras lambda layer (code line 18).

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 21). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of all lane(center, left, right) driving, recovering from the left and right sides of the road and counter clockwise driving.

For details about how I created the training data, see the next section.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to try and test.

My first step was to use a convolution neural network model similar to the LeNet I thought this model might be appropriate because it's always a good way to start with simplest model and LeNet may be the simplest model for image recognition task. To output a single steer angle, I remove the last softmax layer and replace that with a single full connected node without apply activation.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting.

To combat the overfitting, I add dropout to the fully connected layers.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track especially on curve road. To improve the driving behavior in these cases, I had to collect more data about driving curve road.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of 5 convolutional layers following by 4 fully connected layers.

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recovery when fell off the road. These images show what a recovery looks like starting from off the center lane in difference place :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I recorded two laps driving counter clockwise on the center lane in order to get more data points. Here is an example image:

![alt text][image8]

To augment the data set, I also flipped images and angles thinking that this would help model generalize better. For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

After the collection process, I had X number of data points. I then preprocessed this data by resize to (320, 160) which was the input size of the model. Normalization step was deployed in the model using Keras lamda layer.

I finally randomly shuffled the data set and put 20% of the clockwise driving data into a validation set.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 20 as evidenced by validation loss was began to continuously increasing and in the meanwhile train loss remain keep decreasing.I used an adam optimizer so that manually training the learning rate wasn't necessary.

The first architeture I choose was the LeNet. When test this trained model on the driving autonomously on track one, it can drive perfectly on straight lane. But when encounter curve lane, it always fell off the lane. After tweaking a while, I have no way to solove this. So I have to switch to a more complex model PilotNet which was purposed by Nvidia. Using the same data to training PilotNet, it can drive well on the track on all the ways without felt off the road though passing cruve was not perfenctly.

#### 4. Interesting Finding

The model with the lowest loss perform best and steering angle smoothly. When collecting training data, I use keepboard to control the veicle. I can see that the driving image is not smooth because the keyboard input signal frequeeze is low.

I also tried the model with much lower training loss which indicate overfitting. The viecle under this model drive much the way similar to when I drive and can stay on lane nearly all the time. But it adjust steering angle in a much harsh wasy.

Last I want to mention abount the training data. At first I collect the training data casually. To save time I drive the veicle in very high speed mainly 30km/h, this is very hard for me to control the viecle stay on the center of the lane especially when dirve around curve. Sometimes the viecle is nearly fell off the road and the viecle always drift a lot even though you only adjust the steering angle slightly, this is because higher speed will cause more drift with the same steering angle.