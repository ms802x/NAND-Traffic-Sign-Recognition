# **Traffic Sign Recognition** 


### This is the third udacity project for traffic sign classifcation 

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

[image1]: ./wup/1.png "training"
[image2]: ./wup/2.png "valid"
[image3]: ./wup/3.png "testinh"
[image4]: ./wup/4.png "training"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
## Writeup / README

# 1. Data Set Summary & Exploration

I have used numpy to get the basics overview of my data:

* Number of training examples = 34799
* Number of validating examples = 4410
* Number of testing examples = 12630
* Image data shape = (32, 32, 3)
* Number of classes = 43


# 2. Exploratory visualization of the dataset.
The data in my training set :

![alt text][image4]

Here is an exploratory visualization of the data set. It is a bar chart showing how the data is distrbuted in the training, validation and testing set.

![alt text][image1]
![alt text][image2]
![alt text][image3]

we can see from the images above the training, valid, testing data are distrbuted in evenly. 

# 3. Design and Test a Model Architecture

## Preprocessing

Here I tried to convert the traffic sign to gray scale first then I have normlized the model by subtracting and dividing by 128, however, the model failed. Therefore, I have changed the preprocessing by only normlize the rgb image and subtracting its mean and dividing by its variance. 

## Model Architecture

The architecture used here is lenet. This architecture recives and input of (32,32). This 32 by 32 is the image size and it goes through a convolutional layer and subsampling layers (relu and pooling )till it gives a final output which is a Neural network that capable to give prediction.

In the model, Cx indicate convlutional layers, whereas, Sx are the subsampling layers. The layer C1 produce 6 output features with 28 by 28 by 3 input (RGB). The sub sampling layer S2 downsample the 6 features map recived from C1. Then C3, is a convlutional layer which take the features produced by the S2 and maps them to 10 features map of c3. The fourh layer S4 again downsample C3. Then c5 is convolutional layer with 120 features map connected to S4.  S5 is another layer with 84 units and lastly the output layer is fully conetect softmax function with 43 output 

## Model Training

Now in training the model, as I have stated earlier I tried to train gray scale sign images but the prediction was bad so I have used the 3 color rgb image to get more information of the image. The training batch and epochs were 128 and 30 respectivly. The valdiation and testing accuracies were 93.7% and 93.4 respectivly. 

    logits = LeNet(x)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
    loss_operation = tf.reduce_mean(cross_entropy)
    optimizer = tf.train.AdamOptimizer(learning_rate = rate)
    training_operation = optimizer.minimize(loss_operation)

The optimization algorathem used is AdamOptimizer. What makes this algorathem preferable over gradient are:

* Computationally efficient.
* Invariant to diagonal rescale of the gradients.
* Well suited for problems that are large in terms of data and/or parameters.
* Appropriate for problems with very noisy/or sparse gradients.
* Hyper-parameters have intuitive interpretation and typically require little tuning.

Also to make the model give better acuracy, training rate and the epochs had been slightly decreased/increased to get optimum result. the Training rate was 0.003.

## Model on New Images

After training the model, I tried to test it on new trafic sign images from the internet, I got 90% prediction acuracy.  However, my model for the one of the wrong classfied images were confiedent that its classification is correct. The image wrong classfied was the 100km sign was not clear for the algorathem to detect and that is why it was wrong clasified. 

To see the test images and probablities and the predictions check the html file in the repository. 
