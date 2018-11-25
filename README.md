# CNN

A script for a classifier learning to label the CIFAR 10 dataset found here: https://www.cs.toronto.edu/~kriz/cifar.html. The approach uses different number of convolutional layers of different sizes, to form a Convolutional Neural Network.

# Requirements

Script is done in Python 3.6 with Theano 1.0. The cifar dataset should be in folder "./cifar-10-batches-py".

# Details

The CNN is implemented in Python, using Theano. The performance is measured with different CNNs, differing in depth (number of convolutional layers before the fully connected layer). 

Grid search can be perfomed to to obtain the best learning rate (α) and depth of convolutional layers. 

Some intermediate steps are printed along with the final confidence matrix. 

One can plot the distribution of the different classes of images, the cost as a function of iteration and the weights of the first convolutional layer.

Notice that the larger CNNs are slow to run!

# Results

The best accuracy reached is 71.3 %, with a CNN of 4 convolution layers followed by a fully connected layer (version 4, filters [13, 26, 28, 23], epochs 105, alpha 1e-4 and minibatch size 16).
