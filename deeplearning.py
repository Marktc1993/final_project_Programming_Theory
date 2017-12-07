
# coding: utf-8

# A minimum of three different kinds of graphs.  Thus, if you plotted S&P500, NASDAQ, and DJIA against trading day, that would not count as three kinds of graphs but one kind of graph (a stock index value versus time).  As another example, a graph of DJIA values versus time and another graph of DJIA as a percentage of mean versus time would count as two different kinds of graphs.

# In[12]:


"""Mark Conrad
    Programming Theory
    Attribution to open source Software Creators:
    Francois Chollet: https://github.com/fchollet/keras,
    TensorFlow: TensorFlow, the TensorFlow logo and any related marks are trademarks of Google Inc.
    Samir I repurposed his Keras code and made edits. 

"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os 
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
# os.chdir("data")
random_seed = 0
df = pd.read_csv("train/train.csv")


# In[14]:



# 1 Here is a colorful menagerie of exploratory plots 
# Plot I
# # Distribution of outcomes
# The first if a histogram of the distribution of digit labels that we are trying to predict.
plt.figure(1)
plt.title("Histogram of Frequencys per label of Digits in MNIST dataset")
plt.hist(df["label"])
plt.legend()
plt.savefig("Histogram of Frequencys per label of Digits in MNIST dataset")


# In[15]:


# # I don't usually run ahead but before I wade through any jargon here is pure verifiable results that this 
# # model performs well. 

# # Plot II 
# # plot a sample from the input.  
# plt.figure(3)
# Credit to Yassine Ghouzam for correct reshaping procedure for visualization
data = np.array(df)
x = data[:,1:].reshape(-1,28,28,1)
plt.figure(2)
plt.title("Image from MNIST Dataset")
plt.imshow(x[0][:,:,0])
plt.xlabel("Pixels")
plt.ylabel("Pixels")
plt.legend()
plt.savefig("Image from MNIST Dataset")




# plt.figure(3)
# # # Plot III
# # Accuracy of the model over time:
# plt.plot(history.history['acc'], label = "Training Accuracy")
# plt.savefig("Training and Validation accuracy")


# In[16]:


# 4


"""Here I want to demonstrate the ease by which anyone can make a deep learning model using Python and a (front-end) library called Keras. 

Deep learning has been popularized by Google researchers and I want to give you a primer to break through the buzz words and show you how you can do meaningful 
cutting-edge analysis in your current job or project.

I will use Keras (creator Francois Chollet who works at Google and TensorFlow Google's open source Deep Learning backend) to make my predictive model. 

Simply put, deep learning is like an upgraded form of linear regression analysis we have all done in statistics, however it can map to the non-linear 
functions that linear regression falls apart on. With Nvidia graphics cards we can apply deep learning to complex problems such that we can quickly teach this software to 
identify things in images with super-human accuracy. We will explore a model here with with only a few lines of code that achieves state of the art results of 5-10 years ago.  

Here is a model that can identify hand-written digits with ~99% accuracy."""


# # 2
# Conduct three kinds of non-graphical analysis tasks.  Examples include calculating the mean of a field, the median of a field, the autocorrelation of a field, the running mean of a field, the percentage of occurrences of a term, etc.  These should not be exactly the same as in Task #1.  Note that lag-correlation and lag-autocorrelation generally only make sense for timeseries, so if you use the Consumer Complaints database, you can't calculate lag-correlations and lag-autocorrelations.

# In[18]:


# Normalization of data:
# Random factoids
# He initialization: for weights speeds up training. 
# data = pd.read_csv("MNIST_DIGITS")


# 2
df.describe()
# Defining our model -original design attributed to Yassine Ghouzam
model = Sequential()
model.add(Conv2D(filters = 32, kernel_size = (4,4), padding = "Same", activation = 'relu', input_shape = (28,28,1)))
model.add(Conv2D(filters = 32, kernel_size = (4,4), padding = "Same", activation = 'relu'))
model.add(Conv2D(filters = 32, kernel_size = (4,4), padding = "Same", activation = 'relu'))
model.add(MaxPool2D(pool_size=(2,2)))
# drop out neurons with 30% probability.
model.add(Dropout(.3))
# The kernel size determines the magnifying glass that we use to look at the images.
model.add(Conv2D(filters = 64, kernel_size = (3,3), padding = "Same", activation = 'relu'))
model.add(Conv2D(filters = 64, kernel_size = (3,3), padding = "Same", activation = 'relu'))
model.add(Conv2D(filters = 32, kernel_size = (2,2), padding = "Same", activation = 'relu'))
model.add(MaxPool2D(pool_size=(2,2), strides = (2,2)))
model.add(Dropout(.3))
model.add(Flatten())
# now we have wide layers with lots of nodes.
model.add(Dense(256, activation = 'relu'))
# Randomly dropout neurons with 50% probability to help generalize the model.
model.add(Dropout(.5))
# Output layer
model.add(Dense(10, activation = 'softmax'))

# 3
def z_score(array):
    """Here we will normalize the data with mean = 0; standard deviation/ variance = 1
    
    Z-score = (x_i - mean(x))/ standard_deviation(x), we want to do this calculation with arrays to speed up
    calculation.
    
    Parameters -- NumPy array 
    
    Returns -- normalized NumPy array
    
    """
    z_score = np.zeros(array.shape)
    for i in range(array.shape[1]):
        z_score[:,i] = (array[:,i] - np.mean(array[:,i])) / np.std(array[:,i])  
    return z_score
input_array = np.array(df.iloc[1:])
norm_array = input_array/255.
# Another option that shrinks the space even further is to divide each field by the maximum pixel value of 255.

# model_training_accuracy
# # let's check out descriptive statistics.   
# dataframe = pd.Dataframe(array)
# dataframe.describe()


# In[25]:


y = np.array(data[:,0])
# Deep learning likes one-hot encoded vectors.
y = to_categorical(y)


# In[26]:


# Optimizer:
optimizer = Adam(lr=0.001, epsilon = 1e-08, decay = 0)
model.compile(optimizer= optimizer, loss = 'categorical_crossentropy', metrics = ["accuracy"])


# In[27]:


# We can dynamically change the learning rate if the accuracy does not improve, code courtesy of Yassine Ghouzam:
learning_rate_reduction = ReduceLROnPlateau(monitor='acc', patience = 3, verbose =1, factor = 0.5, min_lr =0.00001)


# In[28]:


epochs = 1
batch_size = 32


# In[ ]:


history = model.fit(x, y, batch_size=batch_size, epochs=epochs, verbose = 2, callbacks=[learning_rate_reduction] )


# In[ ]:


history


# In[ ]:


y.shape


# In[ ]:




