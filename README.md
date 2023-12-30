# Cifar10 autoencoder
- The goal of this project is to create a convolutional neural network autoencoder for the CIFAR10 dataset, with a pre-specified architecture.
- [The original colab file can be found here.](https://colab.research.google.com/drive/1mj6U9gSiMDmXaoDH4HbwY_ngvN9GHOBa?usp=sharing)
- The [CIFAR10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html) contains 60,000 32x32 color images of 10 different classes.
- I use the pytorch library for the implementation of this project.
- This is my unique solution to a project created for Mike X Cohen's "A Deep Understanding of Deep Learning" class.
- Much of the code is adapted from this course.
- Images that have been pushed through the autoencoder can be seen at the bottom of the file.



```python
# import libraries
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader,TensorDataset,Subset
import copy
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import pandas as pd


# for importing data
import torchvision
import torchvision.transforms as T

import matplotlib.pyplot as plt
from IPython import display
display.set_matplotlib_formats('svg')
```

    <ipython-input-1-81b64ca807a1>:20: DeprecationWarning: `set_matplotlib_formats` is deprecated since IPython 7.23, directly use `matplotlib_inline.backend_inline.set_matplotlib_formats()`
      display.set_matplotlib_formats('svg')



```python
# use GPU if available
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)
```

    cpu


## Prepare the data
```python
# download the dataset
# download the CIFAR10 dataset

transform = T.Compose([ T.ToTensor(),
                        T.Normalize([.5,.5,.5],[.5,.5,.5])
                       ])
trainset = torchvision.datasets.CIFAR10(root='cifar10',train = True, download=True, transform = transform)
testset = torchvision.datasets.CIFAR10(root='cifar10', train = False, download=True, transform=transform)


# Remove the label from the dataset, leaving only the image data
trainset = [element[0] for element in trainset]
testset = [element[0] for element in testset]
```

    Files already downloaded and verified
    Files already downloaded and verified


## Create the net
```python
def createTheCIFARAE():

  class aenet(nn.Module):
    def __init__(self):
      super().__init__()

            ### input layer
            # encoding layer
      self.enc = nn.Sequential(
        nn.Conv2d(3,16,4,padding=1,stride=2),
        # output size: (32+2*1-4)/2 + 1 = 16
        nn.LeakyReLU(),
        nn.Conv2d(16,32,4,padding=1,stride=2),
        # output size: (16+2*1-4)/2 + 1 = 8
        nn.LeakyReLU(),
        nn.Conv2d(32,64,4,padding=1,stride=2),
        nn.LeakyReLU()
        )

        #decoding layer
      self.dec = nn.Sequential(
        nn.ConvTranspose2d(64,32,4,padding=1,stride=2),
        nn.LeakyReLU(),
        nn.ConvTranspose2d(32,16,4,padding=1,stride=2),
        nn.LeakyReLU(),
        nn.ConvTranspose2d(16,3,4,padding=1,stride=2)
        )

    def forward(self,x):
      return self.dec( self.enc(x) )

  # create the model instance
  net = aenet()

  # loss function
  lossfun = nn.MSELoss()

  # optimizer
  optimizer = torch.optim.Adam(net.parameters(),lr=.001)

  return net,lossfun,optimizer

```

## Test untrained model
```python
# test the untrained model with some images
net,lossfun,optimizer = createTheCIFARAE()

fig,axs = plt.subplots(2,5,figsize=(10,4))

for i in range(5):
  yHat = net(trainset[i])

  #Reshape the images into right image format
  I = yHat.detach().numpy().transpose((1,2,0))
  J = trainset[i].detach().numpy().transpose((1,2,0))
  J=J/2 + .5
  I=I/2 + .5

  axs[1][i].imshow(I)
  axs[0][i].imshow(J)



plt.suptitle("Images before and after convolution with an untrained net")
plt.tight_layout()
plt.show()
```


    
![svg](images/Cifar10autoencoder_5_0.svg)
    

## Train the net

```python
# Train the Net

def function2trainTheModel():

  # number of epochs
  numepochs = 750

  # initialize losses
  trainlosses = torch.zeros(numepochs)
  testlosses = torch.zeros(numepochs)

  # loop over epochs
  for epochi in range(numepochs):

    # pick a set of images at random

    pics2use = np.random.choice(len(trainset),size=32,replace=False)
    X = [trainset[pic] for pic in pics2use]

    net.train()

    #initialize batchloss
    batchlosses = torch.zeros(len(X))

    for i in range(len(X)):

      # forward pass and loss
      yHat = net(X[i])
      loss = lossfun(yHat,X[i])
      batchlosses[i] = loss.item()

      # backprop
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

    trainlosses[epochi] = torch.mean(batchlosses)

    net.eval()

    # Get losses on test data

    # Get images from test set
    pics2use = np.random.choice(len(testset),size=32,replace=False)
    X = [testset[pic] for pic in pics2use]

    # initialize batchloss
    batchlosses = torch.zeros(len(X))

    for i in range(len(X)):

      #forward pass and loss
      yHat = net(X[i])
      loss = lossfun(yHat,X[i])
      batchlosses[i] = loss.item()

    testlosses[epochi] = torch.mean(batchlosses)

  # end epochs

  # function output
  return trainlosses,testlosses,net


```

## Test the trained model
```python
# test the model on a bit of data
trainlosses,testlosses,net = function2trainTheModel()
```


```python
#plot the losses

plt.plot(trainlosses,'s-',label='Train')
plt.plot(testlosses,'o-', color="red", label='Test',)
plt.xlabel('Epochs')
plt.ylabel('Loss (MSE)')
plt.title('Final train loss=%.3f, Final test loss=%.3f)' % (trainlosses[-1], testlosses[-1]))

plt.show()
```


    
![svg](images/Cifar10autoencoder_8_0.svg)
    



```python
# Visualize images through the trained network

fig,axs = plt.subplots(2,10,figsize=(14,4))

pics2see = np.random.choice(len(testset),size=10,replace=False)

for i in range(len(pics2see)):
  yHat = net(testset[pics2see[i]])

  #Reshape the images into right image format
  I = yHat.detach().numpy().transpose((1,2,0))
  J = testset[pics2see[i]].detach().numpy().transpose((1,2,0))
  J=J/2 + .5
  I=I/2 + .5

  axs[1][i].imshow(I)
  axs[0][i].imshow(J)



plt.suptitle("Images before and after convolution with a trained net")
plt.tight_layout()
plt.show()
```

    WARNING:matplotlib.image:Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
    WARNING:matplotlib.image:Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
    WARNING:matplotlib.image:Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
    WARNING:matplotlib.image:Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
    WARNING:matplotlib.image:Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).



    
![svg](images/Cifar10autoencoder_9_1.svg)
    



```python

```
