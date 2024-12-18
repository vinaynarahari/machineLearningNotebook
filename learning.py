# Number 1 rule for buidling something using machine learning
#   if you can build a simple rule-based system that doesn't require machine learning, do that 

# If in doubt run the code, experiment experiment, experiment, visualize visualize visualize

import torch
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 


#creating a scalar

scalar = torch.tensor(7)

#tenor methods

scalar.ndim 

#gets tensor back as item int
scalar.item

vector = torch.tensor([7,7])
vector.shape

#MATRIX - capitalization for convention
MATRIX = torch.tensor([[[1,2,3], [1,2,3] , [1,2,3]]])

print(MATRIX.shape)

# overall naming convention: scalar, vector lower case  | matrix, tensor  upper case

'''
Why random tensors in pytorch?

A lot of neural networks start out by creating a bunch of random tensors and then hypertuning that to make accurate predictions
Start at random numbers -> look at data -> update random numbers -> look at data -> update random numbers

'''

random_tensor = torch.rand(3,4)
print(random_tensor)

zero = torch.zeros(3,4) # good for mask
# default data type is torch.float32
ones = torch.ones(3,4)

# not perferred method
print(torch.range(0,10))
#arange is preffered method parameters (start, end, step)
print(torch.arange(0,10))

print(torch.arange(start = 1, end = 10, step = 2))
print(torch.arange(start = 1, end = 10, step = .5))

# different parameters for tensors 

# most common datatypes float32 and float16  -- datatypes is one of the 3 big errors you could run into
test_tensor = torch.tensor([1,2,3], dtype=None, device = None, requires_grad= False)

#converting tensors

float_16_tensor = random_tensor.type(torch.float16)
float_16_tensor = random_tensor.type(torch.half) #same thing
print(float_16_tensor*random_tensor)

''' 

Quick troubleshoot for the 3 big errors in machine learning:
tensors not the same shape - torch.shape
tensors not the same data type - torch.dtype
tensors not on the right device - torch.device

'''

'''
Main types of tensor operations:
Addition
Subtraction
Mutliplication
Division
Matrix Mutliplication

'''

#basic matric multiplication (can do thsi for subtraction and addition and division using the normal operators)

tensor_1 = torch.rand(3,4)
tensor_2 = torch.rand(4,3)

#Note unless matricies are square you can't do matrix multiplication on the two matricies of the same shape

print(torch.matmul(tensor_1, tensor_2))
print(tensor_1 @ tensor_2) # same thing as matmul
print(torch.mm(tensor_1, tensor_2)) # same as matmul as well

# matrix shapes for tensors is the most common error in machine learning

# to fix tensor shape issues, use transpose to switch axis

tensor_a = torch.rand(3,4)
tensor_a_transposed = tensor_a.T

print(tensor_a_transposed.shape) # new shape is now 4,3

#finding min, max , mean, sum 


x = torch.arange(0,100,10)

torch.min(x), x.min() # get min 


torch.max(x), x.max() # get max

torch.mean(x.type(torch.float32)), x.type(torch.float32).mean() # creates mean, data from arange was in long so we need to convert to float 32 which mean needs

torch.sum(x), x.sum() # gets sum 

x.argmin() # gets the index of the min

x.argmax() # gets the index of max

# Reshape - reshapess an input tensor to a defined shaoe
# View - return a view of the input tensor while keeping the same memory as the original tensor
# Stacking - combines tensors together vertically or horizontally 
# Squeeze - remove all '1' demensions from a tensor
# Unsqueeze - add a '1' demension to target tensor 
# Permute - Return a view of the input with demnsions prermuted in a certain way


x1 = torch.arange(0,10, 1)
print(x1.dtype)

x2 = torch.arange(0.,10., 1)
print(x2.dtype)



x = torch.arange(1.,10.)

x_reshaped = x.reshape(9,1) # adds a dimension for each eleemnt


z = x.view(3,3)

print(z) # changes x to 3x3 view of x without chaning x which is currently 1x9
print(x)
print("\n")

x_stacked = torch.stack([x,x], dim = 0)

print(x_reshaped.squeeze().shape) # got rid of all the sigle dimensions, previous shape ([9,1]) -> ([9])

print(x_reshaped.squeeze().unsqueeze(dim = 0))  # this is readding a single dimension 


'''
Different inputs can change the output:

print(x_reshaped.squeeze().unsqueeze(dim = 0))
tensor([[1., 2., 3., 4., 5., 6., 7., 8., 9.]])

print(x_reshaped.squeeze().unsqueeze(dim = 1))
tensor([[1.],
        [2.],
        [3.],
        [4.],
        [5.],
        [6.],
        [7.],
        [8.],
        [9.]])
'''

c = torch.rand(2,3,5)

c_permuted = c.permute(0,2,1) # switches the order of the shape to [2,5,3] because the inputs for the permute is the index of the current dimension
# c_permuted is a different shape but it shares the same sapce in memory as c but its a different view

print(c)
print(c_permuted.shape)

x = torch.arange(1,10).reshape(1,3,3)

print(x[0,2,0])

print(x[:, 0]) # effectivly calling (0,2,0)


print(x[:, 2, 2]) # effectivly calling (0,2,0)



array = np.arange(1.0, 8.0)

tensor = torch.from_numpy(array) # creates tensor with the same data in float 64 b/c thats numpy default data type (pytorch is float32)


# tensor to numpy

tensor = torch.ones(7)

numpy_tensor = tensor.numpy() # dtype of the numpy will be the data type of tensor (default float32)



# reproducing randomness

RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
random_tensor_C = torch.rand(3,4)

torch.manual_seed(RANDOM_SEED)
random_tensor_D = torch.rand(3,4)

print(random_tensor_C == random_tensor_D)

# running on gpu 



if torch.backends.mps.is_available():
    print("MPS device found.")
else:
    print("MPS device not found.")


#device agnostic code
    

device = "mps" if torch.mps.is_available() else "cpu"

#count number of devices

print(torch.cuda.device_count)

x_device = x.to(device)

print(x_device) # tensor is now on the gpu

tensor_on_cpu = x_device.cpu() #switches the device back to the gpu