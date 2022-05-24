from torch.nn import Module
from torch import nn, relu
import torch.nn.functional as F
import numpy as np

def findConv2dOutShape(H_in, W_in, conv, pool=2):
	# get conv arguments
	kernel_size = conv.kernel_size
	stride = conv.stride
	padding = conv.padding
	dilation = conv.dilation
	H_out = np.floor((H_in+2*padding[0]-dilation[0]*(kernel_size[0]-1)-1)/stride[0]+1)
	W_out = np.floor((H_in+2*padding[1]-dilation[1]*(kernel_size[1]-1)-1)/stride[1]+1)
	if pool:
		H_out/=pool
		W_out/=pool
	#print(int(H_out))
	return int(H_out), int(W_out)


class Net(nn.Module):
	def __init__(self, params):
		super(Net, self).__init__()
		C_in,H_in,W_in = params["input_shape"]
		init_f = params["initial_filters"]
		num_fc1 = params["num_fc1"]
		num_classes = params["num_classes"]
		self.dropout_rate = params["dropout_rate"]
		self.conv1 = nn.Conv2d(C_in, init_f, kernel_size=3)
		h,w = findConv2dOutShape(H_in, W_in, self.conv1)
		self.conv2 = nn.Conv2d(init_f, 2*init_f, kernel_size=3)
		h,w=findConv2dOutShape(h,w,self.conv2)
		self.conv3 = nn.Conv2d(2*init_f, 4*init_f, kernel_size=3)
		h,w=findConv2dOutShape(h,w,self.conv3)
		
		self.conv4 = nn.Conv2d(4*init_f, 8*init_f, kernel_size=3)
		h,w=findConv2dOutShape(h, w, self.conv4)
		# compute the flatten size
		self.num_flatten = h*w*8*init_f
		self.fc1 = nn.Linear(self.num_flatten, num_fc1)
		self.fc2 = nn.Linear(num_fc1, num_classes)
	def forward(self, x):
		x = F.relu(self.conv1(x))
		x = F.max_pool2d(x, 2, 2)
		x = F.relu(self.conv2(x))
		x = F.max_pool2d(x, 2, 2)
		x = F.relu(self.conv3(x))
		x = F.max_pool2d(x, 2, 2)
		x = F.relu(self.conv4(x))
		x = F.max_pool2d(x, 2, 2)
		x = x.view(-1, self.num_flatten)
		x = F.relu(self.fc1(x))
		x = F.dropout(x, self.dropout_rate, training=self.training)
		x = self.fc2(x)
		return x	

# class Model(Module):
#     def __init__(self):
#         super(Model, self).__init__()   
#         self.relu = nn.ReLU()
#         self.pool = nn.MaxPool2d(kernel_size=2, stride=2) 
#         self.conv1 = nn.Conv2d(3, 8, 3, stride=1) 
#         self.conv2 = nn.Conv2d(8, 16, 3, stride=1) 
#         self.conv3 = nn.Conv2d(16, 32, 3, stride=1) 
#         self.conv4 = nn.Conv2d(32, 64, 3, stride=1) 
#         self.fc1 = nn.Linear(230400, 100)
#         self.fc2 = nn.Linear(100, 2)

#     def forward(self, x):
#         y = self.pool(self.relu(self.conv1(x)))
#         y = self.pool(self.relu(self.conv2(x)))
#         y = self.pool(self.relu(self.conv3(x)))
#         y = self.pool(self.relu(self.conv4(x)))
#         y = y.view(-1, 230400)
#         y = self.fc1(y)
#         y = self.relu(y)
#         y = nn.dropout(y, 0.25, training=self.training)
#         y = self.fc2(y)
#         return y