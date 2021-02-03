import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from tcn import TemporalConvNet#, SimpleConvNet
import torch.nn.functional as F

class TCN(nn.Module):
	def __init__(self, input_size, output_size, num_channels, kernel_size, dropout, dilations):
		super(TCN, self).__init__()
		self.tcn = TemporalConvNet(input_size, num_channels, kernel_size, dropout=dropout)
		self.linear = nn.Linear(num_channels[-1], output_size)
		self.softmax = nn.Softmax(dim=2)

	def forward(self, x):
		# x needs to have dimension (N, C, L) in order to be passed into CNN
		output = self.tcn(x.transpose(1, 2)).transpose(1, 2)

		output = self.linear(output).double()
		return self.softmax(output)



# class MyTemporalBlock(nn.Module):
#     def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
#         super(MyTemporalBlock, self).__init__()
#         self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
#                                            stride=stride, padding=padding, dilation=dilation))

#         self.relu = nn.ReLU()
#         self.tanh = nn.Tanh()
#         self.dropout1 = nn.Dropout(dropout)

#         self.conv2 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
#                                            stride=stride, padding=padding, dilation=dilation))
#         self.sig = nn.Sigmoid()
#         self.dropout2 = nn.Dropout(dropout)

#         self.net1 = nn.Sequential(self.conv1, self.tanh, self.dropout1)
#         self.net2 = nn.Sequential(self.conv2, self.sig, self.dropout2)     

#         self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
#         self.init_weights()

#     def init_weights(self):
#         self.conv1.weight.data.normal_(0, 0.01)
#         self.conv2.weight.data.normal_(0, 0.01)
#         if self.downsample is not None:
#             self.downsample.weight.data.normal_(0, 0.01)

#     def forward(self, x):
#         out1 = self.net1(x)
#         out2 = self.net2(x)
#         out = torch.mul(out1, out2)
#         res = x if self.downsample is None else self.downsample(x)
#         return self.relu(out + res)



# class TemporalConvNet(nn.Module):
#     def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
#         super(TemporalConvNet, self).__init__()
#         layers = []
#         num_levels = len(num_channels)
#         for i in range(num_levels):
#             dilation_size = 2 ** i
#             in_channels = num_inputs if i == 0 else num_channels[i-1]
#             out_channels = num_channels[i]

#             layers += [MyTemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
#                                      padding=(kernel_size//2) * dilation_size, dropout=dropout)]

#         self.network = nn.Sequential(*layers)

#     def forward(self, x):
#         return self.network(x)
