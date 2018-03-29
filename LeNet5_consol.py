#!/usr/bin/python 
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import LeNet5_operate
import sys

def main(cmd,argv):
	if cmd == 'train':
		mnist = input_data.read_data_sets("../datasets/MNIST_data", one_hot=True)
		LeNet5_operate.train(mnist,int(argv))
	elif cmd == 'eval':
		mnist = input_data.read_data_sets("../datasets/MNIST_data", one_hot=True)
		LeNet5_operate.evaluate(mnist)
	elif cmd == 'pred':
		LeNet5_operate.prediction(argv)
	else:
		print("train or eval or pred")

if __name__ == '__main__':
	args = len(sys.argv)
	if args < 2:
		print("train or eval or pred")
	elif args == 2:
		main(sys.argv[1], None)
	else:
		main(sys.argv[1], sys.argv[2])
