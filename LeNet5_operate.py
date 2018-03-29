#!/usr/bin/ython 
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import LeNet5_infernece
import os
import sys
import numpy as np

x = None
y = None
g = None
sess = None

# #### 1. 定义神经网络相关的参数

BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.01
LEARNING_RATE_DECAY = 0.99
REGULARIZATION_RATE = 0.0001
MOVING_AVERAGE_DECAY = 0.99
MODEL_SAVE_PATH = "LeNet5_model/"
MODEL_NAME = "LeNet5_model"


# #### 2. 定义训练过程

def train(mnist, training_steps):
    # 定义输出为4维矩阵的placeholder
    x = tf.placeholder(tf.float32, [
            BATCH_SIZE,
            LeNet5_infernece.IMAGE_SIZE,
            LeNet5_infernece.IMAGE_SIZE,
            LeNet5_infernece.NUM_CHANNELS],
        name='x-input')
    y_ = tf.placeholder(tf.float32, [None, LeNet5_infernece.OUTPUT_NODE], name='y-input')
    
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    y = LeNet5_infernece.inference(x,False,regularizer)
    global_step = tf.Variable(0, trainable=False)

    # 定义损失函数、学习率、滑动平均操作以及训练过程。
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        mnist.train.num_examples / BATCH_SIZE, LEARNING_RATE_DECAY,
        staircase=True)

    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    with tf.control_dependencies([train_step, variables_averages_op]):
        train_op = tf.no_op(name='train')
        
    # 初始化TensorFlow持久化类。
    saver = tf.train.Saver()
    with tf.Session() as sess:
		ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
		if ckpt and ckpt.model_checkpoint_path:
			saver.restore(sess, ckpt.model_checkpoint_path)
		else:
			tf.global_variables_initializer().run()
        
		for i in range(training_steps):
			for j in range(10):
				xs, ys = mnist.train.next_batch(BATCH_SIZE)
				reshaped_xs = np.reshape(xs, (
					BATCH_SIZE,
					LeNet5_infernece.IMAGE_SIZE,
					LeNet5_infernece.IMAGE_SIZE,
					LeNet5_infernece.NUM_CHANNELS))
				_, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: reshaped_xs, y_: ys})
			print("After %d training step(s), loss on training batch is %g." % (step, loss_value))
			saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)

# #### 3. 定义验证过程
def evaluate(mnist):
    with tf.Graph().as_default() as g:
		x = tf.placeholder(tf.float32, 
				[500, 
				LeNet5_infernece.IMAGE_SIZE, 
				LeNet5_infernece.IMAGE_SIZE, 
				LeNet5_infernece.NUM_CHANNELS], 
				name='x-input')
		y_ = tf.placeholder(tf.float32, [None, LeNet5_infernece.OUTPUT_NODE], name='y-input')
				
		y = LeNet5_infernece.inference(x, False, None)
		correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
		saver = tf.train.Saver()
		with tf.Session() as sess:
			ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
			if ckpt and ckpt.model_checkpoint_path:
				saver.restore(sess, ckpt.model_checkpoint_path)
				global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
				index = 0
				accu = []
				while index < 5000:
					xs = mnist.validation.images[index:index+500]
					ys = mnist.validation.labels[index:index+500]
					index +=500
					reshaped_xs = np.reshape(xs, (500, LeNet5_infernece.IMAGE_SIZE, LeNet5_infernece.IMAGE_SIZE, LeNet5_infernece.NUM_CHANNELS))
					validate_feed = {x: reshaped_xs, y_: ys}
					accuracy_score = sess.run(accuracy, feed_dict=validate_feed)
					accu.append(accuracy_score)
					print("validation accuracy = %g"%accuracy_score)
				print("validation mean accuracy = %g"%np.mean(accu))
			else:
				print("no checkpoint")

# #### 4. 定义预测初始化
def prediction_init():
	global x,y,g,sess
	g = tf.Graph().as_default()
	sess = tf.Session()
	x = tf.placeholder(tf.float32, 
			[1,
			LeNet5_infernece.IMAGE_SIZE, 
			LeNet5_infernece.IMAGE_SIZE, 
			LeNet5_infernece.NUM_CHANNELS], 
			name='x-input')
				
	y = LeNet5_infernece.inference(x, False, None)
	saver = tf.train.Saver()
	ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
	if ckpt and ckpt.model_checkpoint_path:
		saver.restore(sess, ckpt.model_checkpoint_path)
		return True
	return False


# #### 5. 定义快速预测
def prediction_fast(buff, shape):
	global x,y,g,sess
	xfeed = np.zeros([1,28,28,1])
	if shape == '280X280X3':
		for ibase in np.arange(28):
			for jbase in np.arange(28):
				sum = 0
				for ii in np.arange(10):
					for jj in np.arange(10):
						for kk in np.arange(3):
							sum += ord(buff[(279-ibase*10-ii)*280*3+(jbase*10+jj)*3+kk])
				xfeed[0][ibase][jbase][0] = 1-float(sum)/300/255
	elif shape == '280X280':
		for ibase in np.arange(28):
			for jbase in np.arange(28):
				sum = 0
				for ii in np.arange(10):
					for jj in np.arange(10):
							sum += ord(buff[(279-ibase*10-ii)*280+(jbase*10+jj)])
				xfeed[0][ibase][jbase][0] = 1-float(sum)/100/255

	yy = sess.run(y, feed_dict={x: xfeed})
	pred = np.argmax(yy,1)
	t1 = yy[0][pred[0]]
				
	yy[0][pred[0]] = 0
	pred2 = np.argmax(yy,1)
	t2 = yy[0][pred2[0]]
		
	if t2 <= 0:
		return pred[0], 1.0
	else:
		return pred[0], (t1-t2)/t1


# #### 6. 定义预测
def prediction(filename):
    with tf.Graph().as_default() as g:
		x = tf.placeholder(tf.float32, 
				[1, 
				LeNet5_infernece.IMAGE_SIZE, 
				LeNet5_infernece.IMAGE_SIZE, 
				LeNet5_infernece.NUM_CHANNELS], 
				name='x-input')
				
		y = LeNet5_infernece.inference(x, False, None)
		prediction = tf.argmax(y, 1)
		saver = tf.train.Saver()
		with tf.Session() as sess:
			ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
			if ckpt and ckpt.model_checkpoint_path:
				saver.restore(sess, ckpt.model_checkpoint_path)
				xfeed = np.zeros([1,28,28,1])
				f = open(filename, 'rb+')
				buf = f.read(235254)
				f.close()
				for ibase in np.arange(28):
					for jbase in np.arange(28):
						sum = 0
						for ii in np.arange(10):
							for jj in np.arange(10):
								for kk in np.arange(3):
									sum += ord(buf[54+(279-ibase*10-ii)*280*3+(jbase*10+jj)*3+kk])
						xfeed[0][ibase][jbase][0] = 1-float(sum)/300/255
				prediction_feed = {x: xfeed}
				yy = sess.run(y, feed_dict=prediction_feed)
				
				pred = np.argmax(yy,1)
				t1 = yy[0][pred[0]]
				
				yy[0][pred[0]] = 0
				pred2 = np.argmax(yy,1)
				t2 = yy[0][pred2[0]]
		
		if t2 <= 0:
			return pred[0], 1.0
		else:
			return pred[0], (t1-t2)/t1
