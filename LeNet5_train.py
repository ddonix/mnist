
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import LeNet5_infernece
import os
import sys
import numpy as np


# #### 1. 定义神经网络相关的参数

# In[2]:


BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.01
LEARNING_RATE_DECAY = 0.99
REGULARIZATION_RATE = 0.0001
TRAINING_STEPS = 60
MOVING_AVERAGE_DECAY = 0.99
MODEL_SAVE_PATH = "LeNet5_model/"
MODEL_NAME = "LeNet5_model"


# #### 2. 定义训练过程

# In[3]:


def train(mnist):
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
        tf.global_variables_initializer().run()
        for i in range(TRAINING_STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)

            reshaped_xs = np.reshape(xs, (
                BATCH_SIZE,
                LeNet5_infernece.IMAGE_SIZE,
                LeNet5_infernece.IMAGE_SIZE,
                LeNet5_infernece.NUM_CHANNELS))
            _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: reshaped_xs, y_: ys})

            if i % 10 == 0:
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
# #### 4. 主程序入口

# In[5]:


def main(argv):
	if argv == 'train':
		mnist = input_data.read_data_sets("./datasets/MNIST_data", one_hot=True)
		train(mnist)
	elif argv == 'eval':
		mnist = input_data.read_data_sets("./datasets/MNIST_data", one_hot=True)
		evaluate(mnist)
	else:
		print("train or eval")

if __name__ == '__main__':
	if len(sys.argv) != 2:
		print("train or eval")
	else:
		main(sys.argv[1])
