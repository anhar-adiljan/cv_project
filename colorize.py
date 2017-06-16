from __future__ import print_function
from scipy.misc import imread, imsave
from skimage import color
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import glob

learning_rate = 1e-3
training_epochs = 50
batch_size = 100
display_step = 5
# function that generates a single convolutional layer
# a - activation from the previous layer; w - weight; b - bias
def conv_layer(a, w, b):
	c = tf.nn.conv2d(a, w, strides=[1,1,1,1], padding='SAME')
	c = tf.nn.bias_add(c, b)
	c = tf.nn.relu(c)
	#c = tf.nn.max_pool(c, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
	return c

# function that generates a single fully connected layer
# a - activation from the previous layer; w - weight; b - bias
def fc_layer(a, w, b):
	f = tf.reshape(a, [-1,w.get_shape().as_list()[0]])
	f = tf.add(tf.matmul(f, w), b)
	f = tf.nn.relu(f)
	return f

# function that creates the model
def cnn(x, weights, biases):
	# 2 convolutional layers + 1 pooling layer
	c1 = conv_layer(x,  weights['c1'], biases['c1'])
	c2 = conv_layer(c1, weights['c2'], biases['c2'])
	#p1 = tf.nn.max_pool(c2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
	# 2 convolutional layers + 1 pooling layer
	c3 = conv_layer(c2, weights['c3'], biases['c3'])
	c4 = conv_layer(c3, weights['c4'], biases['c4'])
	#p2 = tf.nn.max_pool(c4, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
	# 2 convolutional layers + 1 pooling layer
	#c5 = conv_layer(c4, weights['c5'], biases['c5'])
	#c6 = conv_layer(c5, weights['c6'], biases['c6'])
	#p3 = tf.nn.max_pool(c6, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
	# 2 convolutional layers + 1 pooling layer
	#c7 = conv_layer(c6, weights['c7'], biases['c7'])
	#c8 = conv_layer(c7, weights['c8'], biases['c8'])
	#p4 = tf.nn.max_pool(c8, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
	# upscale layer
	#cup = tf.image.resize_bilinear(c8, (64,64))
	# 2 fully connected layers
	f1 = fc_layer(c4, weights['f1'], biases['f1'])
	f2 = fc_layer(f1, weights['f2'], biases['f2'])
	# output layer
	out = tf.add(tf.matmul(f2,weights['out']), biases['out'])
	out = tf.reshape(out, [tf.size(x) / (64 * 64), 64, 64, 2])
	return out

# dictionaries that store weights and biases by layers
weights = {
	'c1':  tf.Variable(tf.random_normal([3, 3, 1, 16])),
	'c2':  tf.Variable(tf.random_normal([3, 3, 16, 32])),
	'c3':  tf.Variable(tf.random_normal([3, 3, 32, 64])),
	'c4':  tf.Variable(tf.random_normal([3, 3, 64, 128])),
	#'c5':  tf.Variable(tf.random_normal([3, 3, 32, 64])),
	#'c6':  tf.Variable(tf.random_normal([3, 3, 64, 128])),
	#'c7':  tf.Variable(tf.random_normal([3, 3, 128, 128])),
	#'c8':  tf.Variable(tf.random_normal([3, 3, 128, 128])),
	'f1':  tf.Variable(tf.random_normal([128, 32])),
	'f2':  tf.Variable(tf.random_normal([32, 8])),
	'out': tf.Variable(tf.random_normal([8, 2]))
}

biases = {
	'c1':  tf.Variable(tf.random_normal([16])),
	'c2':  tf.Variable(tf.random_normal([32])),
	'c3':  tf.Variable(tf.random_normal([64])),
	'c4':  tf.Variable(tf.random_normal([128])),
	#'c5':  tf.Variable(tf.random_normal([64])),
	#'c6':  tf.Variable(tf.random_normal([128])),
	#'c7':  tf.Variable(tf.random_normal([128])),
	#'c8':  tf.Variable(tf.random_normal([128])),
	'f1':  tf.Variable(tf.random_normal([32])),
	'f2':  tf.Variable(tf.random_normal([8])),
	'out': tf.Variable(tf.random_normal([2]))
}
	
# tf graph input
x = tf.placeholder(tf.float32, [None, 64, 64, 1])
y = tf.placeholder(tf.float32, [None, 64, 64, 2])

# construct the model and define loss & optimizer
pred = cnn(x, weights, biases)

# define loss and optimizer
cost = tf.reduce_mean(tf.nn.l2_loss(y - pred))
opt = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# load the images
print('Loading training images...')
tr_path = 'cse190-data/train/*.png'
tr_imgs = []
for fn in glob.glob(tr_path):
	tr_imgs.append(imread(fn,mode='RGB'))
# convert training images to grayscale
tr_imgs = np.array(tr_imgs)
tr_imgs = color.rgb2luv(tr_imgs)
#tr_gray = color.rgb2gray(tr_imgs)
tr_n = len(tr_imgs)
tr_x = tr_imgs[:,:,:,0].reshape((tr_n, 64, 64, 1))
tr_y = tr_imgs[:,:,:,1:].reshape((tr_n, 64, 64, 2))
print('%d training images loaded!' % tr_n)

print('Loading test images...')
tst_path = 'cse190-data/test/*.png'
tst_imgs = []
for fn in glob.glob(tst_path):
	tst_imgs.append(imread(fn,mode='RGB'))
# convert test images to grayscale
tst_imgs = np.array(tst_imgs)
tst_imgs = color.rgb2luv(tst_imgs)
tst_n = len(tst_imgs)
tst_x = tst_imgs[:,:,:,0].reshape((tst_n, 64, 64, 1))
tst_y = tst_imgs[:,:,:,1:].reshape((tst_n, 64, 64, 2))
print('%d test images loaded!' % tst_n)

# initialize the variables
init = tf.global_variables_initializer()

# run the model
with tf.Session() as sess:
	sess.run(init)
	tr_losses = []
	tst_losses = []
	for epoch in range(training_epochs):
		avg_cost = 0.
		total_batch = int(tr_n / batch_size)
		for i in range(total_batch):
			batch_x = tr_x[i * batch_size : (i+1) * batch_size, :, :, :]
			batch_y = tr_y[i * batch_size : (i+1) * batch_size, :, :, :]
			# run back-propagation
			_, c = sess.run([opt, cost], feed_dict={x: batch_x, y: batch_y})
			avg_cost += c / total_batch
		# evaluate model
		tst_loss = sess.run(cost, feed_dict={x: tst_x, y: tst_y})
		tr_losses.append(avg_cost / (tr_n * 64 * 64))
		tst_losses.append(tst_loss / (tst_n * 64 * 64))
		print('Epoch:', '%04d' % epoch, \
			  'training loss:', '{:.3f}'.format(avg_cost), \
			  'test loss:', '{:.3f}'.format(tst_loss))
		if epoch % display_step == 0:
			# visualize colorization on training and test sets
			img_out = sess.run(pred, feed_dict={x: tst_x})
			img_out = np.concatenate((tst_x, img_out), axis=3)
			for i in range(tst_n):
				img = img_out[i]
				img = color.luv2rgb(img)
				fname = 'out/test_out' + str(i) + '_epoch' + str(epoch) + '.png'
				imsave(fname, img)
			img_out = sess.run(pred, feed_dict={x: tr_x})
			img_out = np.concatenate((tr_x, img_out), axis=3)
			for i in range(tr_n):
				img = img_out[i]
				img = color.luv2rgb(img)
				fname = 'out/tr_out' + str(i) + '_epoch' + str(epoch) + '.png'
				imsave(fname, img)
	print ("Optimization finished!")
	# plot error
	plt.figure(1)
	plt.plot(tr_losses, 'r')
	plt.plot(tst_losses, 'g')
	plt.savefig('error.png')