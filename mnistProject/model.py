#-*- coding: utf-8 -*-
import sys
sys.path.append("./")
import tensorflow as tf
import numpy as np

def batchnormalize(X, eps=1e-8, g=None, b=None):
    if X.get_shape().ndims == 4:
        mean = tf.reduce_mean(X, [0,1,2])
        std = tf.reduce_mean( tf.square(X-mean), [0,1,2] )
        X = (X-mean) / tf.sqrt(std+eps)

        if g is not None and b is not None:
            g = tf.reshape(g, [1,1,1,-1])
            b = tf.reshape(b, [1,1,1,-1])
            X = X*g + b

    elif X.get_shape().ndims == 2:
        mean = tf.reduce_mean(X, 0)
        std = tf.reduce_mean(tf.square(X-mean), 0)
        X = (X-mean) / tf.sqrt(std+eps)

        if g is not None and b is not None:
            g = tf.reshape(g, [1,-1])
            b = tf.reshape(b, [1,-1])
            X = X*g + b

    else:
        raise NotImplementedError

    return X

def lrelu(X, leak=0.2):
    f1 = 0.5 * (1 + leak)
    f2 = 0.5 * (1 - leak)
    return f1 * X + f2 * tf.abs(X)

def bce(o, t):
    o = tf.clip_by_value(o, 1e-7, 1. - 1e-7)
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=o, labels=t))

class VDSN():
    def __init__(
            self,
            batch_size=100,
            image_shape=[28,28,1],
            dim_y=10,
            dim_W1=1024,
            dim_W2=128,
            dim_W3=64,
            dim_F_I=512,
            dim_channel=1,
            ):

        self.batch_size = batch_size
        self.image_shape = image_shape
        self.dim_y = dim_y
        self.dim_F_I = 512
        self.dim_F_V = dim_W1 - dim_F_I
        self.dim_W1 = dim_W1
        self.dim_W2 = dim_W2
        self.dim_W3 = dim_W3
        self.dim_channel = dim_channel

        self.gen_W1 = tf.Variable(tf.random_normal([dim_W1, dim_W1], stddev=0.02), name='gen_W1')
        self.gen_W2 = tf.Variable(tf.random_normal([dim_W1, dim_W2*7*7], stddev=0.02), name='gen_W2')
        self.gen_W3 = tf.Variable(tf.random_normal([5,5,dim_W3,dim_W2], stddev=0.02), name='gen_W3')
        self.gen_W4 = tf.Variable(tf.random_normal([5,5,dim_channel,dim_W3], stddev=0.02), name='gen_W4')

        self.discrim_W1 = tf.Variable(tf.random_normal([self.dim_F_V, self.dim_F_V], stddev=0.02), name='discrim_W1')
        self.discrim_W2 = tf.Variable(tf.random_normal([self.dim_F_V, self.dim_y], stddev=0.02), name='discrim_W2')
        self.discrim_b1 = self.bias_variable([self.dim_F_V], name='dis_b1')
        self.discrim_b2 = self.bias_variable([self.dim_y], name='dis_b2')

        self.encoder_W1 = tf.Variable(tf.random_normal([5, 5, dim_channel , dim_W3], stddev=0.02),name='encoder_W1')
        self.encoder_W2 = tf.Variable(tf.random_normal([5, 5, dim_W3 , dim_W2], stddev=0.02), name='encoder_W2')
        self.encoder_W3 = tf.Variable(tf.random_normal([dim_W2 * 7 * 7 , dim_W1], stddev=0.02),name='encoder_W3')
        self.encoder_b1 = self.bias_variable([dim_W3],name='en_b1')
        self.encoder_b2 = self.bias_variable([dim_W2],name='en_b2')
        self.encoder_b3 = self.bias_variable([dim_W1],name='en_b3')


    def build_model(self, gen_disentangle_weight=1, gen_regularizer_weight=1, dis_regularizer_weight=1):

        '''
         Y for class label
        '''
        Y = tf.placeholder(tf.float32, [None, self.dim_y])

        image_real_left = tf.placeholder(tf.float32, [None] + self.image_shape)
        image_real_right = tf.placeholder(tf.float32, [None] + self.image_shape)
        h_fc1_left = self.encoder(image_real_left)
        h_fc1_right = self.encoder(image_real_right)

        #  F_V for variance representation
        #  F_I for identity representation
        F_I_left, F_V_left = tf.split(h_fc1_left, num_or_size_splits=2, axis = 1)
        F_I_right, F_V_right = tf.split(h_fc1_right, num_or_size_splits=2, axis = 1)
        h4_right = self.generator(F_I_left, F_V_right)
        h4_left = self.generator(F_I_right, F_V_left)

        image_gen_left = tf.nn.sigmoid(h4_left)
        image_gen_right = tf.nn.sigmoid(h4_right)

        Y_prediction_left = self.discriminate(F_V_left)
        Y_prediction_right = self.discriminate(F_V_right)

        Y_result_left = tf.reduce_sum(Y * tf.nn.softmax(Y_prediction_left), axis=1)
        Y_result_right = tf.reduce_sum(Y * tf.nn.softmax(Y_prediction_right), axis=1)

        dis_max_prediction_left = [tf.reduce_max(Y_result_left), tf.reduce_mean(Y_result_left), tf.reduce_min(Y_result_left)];
        dis_max_prediction_right = [tf.reduce_max(Y_result_right), tf.reduce_mean(Y_result_right), tf.reduce_min(Y_result_right)];

        gen_vars = filter(lambda x: x.name.startswith('gen'), tf.trainable_variables())
        encoder_vars = filter(lambda x: x.name.startswith('encoder'), tf.trainable_variables())
        discriminator_vars = filter(lambda x: x.name.startswith('discrim'), tf.trainable_variables())

        regularizer = tf.contrib.layers.l2_regularizer(0.1)
        gen_regularization_loss = tf.contrib.layers.apply_regularization(
            regularizer, weights_list= gen_vars + encoder_vars)
        dis_regularization_loss = tf.contrib.layers.apply_regularization(
            regularizer, weights_list=discriminator_vars)
        
        gen_recon_cost_left = tf.nn.l2_loss(image_real_left - image_gen_left) / self.batch_size
        gen_recon_cost_right = tf.nn.l2_loss(image_real_left - image_gen_left) / self.batch_size
        gen_disentangle_cost_left = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=1-Y, logits=Y_prediction_left))
        gen_disentangle_cost_right = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=1-Y, logits=Y_prediction_right))
        dis_loss_left = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=Y_prediction_left))
        dis_loss_right = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=Y_prediction_right))

        gen_recon_cost = (gen_recon_cost_left + gen_recon_cost_right) / 2
        gen_disentangle_cost = (gen_disentangle_cost_left + gen_disentangle_cost_right) / 2
        gen_total_cost = gen_recon_cost + gen_disentangle_weight * gen_disentangle_cost + gen_regularizer_weight * gen_regularization_loss
        dis_cost_tf = (dis_loss_left + dis_loss_right) / 2
        dis_total_cost_tf = dis_cost_tf + dis_regularizer_weight * dis_regularization_loss

        tf.summary.scalar('gen_recon_cost', gen_recon_cost)
        tf.summary.scalar('gen_disentangle_cost', gen_disentangle_cost)
        tf.summary.scalar('gen_total_cost', gen_total_cost)
        tf.summary.scalar('dis_cost_tf', dis_cost_tf)
        tf.summary.scalar('dis_total_cost_tf', dis_total_cost_tf)

        return Y, image_real_left, image_real_right, gen_recon_cost, gen_disentangle_cost, gen_total_cost, \
               dis_cost_tf, dis_total_cost_tf, image_gen_left, image_gen_right, \
               dis_max_prediction_left, dis_max_prediction_right

    def encoder(self, image):

        # First convolutional layer - maps one grayscale image to 64 feature maps.
        with tf.name_scope('encoder_conv1'):
            h_conv1 = lrelu(tf.nn.conv2d(image, self.encoder_W1, strides=[1, 1, 1, 1], padding='SAME') + self.encoder_b1)
        # First pooling layer - downsamples by 2X.
        with tf.name_scope('encoder_pool1'):
            h_pool1 = self.max_pool_2x2(h_conv1)

        # Second convolutional layer -- maps 64 feature maps to 128.
        with tf.name_scope('encoder_conv2'):
            h_conv2 = tf.nn.conv2d(h_pool1, self.encoder_W2, strides=[1, 1, 1, 1], padding='SAME')+self.encoder_b2
            h_conv2 = lrelu(batchnormalize(h_conv2))

        # Second pooling layer.
        with tf.name_scope('encoder_pool2'):
            h_pool2 = self.max_pool_2x2(h_conv2)

        # Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
        # is down to 7x7x64 feature maps -- maps this to 1024 features.

        with tf.name_scope('encoder_fc1'):
            h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 128])
            h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, self.encoder_W3) + self.encoder_b3)

        return h_fc1

    def discriminate(self, F_V):
        # 512 to 512
        h1 = lrelu( batchnormalize(tf.matmul(F_V, self.discrim_W1) + self.discrim_b1))
        # 512 to 10
        h2 = lrelu( batchnormalize(tf.matmul(h1, self.discrim_W2) + self.discrim_b2))
        return h2

    def generator(self, F_I,F_V):

        F_combine = tf.concat(axis=1, values=[F_I,F_V])
        h1 = tf.nn.relu(batchnormalize(tf.matmul(F_combine, self.gen_W1)))
        h2 = tf.nn.relu(batchnormalize(tf.matmul(h1, self.gen_W2)))
        h2 = tf.reshape(h2, [-1,7,7,self.dim_W2])

        output_shape_l3 = [self.batch_size,14,14,self.dim_W3]
        h3 = tf.nn.conv2d_transpose(h2, self.gen_W3, output_shape=output_shape_l3, strides=[1,2,2,1])
        h3 = tf.nn.relu( batchnormalize(h3) )
        output_shape_l4 = [self.batch_size,28,28,self.dim_channel]
        h4 = tf.nn.conv2d_transpose(h3, self.gen_W4, output_shape=output_shape_l4, strides=[1,2,2,1])
        return h4

    # def samples_generator(self, batch_size):
    #     Z = tf.placeholder(tf.float32, [batch_size, self.dim_z])
    #     Y = tf.placeholder(tf.float32, [batch_size, self.dim_y])
    #
    #     yb = tf.reshape(Y, [batch_size, 1, 1, self.dim_y])
    #     Z_ = tf.concat(axis=1, values=[Z,Y])
    #     h1 = tf.nn.relu(batchnormalize(tf.matmul(Z_, self.gen_W1)))
    #     h1 = tf.concat(axis=1, values=[h1, Y])
    #     h2 = tf.nn.relu(batchnormalize(tf.matmul(h1, self.gen_W2)))
    #     h2 = tf.reshape(h2, [batch_size,7,7,self.dim_W2])
    #     h2 = tf.concat(axis=3, values=[h2, yb*tf.ones([batch_size, 7, 7, self.dim_y])])
    #
    #     output_shape_l3 = [batch_size,14,14,self.dim_W3]
    #     h3 = tf.nn.conv2d_transpose(h2, self.gen_W3, output_shape=output_shape_l3, strides=[1,2,2,1])
    #     h3 = tf.nn.relu( batchnormalize(h3) )
    #     h3 = tf.concat(axis=3, values=[h3, yb*tf.ones([batch_size, 14,14,self.dim_y])] )
    #
    #     output_shape_l4 = [batch_size,28,28,self.dim_channel]
    #     h4 = tf.nn.conv2d_transpose(h3, self.gen_W4, output_shape=output_shape_l4, strides=[1,2,2,1])
    #     x = tf.nn.sigmoid(h4)
    #     return Z,Y,x

    def bias_variable(self, shape, name=None):
      """bias_variable generates a bias variable of a given shape."""
      initial = tf.constant(0.1, shape=shape, name=name)
      return tf.Variable(initial)

    def max_pool_2x2(self, x):
        """max_pool_2x2 downsamples a feature map by 2X."""
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1], padding='SAME')
