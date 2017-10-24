#-*- coding: utf-8 -*-
import sys
sys.path.append("./")
import tensorflow as tf
import Decoder

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
            dim_z=100,
            dim_y=10,
            dim_W1=1024,
            dim_W2=128,
            dim_W3=64,
            dim_F_I=512,
            dim_channel=1,
            ):

        self.batch_size = batch_size
        self.image_shape = image_shape
        self.dim_z = dim_z
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

        self.discrim_W1 = tf.Variable(tf.random_normal([5,5,dim_channel+dim_y,dim_W3], stddev=0.02), name='discrim_W1')
        self.discrim_W2 = tf.Variable(tf.random_normal([5,5,dim_W3+dim_y,dim_W2], stddev=0.02), name='discrim_W2')
        self.discrim_W3 = tf.Variable(tf.random_normal([dim_W2*7*7+dim_y,dim_W1], stddev=0.02), name='discrim_W3')
        self.discrim_W4 = tf.Variable(tf.random_normal([dim_W1+dim_y,1], stddev=0.02), name='discrim_W4')

        self.encoder_W1 = tf.Variable(tf.random_normal([5, 5, dim_channel , dim_W3], stddev=0.02),name='encoder_W1')
        self.encoder_W2 = tf.Variable(tf.random_normal([5, 5, dim_W3 , dim_W2], stddev=0.02), name='encoder_W2')
        self.encoder_W3 = tf.Variable(tf.random_normal([dim_W2 * 7 * 7 , dim_W1], stddev=0.02),name='encoder_W3')
        # self.encoder_W4 = tf.Variable(tf.random_normal([dim_W1 + dim_y, 1], stddev=0.02), name='encoder_W4')
        self.encoder_b1 = self.bias_variable([dim_W3],name='en_b1')
        self.encoder_b2 = self.bias_variable([dim_W2],name='en_b2')
        self.encoder_b3 = self.bias_variable([dim_W1],name='en_b3')


    def build_model(self):

        '''
         Y for class label
        '''
        # Z = tf.placeholder(tf.float32, [self.batch_size, self.dim_z])
        Y = tf.placeholder(tf.float32, [self.batch_size, self.dim_y])
        image_real = tf.placeholder(tf.float32, [self.batch_size]+self.image_shape)
        h_fc1 = self.encoder(image_real)

        #  F_V for variance representation
        #  F_I for identity representation
        F_I,F_V = tf.split(h_fc1, num_or_size_splits=2, axis = 1)
        h4 = self.generator(F_I,F_V)

        image_gen = tf.nn.sigmoid(h4)

        # raw_real = self.discriminate(image_real, Y)
        # p_real = tf.nn.sigmoid(raw_real)
        # raw_gen = self.discriminate(image_gen, Y)
        # p_gen = tf.nn.sigmoid(raw_gen)
        # discrim_cost_real = bce(raw_real, tf.ones_like(raw_real))
        # discrim_cost_gen = bce(raw_gen, tf.zeros_like(raw_gen))
        # discrim_cost = discrim_cost_real + discrim_cost_gen

        # gen_cost = bce( raw_gen, tf.ones_like(raw_gen))

        gen_vars = filter(lambda x: x.name.startswith('gen'), tf.trainable_variables())
        encoder_vars = filter(lambda x: x.name.startswith('encoder'), tf.trainable_variables())
        regularizer = tf.contrib.layers.l2_regularizer(0.1)
        gen_regularization_loss = tf.contrib.layers.apply_regularization(
            regularizer, weights_list= gen_vars + encoder_vars)
        gen_cost = (tf.nn.l2_loss(image_real - image_gen))/self.batch_size

        return Y, image_real, gen_cost, image_gen, gen_regularization_loss

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

    def discriminate(self, image, Y):
        yb = tf.reshape(Y, tf.stack([self.batch_size, 1, 1, self.dim_y]))
        X = tf.concat(axis=3, values=[image, yb*tf.ones([self.batch_size, 28, 28, self.dim_y])])

        h1 = lrelu( tf.nn.conv2d( X, self.discrim_W1, strides=[1,2,2,1], padding='SAME' ))
        h1 = tf.concat(axis=3, values=[h1, yb*tf.ones([self.batch_size, 14, 14, self.dim_y])])

        h2 = lrelu( batchnormalize( tf.nn.conv2d( h1, self.discrim_W2, strides=[1,2,2,1], padding='SAME')) )
        h2 = tf.reshape(h2, [self.batch_size, -1])
        h2 = tf.concat(axis=1, values=[h2, Y])

        h3 = lrelu( batchnormalize( tf.matmul(h2, self.discrim_W3 ) ))
        h3 = tf.concat(axis=1, values=[h3, Y])
        
        h4 = lrelu(batchnormalize(tf.matmul(h3,self.discrim_W4)))
        
        return h4

    def generator(self, F_I,F_V):

        F_combine = tf.concat(axis=1, values=[F_I,F_V])
        h1 = tf.nn.relu(batchnormalize(tf.matmul(F_combine, self.gen_W1)))
        h2 = tf.nn.relu(batchnormalize(tf.matmul(h1, self.gen_W2)))
        h2 = tf.reshape(h2, [self.batch_size,7,7,self.dim_W2])

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
