import os
import numpy as np
from model import *
from util import *
from load import mnist_with_valid_set

n_epochs = 100
learning_rate = 0.0002
batch_size = 128
image_shape = [28,28,1]
dim_z = 100
dim_W1 = 1024
dim_W2 = 128
dim_W3 = 64
dim_channel = 1
gen_regularizer_weight = 0.01
visualize_dim=128

# train image validation image, test image, train label, validation label, test label
trX, vaX, teX, trY, vaY, teY = mnist_with_valid_set()

dcgan_model = VDSN(
        batch_size=batch_size,
        image_shape=image_shape,
        dim_z=dim_z,
        dim_W1=dim_W1,
        dim_W2=dim_W2,
        dim_W3=dim_W3,
        )


Y_tf, image_tf, g_cost_tf, image_gen, gen_reg_cost_tf = dcgan_model.build_model()
sess = tf.InteractiveSession()
saver = tf.train.Saver(max_to_keep=10)

discrim_vars = filter(lambda x: x.name.startswith('discrim'), tf.trainable_variables())
gen_vars = filter(lambda x: x.name.startswith('gen'), tf.trainable_variables())
encoder_vars = filter(lambda x: x.name.startswith('encoder'), tf.trainable_variables())
# gen_regularizer_cost = gen_regularizer_weight * (tf.nn.l2_loss(gen_vars) + tf.nn.l2_loss(encoder_vars) )

discrim_vars = [i for i in discrim_vars]
gen_vars = [i for i in gen_vars]
encoder_vars = [i for i in encoder_vars]
gen_loss = g_cost_tf + gen_regularizer_weight * gen_reg_cost_tf
# train_op_discrim = tf.train.AdamOptimizer(learning_rate, beta1=0.5).minimize(d_cost_tf, var_list=discrim_vars)
train_op_gen = tf.train.AdamOptimizer(learning_rate, beta1=0.5).minimize(gen_loss, var_list=gen_vars + encoder_vars)

# Z_tf_sample, Y_tf_sample, image_tf_sample = dcgan_model.samples_generator(batch_size=visualize_dim)

tf.global_variables_initializer().run()

# Z_np_sample = np.random.uniform(-1, 1, size=(visualize_dim,dim_z))
# Y_np_sample = OneHot( np.random.randint(10, size=[visualize_dim]))
iterations = 0
k = 2

step = 20

for epoch in range(n_epochs):
    index = np.arange(len(trY))
    np.random.shuffle(index)
    trX = trX[index]
    trY = trY[index]

    for start, end in zip(
            range(0, len(trY), batch_size),
            range(batch_size, len(trY), batch_size)
            ):

        # pixel value normalized -> from 0 to 1
        Xs = trX[start:end].reshape( [-1, 28, 28, 1]) / 255.
        Ys = OneHot(trY[start:end],10)

        # if np.mod( iterations, k ) != 0:
        _, gen_loss_val, gen_reg_val = sess.run(
                [train_op_gen, g_cost_tf, gen_reg_cost_tf],
                feed_dict={
                    Y_tf:Ys,
                    image_tf:Xs
                    })
        # discrim_loss_val, p_real_val, p_gen_val = sess.run([d_cost_tf,p_real,p_gen], feed_dict={image_tf:Xs, Y_tf:Ys})
        print("=========== updating G ==========")
        print("iteration:", iterations)
        print("gen loss:", gen_loss_val)
        print("total gen loss:", gen_loss_val + gen_regularizer_weight * gen_reg_val)
        # print("discrim loss:", discrim_loss_val)

        # else:
        # _, discrim_loss_val = sess.run(
        #         [train_op_discrim, d_cost_tf],
        #         feed_dict={
        #             Z_tf:Zs,
        #             Y_tf:Ys,
        #             image_tf:Xs
        #             })
        # gen_loss_val, p_real_val, p_gen_val = sess.run([g_cost_tf, p_real, p_gen], feed_dict={Z_tf:Zs, image_tf:Xs, Y_tf:Ys})
        # print("=========== updating D ==========")
        # print("iteration:", iterations)
        # print("gen loss:", gen_loss_val)
        # print("discrim loss:", discrim_loss_val)

        # print("Average P(real)=", p_real_val.mean())
        # print("Average P(gen)=", p_gen_val.mean())

        if np.mod(iterations, step) == 0:
            generated_samples = sess.run(
                    image_gen,
                    feed_dict={
                        image_tf:vaX[0:visualize_dim].reshape( [-1, 28, 28, 1]) / 255
                        })
            # generated_samples = (generated_samples + 1.)/2.
            save_visualization(generated_samples, (14,14), save_path='./vis/sample_%04d.jpg' % int(iterations/step))

        iterations += 1

