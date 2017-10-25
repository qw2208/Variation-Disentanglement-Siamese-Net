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
dis_regularizer_weight = 0.01
gen_disentangle_weight = 1
visualize_dim=128

# train image validation image, test image, train label, validation label, test label
trX, vaX, teX, trY, vaY, teY = mnist_with_valid_set()

VDSN_model = VDSN(
        batch_size=batch_size,
        image_shape=image_shape,
        dim_z=dim_z,
        dim_W1=dim_W1,
        dim_W2=dim_W2,
        dim_W3=dim_W3,
        )


Y_tf, image_tf_real_left, image_tf_real_right, g_recon_cost_tf_left, g_recon_cost_tf_right, gen_disentangle_cost_tf_left, \
    gen_disentangle_cost_tf_right, dis_cost_tf_left, dis_cost_tf_right, image_gen_left, image_gen_right, gen_reg_cost_tf, \
    dis_reg_cost_tf, dis_max_prediction_tf_left, dis_max_prediction_tf_right = VDSN_model.build_model()
sess = tf.InteractiveSession()
saver = tf.train.Saver(max_to_keep=10)

discrim_vars = filter(lambda x: x.name.startswith('dis'), tf.trainable_variables())
gen_vars = filter(lambda x: x.name.startswith('gen'), tf.trainable_variables())

# include en_* and encoder_* W and b,
encoder_vars = filter(lambda x: x.name.startswith('en'), tf.trainable_variables())
# gen_regularizer_cost = gen_regularizer_weight * (tf.nn.l2_loss(gen_vars) + tf.nn.l2_loss(encoder_vars) )

discrim_vars = [i for i in discrim_vars]
gen_vars = [i for i in gen_vars]
encoder_vars = [i for i in encoder_vars]
gen_loss_left = g_recon_cost_tf_left + gen_disentangle_weight * gen_disentangle_cost_tf_left
gen_loss_right = g_recon_cost_tf_right + gen_disentangle_weight * gen_disentangle_cost_tf_right 
gen_loss = gen_loss_left + gen_loss_right + gen_regularizer_weight * gen_reg_cost_tf
dis_loss_left = dis_cost_tf_left
dis_loss_right = dis_cost_tf_right
dis_loss = dis_loss_right + dis_loss_left + dis_regularizer_weight * dis_reg_cost_tf

train_op_discrim = tf.train.AdamOptimizer(
    learning_rate, beta1=0.5).minimize(dis_loss, var_list=discrim_vars)
train_op_gen = tf.train.AdamOptimizer(
    learning_rate, beta1=0.5).minimize(gen_loss, var_list=gen_vars + encoder_vars)

# Z_tf_sample, Y_tf_sample, image_tf_sample = dcgan_model.samples_generator(batch_size=visualize_dim)

tf.global_variables_initializer().run()

# Z_np_sample = np.random.uniform(-1, 1, size=(visualize_dim,dim_z))
# Y_np_sample = OneHot( np.random.randint(10, size=[visualize_dim]))
iterations = 0
k = 2

step = 20

def randomPickRight(start, end, trX, trY, indexTable):
    randomList = []
    for i in range(start, end):
	while True:
            randomPick = np.random.choice(indexTable[trY[i]], 1)[0]
	    if randomPick == i:
		continue
	    else:
		randomList.append(randomPick)
		break
    print randomList
    return trX[randomList]


for epoch in range(n_epochs):
    index = np.arange(len(trY))
    np.random.shuffle(index)
    trX = trX[index]
    trY = trY[index]

    indexTable = [[] for i in range(10)]
    for index in range(len(trY)):
        indexTable[trY[index]].append(index)

    for start, end in zip(
            range(0, len(trY), batch_size),
            range(batch_size, len(trY), batch_size)
            ):

        # pixel value normalized -> from 0 to 1
        Xs_left = trX[start:end].reshape( [-1, 28, 28, 1]) / 255.
        Ys = OneHot(trY[start:end],10)

        Xs_right = randomPickRight(start, end, trX, trY, indexTable).reshape( [-1, 28, 28, 1]) / 255.

        if np.mod( iterations, k ) != 0:
            _, gen_loss_val_left, gen_loss_val_right, gen_disentangle_val_left, gen_disentangle_val_right, gen_reg_val, \
                    dis_max_prediction_tf_left, dis_max_prediction_tf_right = sess.run(
                    [train_op_gen, g_recon_cost_tf_left, g_recon_cost_tf_right, gen_disentangle_cost_tf_left, \
                    gen_disentangle_cost_tf_right, gen_reg_cost_tf, dis_max_prediction_tf_left, dis_max_prediction_tf_right],
                    feed_dict={
                        Y_tf:Ys,
                        image_tf_real_left: Xs_left,
                        image_tf_real_right: Xs_right 
                        })
            gen_loss_val = float(gen_loss_val_left + gen_loss_val_right) / 2
            dis_max_prediction_val = float(dis_max_prediction_tf_left + dis_max_prediction_tf_right) / 2
            print("=========== updating G ==========")
            print("iteration:", iterations)
            print("gen reconstruction loss:", gen_loss_val)
            print("gen disentanglement loss :", gen_loss_val)
            print("total gen loss:", gen_loss_val +
                  gen_disentangle_weight * gen_loss_val + gen_regularizer_weight * gen_reg_val)
            print("discrim correct prediction :", dis_max_prediction_val)

        else:
            _, discrim_loss_val_left, discrim_loss_val_right, discrim_reg_loss_val,\
                    dis_max_prediction_val_left, dis_max_prediction_val_right = sess.run(
                    [train_op_discrim, dis_cost_tf_left, dis_cost_tf_right, dis_reg_cost_tf, \
                    dis_max_prediction_tf_left, dis_max_prediction_tf_right],
                    feed_dict={
                        Y_tf:Ys,
                        image_tf_real_left: Xs_left,
                        image_tf_real_right: Xs_right
                        })

            discrim_loss_val = float(discrim_loss_val_right + discrim_loss_val_left) / 2
            dis_max_prediction_val = float(dis_max_prediction_val_left + dis_max_prediction_val_right) / 2
            print("=========== updating D ==========")
            print("iteration:", iterations)
            print("discriminator loss:", discrim_loss_val)
            print("discriminator total loss:", discrim_loss_val + dis_regularizer_weight * discrim_reg_loss_val)
            print("discrim correct prediction :", dis_max_prediction_val)


        if np.mod(iterations, step) == 0:
            indexTableVal = [[] for i in range(10)]
            for index in range(len(vaY)):
                indexTableVal[vaY[index]].append(index)
            corrRightVal = randomPickRight(0, visualize_dim, vaX, vaY, indexTableVal)
            generated_samples_left = sess.run(
                    image_gen_left,
                    feed_dict={
                        image_tf_real_left: vaX[0:visualize_dim].reshape( [-1, 28, 28, 1]) / 255, 
                        image_tf_real_right: corrRightVal.reshape( [-1, 28, 28, 1]) / 255
                        })
            # generated_samples = (generated_samples + 1.)/2.
            save_visualization(generated_samples_left, (14,14), save_path='./vis/sample_%04d.jpg' % int(iterations/step))

        iterations += 1

