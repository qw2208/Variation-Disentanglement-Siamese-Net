import numpy as np

'''
This function would train a classifier on top of the representation F_V,
make sure it cannot train out the Identity
'''

def validate_F_V_classification_fail(conf):

    train_op = tf.train.AdamOptimizer(
        learning_rate, beta1=0.5).minimize(dis_total_cost_tf, var_list=discrim_vars, global_step=global_step)