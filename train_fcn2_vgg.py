#!/usr/bin/env python

import scipy as scp
import scipy.misc

import numpy as np
import logging
import tensorflow as tf
import sys

import fcn2_vgg
import utils

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.INFO,
                    stream=sys.stdout)

# Load images
ind_img = [2, 8, 10, 11, 12, 15]
num_img = len(ind_img)
img = np.zeros((num_img, 2016, 2016, 3))
lab = np.zeros((num_img, 2016, 2016, 1))
for i, ind in enumerate(ind_img):
    img[i, :, :, :] = 1.0 * scp.misc.imread("data/%02i_original.png" % ind)
    lab[i, :, :, 0] = scp.misc.imread("data/%02i_label_dna.png" % ind)[:, :, 0] // 255
    lab[i, :, :, 0] += 2 * (scp.misc.imread("data/%02i_label_nucleosome_new.png" % ind)[:, :, 0] // 255)

img_train = img[[0, 1, 2, 4], :, :, :]
lab_train = lab[[0, 1, 2, 4], :, :, :]
img_val = img[[3, 5], :, :, :]
lab_val = lab[[3, 5], :, :, :]
num_train = 4
num_val = 2

# Specify training parameter
num_iter = 1400
size_random_crop = [800, 800, 4]
starter_learning_rate = 1e-4
learning_decay_rate = 0.5
decay_every = 200
pred_every = 10
val_every = 10
save_every = 200
file_name_log_train = 'loss_train.csv'
file_name_log_val = 'loss_val.csv'
# Layers get trained progressively
# Define which layer will be trained beginning at which iteration
#   1: Originally fc layers, upscoring layers
#   2: Scoring layer pool 4
#   3: Scoring layer pool 3
#   4: Scoring layer pool 2
#   5: Scoring layer pool 1
#   6: Convolutional layers
train_step_start = [200, 400, 600, 800, 1000]

# Open output file
log_file_train = open(file_name_log_train, 'w', 1)
log_file_train.write('iteration,loss,accuracy\n')
log_file_val = open(file_name_log_val, 'w', 1)
log_file_val.write('iteration,loss,accuracy\n')

with tf.Session() as sess:
    # Preprocess images
    image = tf.placeholder(tf.float32)
    label = tf.placeholder(tf.int32)
    img_lab = tf.concat([image, tf.cast(label, tf.float32)], axis=2)
    img_lab = tf.random_crop(img_lab, size_random_crop)
    img_lab = tf.image.random_flip_left_right(img_lab)
    img_lab = tf.image.random_flip_up_down(img_lab)
    img_lab = tf.cond(tf.reshape(tf.random_uniform([1]), []) > 0.5, lambda: img_lab,
                      lambda: tf.image.transpose_image(img_lab))
    img_proc, lab_proc = tf.split(tf.expand_dims(img_lab, 0), [3, 1], 3)
    lab_proc = tf.cast(tf.reshape(lab_proc, tf.shape(lab_proc)[0:-1]), tf.int32)
    
    # Build network
    vgg_fcn = fcn2_vgg.FCN2VGG(vgg16_npy_path='vgg16.npy', load_semantic_net=False)
    with tf.name_scope("content_vgg"):
        vgg_fcn.build(img_proc, debug=True, num_classes=3, train=True)
    
    # Determine loss
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=lab_proc, logits=vgg_fcn.upscore32))
    
    # Declare optimizer
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, decay_every, learning_decay_rate,
                                               staircase=False)
    var_list_conv = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "conv*")
    var_list_fc = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "fc*")
    var_list_upscore = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "upscore*")
    var_list_fr = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "score_fr*")
    var_list_p4 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "score_pool4*")
    var_list_p3 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "score_pool3*")
    var_list_p2 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "score_pool2*")
    var_list_p1 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "score_pool1*")
    var_list1 = var_list_fc + var_list_upscore + var_list_fr
    train_step1 = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step, var_list=var_list1)
    var_list2 = var_list_fc + var_list_upscore + var_list_fr + var_list_p4
    train_step2 = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step, var_list=var_list2)
    var_list3 = var_list_fc + var_list_upscore + var_list_fr + var_list_p4 + var_list_p3
    train_step3 = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step, var_list=var_list3)
    var_list4 = var_list_fc + var_list_upscore + var_list_fr + var_list_p4 + var_list_p3 + var_list_p2
    train_step4 = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step, var_list=var_list4)
    var_list5 = var_list_fc + var_list_upscore + var_list_fr + var_list_p4 + var_list_p3 + var_list_p2 + var_list_p1
    train_step5 = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step, var_list=var_list5)
    var_list6 = var_list_fc + var_list_upscore + var_list_fr + var_list_p4 + var_list_p3 + var_list_p2 + var_list_p1 + var_list_conv
    train_step6 = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step, var_list=var_list6)
    
    print('Finished building Network.')
    
    logging.warning("Score weights are initialized random.")
    logging.warning("Do not expect meaningful results.")
    
    logging.info("Start Initializing Variabels.")
    
    init = tf.global_variables_initializer()
    sess.run(init)
    
    print('Running the Network')
    
    # Do FCN training
    for i in range(num_iter):
        if i < train_step_start[0]:
            train_step = train_step1
        elif i < train_step_start[1]:
            train_step = train_step2
        elif i < train_step_start[2]:
            train_step = train_step3
        elif i < train_step_start[3]:
            train_step = train_step4
        elif i < train_step_start[4]:
            train_step = train_step5
        else:
            train_step = train_step6
        if (i + 1) % pred_every == 0:
            ts_out, loss_np, pred, img_orig, lab_orig = sess.run(
                [train_step, loss, vgg_fcn.pred_up, img_proc, lab_proc],
                feed_dict={image: img_train[i % num_train, :, :, :], label: lab_train[i % num_train, :, :, :]})
            
            acc = np.mean(pred[0] == lab_orig[0])
            log_file_train.write(str(i + 1) + ',' + str(loss_np) + ',' + str(acc) + '\n')
            
            pred_color = utils.color_image(pred[0], num_classes=3)
            orig_color = utils.color_image(lab_orig[0], num_classes=3)
            
            scp.misc.imsave('train_pred_%04i.png' % (i + 1), pred_color)
            scp.misc.imsave('train_lab_orig_%04i.png' % (i + 1), orig_color)
            scp.misc.imsave('train_img_orig_%04i.png' % (i + 1), img_orig[0])
        else:
            ts_out = sess.run(train_step,
                              feed_dict={image: img_train[i % num_train, :, :, :],
                                         label: lab_train[i % num_train, :, :, :]})
        
        if (i + 1) % val_every == 0:
            loss_np, pred, img_orig, lab_orig = sess.run(
                [loss, vgg_fcn.pred_up, img_proc, lab_proc],
                feed_dict={image: img_val[(i // val_every) % num_val, :, :, :],
                           label: lab_val[(i // val_every) % num_val, :, :, :]})
            
            acc = np.mean(pred[0] == lab_orig[0])
            log_file_val.write(str(i + 1) + ',' + str(loss_np) + ',' + str(acc) + '\n')
            
            pred_color = utils.color_image(pred[0], num_classes=3)
            orig_color = utils.color_image(lab_orig[0], num_classes=3)
            
            scp.misc.imsave('val_pred_%04i.png' % (i + 1), pred_color)
            scp.misc.imsave('val_lab_orig_%04i.png' % (i + 1), orig_color)
            scp.misc.imsave('val_img_orig_%04i.png' % (i + 1), img_orig[0])
        
        if (i + 1) % save_every == 0:
            vgg_fcn.save(sess, file_name='vgg16_%04i.npy' % (i + 1))

# Close output files
log_file_train.close()
log_file_val.close()
