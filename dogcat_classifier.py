import random
from random import shuffle
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
import matplotlib.cm as cm
import pandas as pd
import numpy as np
import os
from tqdm import tqdm


train_dir = './dogs_cats/train/'
test_dir = './dogs_cats/test/'
model_save_dir = './model'
img_size = 160
channels = 3    # 1 or 3
img_class = 2  # convert to one-hot
batch_size = 35
lr = 1e-4


# output: 2-dimension column vectors (one-hot)
def pic_label(filename):
    label = filename.split('.')[0]
    if label == 'cat':
        return [1, 0]
    elif label == 'dog':
        return [0, 1]
    return None

# split the training data into train_set & validation_set
def load_training_data(train_dir, img_size, channels, batch_size, limit = 0):
    train_data = []
    count = 0;
#    for filename in os.listdir(train_dir):
    for filename in tqdm(os.listdir(train_dir)):
        count += 1
        if limit > 0 and count > limit:
            break
        if not filename.endswith('.jpg'):
            continue
        jpg = os.path.join(train_dir, filename)
        if channels == 1:
            img = cv2.imread(jpg, cv2.IMREAD_GRAYSCALE)
        elif channels == 3:
            img = cv2.imread(jpg, cv2.IMREAD_COLOR) # BGR
        else:
            print('WTF')
            return None
        
        img = cv2.resize(img, (img_size, img_size))        
        img = np.array(img, dtype = np.float32)
        img = img.reshape(img_size, img_size, channels)
        img = img / 255.0
        
        label = pic_label(filename)
        label = np.array(label, dtype = np.float32)
        
        train_data.append([img, label])

    # shuffle the data and split it into training_set & validation_set
    random.shuffle(train_data)
    validation_size = len(train_data) // 10
    train_set = train_data[validation_size:]
    validation_set = train_data[:validation_size]
    return train_set, validation_set

# return: x_batch, y_batch, next_index
def next_batch(train_set, from_index, batch_size):
    if from_index < 0 or from_index >= len(train_set):
        return None, None, -1
    
    batch_x = []
    batch_y = []
    i = from_index;
    while (i < len(train_set) and i - from_index < batch_size):
        sample = train_set[i]
        batch_x.append(sample[0])
        batch_y.append(sample[1])
        i += 1

    batch_x = np.array(batch_x, dtype = np.float32)
    batch_y = np.array(batch_y, dtype = np.float32)
    return batch_x, batch_y, i



'''
    load train_data and split it into training set with validation set
'''
train_set, validation_set = load_training_data(train_dir, img_size, channels, batch_size)


def showcat(train_set, count = 9):
    batch_x, batch_y, _ = next_batch(train_set, 0, count)
    plt.figure(figsize=(10, 10))
    for i in range(count):
        img3d = batch_x[i, ...]
#        print('img3d.shape: ', img3d.shape)
        plt.subplot(3, 3, i+1)
        if channels == 3:
            plt.imshow(cv2.cvtColor(img3d, cv2.COLOR_BGR2RGB))
        else:
            img3d = img3d.reshape(img3d.shape[0], img3d.shape[1])
            plt.imshow(img3d, cmap=cm.gray)
        plt.show
#        print('label: ', batch[1][i])
showcat(train_set)



# build CNN network
def init_W(shape):
    initial = tf.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(initial)

def init_bias(shape):
    initial = tf.constant(0.1, shape = shape)
    return tf.Variable(initial)

def conv2d(x, W, strides = 1):
    return tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding = 'SAME')

def max_pool(x, ksize = 2, strides = 2):
    return tf.nn.max_pool(x, ksize=[1, ksize, ksize, 1], strides=[1, strides, strides, 1], padding = 'SAME')


def cnn_stack(x, channels, kernels, stack_depth, kernel_size = 3, activate = True, pooling = True):
    for i in range(stack_depth):
        W = init_W([kernel_size, kernel_size, channels, kernels])
        b = init_bias([kernels])
        x = conv2d(x, W)
        x = tf.nn.bias_add(x, b)
        if activate:
            x = tf.nn.relu(x)
        channels = kernels

    if pooling:
        x = max_pool(x)
    return x

'''--------------------------  build the network model ---------------------------
input:
        img_size               image size
        channel                1 or 3
        img_class              # of classes

output: 
        x, y, dropout_keep     placeholders for feeding in
        pred                   the prediction    (? * 2)
        cross_entropy          loss function
'''
def nn_model(img_size, channels, img_class):
    x = tf.placeholder(tf.float32, shape = [None, img_size, img_size, channels])
    y = tf.placeholder(tf.float32, shape = [None, img_class])
    dropout_keep = tf.placeholder(tf.float32)

    h_pool = x
    ch_fm = channels
    kernels = 64
    h_pool = cnn_stack(h_pool, ch_fm, kernels, kernel_size = 3, stack_depth = 3)

    ch_fm = kernels
    kernels *= 2
    h_pool = cnn_stack(h_pool, ch_fm, kernels, kernel_size = 3, stack_depth = 2)

    ch_fm = kernels
    kernels *= 2
    h_pool = cnn_stack(h_pool, ch_fm, kernels, kernel_size = 3, stack_depth = 1)

    ch_fm = kernels
    kernels *= 2
    h_pool = cnn_stack(h_pool, ch_fm, kernels, kernel_size = 2, stack_depth = 1)

    print('after convolution stack, h_pool.shape: ', h_pool.shape)
    featuremap_size = h_pool.shape[1] * h_pool.shape[2] * h_pool.shape[3]
    flatten = tf.reshape(h_pool, [-1, featuremap_size])

    fc_neurons = 1024
    W_fc1 = init_W([int(flatten.shape[1]), fc_neurons])
    b_fc1 = init_bias([fc_neurons])
    fc1 = tf.matmul(flatten, W_fc1) + b_fc1
    h_fc1 = tf.nn.relu(fc1)
    h_fc1_drop = h_fc1 # tf.nn.dropout(h_fc1, dropout_keep)

    W_fc2 = init_W([fc_neurons, fc_neurons])
    b_fc2 = init_bias([fc_neurons])
    fc2 = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    h_fc2 = tf.nn.relu(fc2)
    h_fc2_drop = tf.nn.dropout(h_fc2, dropout_keep)

    W_fc3 = init_W([fc_neurons, img_class])
    b_fc3 = init_bias([img_class])
    output = tf.matmul(h_fc2_drop, W_fc3) + b_fc3

    pred = tf.nn.softmax(output)
    loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=y))
    return x, y, dropout_keep, pred, loss



# prepare session
tf.reset_default_graph()
with tf.Session() as sess:
    x, y, dropout_keep, pred, loss = nn_model(img_size, channels, img_class)

    optimizer = tf.train.AdamOptimizer(lr)
    train_step = optimizer.minimize(loss)

    # compute accuracy
    correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    init = tf.global_variables_initializer()
    sess.run(init)

    # training process
    epoch_count = 2000
    best_validation_loss = 1000000000.0
    current_epoch = 0
    EARLY_STOP_PATIENCE = 20

    for epoch in range(epoch_count):
        print('epoch#: ', epoch)
        
        random.shuffle(train_set)
        batch_x, batch_y, train_index = next_batch(train_set, 0, batch_size)
        
        while train_index > 0:
            # only for comput train_accuracy later
            _batch_x, _batch_y = batch_x, batch_y
            
            train_step.run(feed_dict = {x:batch_x, y:batch_y, dropout_keep:0.5})
            batch_x, batch_y, train_index = next_batch(train_set, train_index, batch_size)

        # retrieve a whole batch again for computing training loss/accuracy
        batch_x, batch_y, train_index = next_batch(train_set, 0, batch_size)
        train_loss = loss.eval(feed_dict = {x:_batch_x, y:_batch_y, dropout_keep:1.0})
        train_loss /= batch_size
        train_accuracy = accuracy.eval(feed_dict = {x:_batch_x, y:_batch_y, dropout_keep:1.0})

        # ------------------- loss & accuracy on validation set -------------------
        vali_batchcount = 0
        validation_loss = 0.0
        validation_accuracy = 0.0
        validation_batch = batch_size // 2
        validation_x, validation_y, validation_index = next_batch(validation_set, 0, validation_batch)
        while validation_index > 0:
            if len(validation_x) != validation_batch:   # just drop the incompleted batch
                break
            _loss = loss.eval(feed_dict = {x:validation_x, y:validation_y, dropout_keep:1.0})
            _accuracy = accuracy.eval(feed_dict = {x:validation_x, y:validation_y, dropout_keep:1.0})
            validation_loss += _loss
            validation_accuracy += _accuracy
            validation_x, validation_y, validation_index = next_batch(validation_set, validation_index, validation_batch)
            vali_batchcount += 1
        validation_loss /= (vali_batchcount * validation_batch)
        validation_accuracy /= vali_batchcount
        # ------------------- loss & accuracy on validation set -------------------
        
        print('training loss: ', train_loss, ',\tvalidation loss: ', validation_loss)
        print('training accuracy: ', train_accuracy, ',\tvalidation accuracy: ', validation_accuracy)

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            current_epoch = epoch
            # save the model
            saver = tf.train.Saver()
            model_saved = saver.save(sess, os.path.join(model_save_dir, 'dogcat'))
            print('model saved in :{0}'.format(model_saved))
        elif (epoch - current_epoch) >= EARLY_STOP_PATIENCE:
            print('early stopping')
            break

print('OMG! epochs finished.')
