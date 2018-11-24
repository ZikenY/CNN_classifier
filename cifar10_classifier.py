import pickle
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

cifar10_dir = './cifar10'
model_save_dir = './model'
img_w = 32
img_h = 32
channels = 3    # 1 or 3
img_class = 10  # convert to one-hot
batch_size = 100
lr = 1e-4

def load_CIFAR_batch(filename):
    with open(filename, 'rb') as f:
        datadict = pickle.load(f, encoding='latin1')
        X = datadict['data']
        Y = datadict['labels']
        X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
        Y = np.array(Y)
        return X, Y


def load_CIFAR10(data_dir):
    xs = []
    ys = []
    for b in range(1, 2):
        f = os.path.join(data_dir, 'data_batch_%d' % (b,))
        X, Y = load_CIFAR_batch(f)
        xs.append(X)
        ys.append(Y)
    Xtr = np.concatenate(xs)
    Ytr = np.concatenate(ys)
    del X, Y
    Xte, Yte = load_CIFAR_batch(os.path.join(data_dir, 'test_batch'))
    return Xtr, Ytr, Xte, Yte

def one_hot_me(label, classes = 10):
    if label < 0 or label >= classes:
        print('WTF one_hot_me: ', label, ', classes = ', classes)
        return None
    one_hot = [0] * 10
    one_hot[label] = 1
    return one_hot

def one_hot_labels(labels, classes = 10):
    one_hots = list()
    for label in labels:
        oh = one_hot_me(label, classes)
        one_hots.append(oh)
    return np.array(one_hots)

def pack_xy(x, y):
    if len(x) != len(y):
        print('WTF pack_data_label: ', x, ', ', y);
        return None
    dataset = list()
    for i in range(0, len(x)):
        dataset.append([x[i], y[i]])
    return np.array(dataset)

# split the training data into train_set & validation_set
def load_training_data(train_dir, img_w, img_h, channels, batch_size, limit = 0):
    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)
    y_train = one_hot_labels(y_train)
    y_test = one_hot_labels(y_test)

    train_data = pack_xy(X_train, y_train)
    test_data = pack_xy(X_test, y_test)

    print('Training data shape: ', X_train.shape)
    print('Training labels shape: ', y_train.shape)
    print('Test data shape: ', X_test.shape)
    print('Test labels shape: ', y_test.shape)
    print('train_data shape: ', train_data.shape)
    print('test_data shape: ', test_data.shape)

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
train_set, validation_set = load_training_data(cifar10_dir, img_w, img_h, channels, batch_size)


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

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides = [1, 1, 1, 1], padding = 'VALID')

def max_pool(x):
    return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')


'''--------------------------  build the network model ---------------------------
input:
        img_h, img_w           image size
        channel                1 or 3
        img_class              # of classes
        conv_depth             # of [convolution, max_pooling, relu] layer_pack

output: 
        x, y, dropout_keep     placeholders for feeding in
        pred                   the prediction    (? * 2)
        cross_entropy          loss function
'''
def nn_model(img_h, img_w, channels, img_class, conv_depth):
    x = tf.placeholder(tf.float32, shape = [None, img_h, img_w, channels])
    y = tf.placeholder(tf.float32, shape = [None, img_class])
    dropout_keep = tf.placeholder(tf.float32)

    # ---------------------  CNN stack  -----------------------
    last_channels = channels;
    kernels = 32
    h_pool = x
    for i in range(conv_depth):
        W = init_W([3, 3, last_channels, kernels])
        b = init_bias([kernels])
        conv = conv2d(h_pool, W) + b
        h_pool = max_pool(conv)
        last_channels = kernels
        kernels *= 2
        
    W2 = init_W([2, 2, last_channels, kernels])
    b2 = init_bias([kernels])
    conv2 = conv2d(h_pool, W2) + b2
    h2 = max_pool(conv2)
        
#    featuremap_size = h_pool.shape[1] * h_pool.shape[2] * h_pool.shape[3]
#    flatten = tf.reshape(h_pool, [-1, featuremap_size])
    featuremap_size = h2.shape[1] * h2.shape[2] * h2.shape[3]
    flatten = tf.reshape(h2, [-1, featuremap_size])
    # ---------------------  CNN stack  -----------------------

    fc_neurons = 128
    W_fc1 = init_W([int(flatten.shape[1]), fc_neurons])
    b_fc1 = init_bias([fc_neurons])
    fc1 = tf.matmul(flatten, W_fc1) + b_fc1
    h_fc1 = tf.nn.relu(fc1)
    h_fc1_drop = h_fc1 # tf.nn.dropout(h_fc1, dropout_keep)
    # h_fc1_drop = tf.nn.dropout(h_fc1, dropout_keep)

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
sess = tf.InteractiveSession()
x, y, dropout_keep, pred, loss = nn_model(img_h, img_w, channels, img_class, 1)

optimizer = tf.train.AdamOptimizer(lr)
train_step = optimizer.minimize(loss)

# compute accuracy
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()
sess.run(init)

# training process
epoch_count = 200
best_validation_loss = 1000000000.0
best_validation_accuracy = 0.0
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

    train_loss = loss.eval(feed_dict = {x:_batch_x, y:_batch_y, dropout_keep:1.0})
    train_accuracy = accuracy.eval(feed_dict = {x:_batch_x, y:_batch_y, dropout_keep:1.0})

    vali_count = 0
    validation_loss = 0.0
    validation_accuracy = 0.0
    validation_x, validation_y, validation_index = next_batch(validation_set, 0, batch_size)
    while validation_index > 0:
        _loss = loss.eval(feed_dict = {x:validation_x, y:validation_y, dropout_keep:1.0})
        _accuracy = accuracy.eval(feed_dict = {x:validation_x, y:validation_y, dropout_keep:1.0})
        validation_loss += _loss * (len(validation_x) / batch_size)
        validation_accuracy += _accuracy * (len(validation_x) / batch_size)
        validation_x, validation_y, validation_index = next_batch(validation_set, validation_index, batch_size)
        vali_count += 1
    validation_loss /= vali_count
    validation_accuracy /= vali_count
    
    print('training loss: ', train_loss, ', \tvalidation loss: ', validation_loss)
    print('training accuracy: ', train_accuracy, '\tvalidation accuracy: ', validation_accuracy)
    
    if validation_loss < best_validation_loss \
      or abs(best_validation_loss - validation_loss) < 0.1 \
      and validation_accuracy > best_validation_accuracy:
        best_validation_loss = validation_loss
        best_validation_accuracy = validation_accuracy
        current_epoch = epoch
        # save the model
        saver = tf.train.Saver()
        model_saved = saver.save(sess, os.path.join(model_save_dir, 'cifar10'))
        print('model saved in :{0}'.format(model_saved))
    elif (epoch - current_epoch) >= EARLY_STOP_PATIENCE:
        print('early stopping')
        break
            
sess.close()
print('OMG! epochs finished.')

