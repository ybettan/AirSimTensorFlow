# FIXME: update
'''
This script is teaching the netwrork how to drive.

The input consist of (x_prev, y_prev, x_curr, y_curr, x_next, y_next, v) and 
the output consist of (a, steering).

    v - velocity
    a - acceleration

Training proccess consist of teaching the network to drive from (x_curr, y_curr) 
to (x_next, y_next) when arriving from (x_prev, y_prev) and with velocity v.

The testing stage is to give a slope to the network and to expect it to drive
correctly after learnig how to drive between any 3 points with a given velocity.
'''

import tensorflow as tf
import numpy as np
import pickle
import os
import random
from matplotlib import pyplot as plt
import itertools #FIXME: remove


def read_data(dirname, n_input, n_output):
    '''
    dirname: the directory name pointing to the data files
    (input, targets): normelized np-arrays with the data
    '''
    lines = []
    
    for filename in os.listdir(dirname):
    
        #FIXME: learn from all files and not only streight driving
        #if filename != 'collect_straight_no_throttle_no_steer.csv':
        if filename != 'collect_straight_low_throttle_low_steer.csv':
            continue
    
        with open("{}/{}".format(dirname, filename), "r") as f:
        
            # remove the title line
            f.readline()
        
            for line in f.readlines():

                # read all the data
                line = line.split(',')[:n_input + n_output]
                line = [float(x) for x in line]
                lines.append(line)

    # shuffle the data
    random.shuffle(lines)

    # split it (data, tags)
    inputs = np.array([line[0:n_input] for line in lines])
    targets = np.array([line[n_input:] for line in lines])

    # normelize only the features (the inputs)
    lines = tf.keras.utils.normalize(inputs)

    return (inputs, targets)

def plot_regression(inputs, targets, n_samples):

    # plot steering
    plt.plot([x[0] for x in targets[:n_samples]], color = 'red', label = 'Real')
    plt.plot([x[0] for x in prediction[0][:n_samples]], color = 'blue', label = 'Predicted')
    plt.title('Steering - arg#1')
    plt.legend()
    plt.show()
    
    # plot throttle
    plt.plot([x[0] for x in targets[:n_samples]], color = 'red', label = 'Real')
    plt.plot([x[1] for x in prediction[0][:n_samples]], color = 'blue', label = 'Predicted')
    plt.title('Throttle - arg#2')
    plt.legend()
    plt.show()



# consts
IMU_DATA_DIR = './data_dir'
PARAMFILE = 'params.pkl'

# Hyperparamters
N_INPUT = 14
N_HIDDEN = 10
N_OUTPUT = 2
N_SAMPLES = 1000
LR = 0.01 
EPOCHS = 1000
DISPLAY_STEP = 10

##FIXME: remove OR
#lst_in = [list(i) for i in itertools.product([0, 1], repeat=N_INPUT)]
#lst_out = []
#for l in lst_in:
#    if not any(l):
#        lst_out.append([0, 0])
#    else:
#        lst_out.append([1, 1])
#inputs = np.array(lst_in)
#targets = np.array(lst_out)

##FIXME: remove +
#mat_in = []
#mat_out = []
#for _ in range(10):
#    lst_in = []
#    for _ in range(14):
#        lst_in.append(random.random())
#    lst_out = [sum(lst_in), sum(lst_in)]
#    mat_in.append(lst_in)
#    mat_out.append(lst_out)
#inputs = np.array(mat_in)
#targets = np.array(mat_out)


(inputs, targets) = read_data(IMU_DATA_DIR, N_INPUT, N_OUTPUT)

print("\ninputs.len = {}".format(len(inputs)))
print("targets.len = {}\n".format(len(targets)))


# Placeholders
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

# Weights
W1 = tf.Variable(tf.random_uniform([N_INPUT, N_HIDDEN], -1.0, 1.0))
W2 = tf.Variable(tf.random_uniform([N_HIDDEN, N_OUTPUT], -1.0, 1.0))

# Bias
b1 = tf.Variable(tf.zeros([N_HIDDEN]))
b2 = tf.Variable(tf.zeros([N_OUTPUT]))

L2 = tf.nn.relu(tf.matmul(X, W1) + b1)
L3 = tf.nn.relu(tf.matmul(L2, W2) + b2)
#L2 = tf.sigmoid(tf.matmul(X, W1) + b1)
#L3 = tf.sigmoid(tf.matmul(L2, W2) + b2)
#L3 = tf.nn.softmax(tf.matmul(L2, W2) + b2)
cost = tf.losses.mean_squared_error(Y, L3)
#cost = tf.losses.absolute_difference(Y, L3)
#cost = tf.reduce_mean(-Y*tf.log(L3) - (1-Y) * tf.log(1-L3)) # cross Entrophy ?!
optimizer = tf.train.GradientDescentOptimizer(LR).minimize(cost)
#optimizer = tf.train.AdamOptimizer(LR).minimize(cost)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    #FIXME: try to shuffle the data first
    for step in range(EPOCHS):
        _, c = sess.run([optimizer, cost], feed_dict = {X: inputs[:N_SAMPLES], Y: targets[:N_SAMPLES]})

        if step % DISPLAY_STEP == 0:
            print("Cost: ", c)

    #answer = tf.equal(tf.floor(L6 + 0.1), Y)
    answer = tf.equal(tf.floor(L3 + 0.1), Y)
    accuracy = tf.reduce_mean(tf.cast(answer, "float"))

    #print(sess.run([L3], feed_dict = {X: inputs, Y: targets}))
    prediction = sess.run([L3], feed_dict = {X: inputs[:N_SAMPLES], Y: targets[:N_SAMPLES]})
    print(prediction)
    print("Accuracy: ", accuracy.eval({X: inputs[:N_SAMPLES], Y: targets[:N_SAMPLES]}))

    # exporting network weights to a .pkl file to be loaded in testing
    params = [sess.run(param) for param in tf.trainable_variables()]
    pickle.dump(params, open(PARAMFILE, 'wb'))


# plot the regression
plot_regression(inputs, targets, N_SAMPLES)


