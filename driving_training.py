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

# consts
IMU_DATA_DIR = './data_dir'
PARAMFILE = 'params.pkl'

# Hyperparamters
n_input = 12
n_hidden = 10
n_output = 2
lr = 0.1
epochs = 1000
display_step = 10

# Dataset in the next format:
#   input = prev_x, prev_y, curr_x, curr_y, next_x, next_y, velocity 
#   targets (tags) = acceleration, steering
inputs = []
targets = []

for filename in os.listdir(IMU_DATA_DIR):

    #FIXME: learn from all files and not only streight driving
    if filename != 'collect_straight.csv':
        continue

    with open("{}/{}".format(IMU_DATA_DIR, filename), "r") as f:
    
        # remove the title line
        f.readline()
    
        for line in f.readlines():
    
            # get input (x_prev, y_prev, x_curr, y_curr, x_next, y_next)
            input_line = line.split(',')[0:n_input]
            input_line = [float(x) for x in input_line]
    
            # get targe (x_prev, y_prev, x_curr, y_curr, x_next, y_next)
            target_line = line.split(',')[n_input:-1]
            target_line = [float(x) for x in target_line]
    
            inputs.append(input_line)
            targets.append(target_line)

inputs = np.array(inputs)
targets = np.array(targets)

print("\ninputs.len = {}".format(len(inputs)))
print("targets.len = {}\n".format(len(targets)))


# Placeholders
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

# Weights
W1 = tf.Variable(tf.random_uniform([n_input, n_hidden], -1.0, 1.0))
W2 = tf.Variable(tf.random_uniform([n_hidden, n_output], -1.0, 1.0))

# Bias
b1 = tf.Variable(tf.zeros([n_hidden]))
b2 = tf.Variable(tf.zeros([n_output]))

#L2 = tf.nn.relu(tf.matmul(X, W1) + b1)
#L3 = tf.nn.relu(tf.matmul(L2, W2) + b2)
L2 = tf.sigmoid(tf.matmul(X, W1) + b1)
L3 = tf.sigmoid(tf.matmul(L2, W2) + b2)
#L3 = tf.nn.softmax(tf.matmul(L2, W2) + b2)
cost = tf.losses.mean_squared_error(Y, L3)
#cost = tf.reduce_mean(-Y*tf.log(L3) - (1-Y) * tf.log(1-L3)) # cross Entrophy ?!
optimizer = tf.train.GradientDescentOptimizer(lr).minimize(cost)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for step in range(epochs):
        _, c = sess.run([optimizer, cost], feed_dict = {X: inputs, Y: targets})

        if step % display_step == 0:
            print("Cost: ", c)

    answer = tf.equal(tf.floor(L3 + 0.1), Y)
    accuracy = tf.reduce_mean(tf.cast(answer, "float"))

    print(sess.run([L3], feed_dict = {X: inputs, Y: targets}))
    print("Accuracy: ", accuracy.eval({X: inputs, Y: targets}))

    # exporting network weights to a .pkl file to be loaded in testing
    params = [sess.run(param) for param in tf.trainable_variables()]
    pickle.dump(params, open(PARAMFILE, 'wb'))


