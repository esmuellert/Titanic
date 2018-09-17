import numpy as np
import tensorflow as tf
#import matplotlib.pyplot as plt

# get data from train_1.csv
tmp = np.genfromtxt("train_1.csv", dtype=float, delimiter=",")

tem = tmp[1:,2:]
ave = np.mean(tem,axis = 0)*np.ones((891,6))
ptp = np.ptp(tem,axis = 0)*np.ones((891,6))
tem = np.subtract(tem,ave)
tem = np.true_divide(tem,ptp)
tmp[1:,2:] = tem

train_data = tmp[1:601,2:]
validation_data = tmp[601:,2:]
train_lable = tmp[1:,1]
train_lable = train_lable.reshape(891,1)

def add_layer(input,in_size,out_size):
    theta = tf.Variable(tf.random_normal([in_size,out_size]))
    if out_size==1:
        z = tf.matmul(input,theta)
        a = tf.sigmoid(z)
    else:
        z = tf.matmul(input,theta)
        a = tf.sigmoid(z)
        a_0 = tf.ones([600,1])
        a = tf.concat([a_0,a],1)
    return theta,a,z


# define placeholder for inputs to network
xs = tf.placeholder(tf.float32,[None,6])
ys = tf.placeholder(tf.float32,[None,1])

# add hidden layer
theta_1,l1,z_1 = add_layer(xs,6,12)
# add output layer
theta_2,output,z_2 = add_layer(l1,13,1)

# the error between prediction and real data
J_theta = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = ys,logits = z_2))

a = 0.1 # learning rate
train_step = tf.train.GradientDescentOptimizer(a).minimize(J_theta)

#initialize
init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

for i in range(100000):
    #training
    sess.run(train_step,feed_dict={xs:train_data[0:600],ys:train_lable[0:600]})
    
    if i%1000 == 0:
        loss = sess.run(J_theta,feed_dict={xs:train_data[0:600],ys:train_lable[0:600]})
        print("J_theta:",loss)
        theta1 = sess.run(theta_1)
        theta2 = sess.run(theta_2)
        #print(theta1,theta2,train_data[601:890])


