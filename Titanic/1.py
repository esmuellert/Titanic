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
validation_data0 = tmp[601:,2:]

train_lable = tmp[1:601,1]
validation_lable0 = tmp[601:,1]
train_lable = train_lable.reshape(600,1)
validation_lable0 = validation_lable0.reshape(291,1)

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

#regularzation
lamda = 0.06
regularzation = (tf.reduce_sum(tf.square(theta_1))+tf.reduce_sum(tf.square(theta_2)))*lamda/600

# the error between prediction and real data
J_theta = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = ys,logits = z_2)) + regularzation

a = 0.8 # learning rate
train_step = tf.train.GradientDescentOptimizer(a).minimize(J_theta)

#prediction
validation_data = tf.constant(validation_data0,tf.float32)
validation_lable = tf.constant(validation_lable0,tf.float32)

a_1 = tf.sigmoid(tf.matmul(validation_data,theta_1))
a_0 = tf.ones([291,1])
a_20 = tf.concat([a_0,a_1],1)
a_2 = tf.sigmoid(tf.matmul(a_20,theta_2))
prediction = tf.round(a_2)

validation = tf.subtract(prediction,validation_lable)
validation = tf.abs(validation)
accuracy =tf.reduce_sum(validation) 
accuracy = (291-accuracy)/291

#initialize
init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

for i in range(200000):
    #training
    sess.run(train_step,feed_dict={xs:train_data,ys:train_lable})
    
    if i%1000 == 0:
        loss = sess.run(J_theta,feed_dict={xs:train_data,ys:train_lable})
        print("J_theta:",loss)
        acc = sess.run(accuracy)
        print(acc)
        






