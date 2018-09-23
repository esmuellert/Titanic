import numpy as np
import tensorflow as tf


# get data from train_1.csv
tmp = np.genfromtxt("train_2.csv", dtype=float, delimiter=",")

# set training_data quantity and validation data quantity
train_quantity = 750
validation_quantity = 891-train_quantity

# normalization
tem = tmp[1:,3:]
ave = np.mean(tem,axis = 0)*np.ones((tem.shape[0],tem.shape[1]))
ptp = np.ptp(tem,axis = 0)*np.ones((tem.shape[0],tem.shape[1]))
tem = np.subtract(tem,ave)
tem = np.true_divide(tem,ptp)
tmp[1:,3:] = tem

#apart train,validation
train_data = tmp[1:train_quantity+1,3:]
validation_data0 = tmp[train_quantity+1:,3:]

train_lable = tmp[1:train_quantity+1,2]
validation_lable0 = tmp[train_quantity+1:,2]

train_lable = train_lable.reshape(train_quantity,1)
validation_lable0 = validation_lable0.reshape(validation_quantity,1)

# random the rank of train data and validation data
# np.random.shuffle(train_data)
# np.random.shuffle(validation_data0)

# define layer function
def add_layer(input,in_size,out_size):
    theta = tf.Variable(tf.random_normal([in_size,out_size]))
    if out_size==1:
        z = tf.matmul(input,theta)
        a = tf.sigmoid(z)
    else:
        z = tf.matmul(input,theta)
        a = tf.nn.relu(z)
        a_0 = tf.ones([train_quantity,1])
        a = tf.concat([a_0,a],1)
    return theta,a,z


# define placeholder for inputs to network
xs = tf.placeholder(tf.float32,[None,tem.shape[1]])
ys = tf.placeholder(tf.float32,[None,1])

# add hidden layer
theta_1,l1,z_1 = add_layer(xs,7,13)
# add output layer
theta_2,output,z_2 = add_layer(l1,14,1)

#regularzation
lamda = 0.09
regularzation = (tf.reduce_sum(tf.square(theta_1))+tf.reduce_sum(tf.square(theta_2)))*lamda/train_quantity

# the error between prediction and real data
J_theta = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = ys,logits = z_2)) + regularzation

a =  0.5
# learning rate
train_step = tf.train.GradientDescentOptimizer(a).minimize(J_theta)

#prediction
validation_data = tf.constant(validation_data0,tf.float32)
validation_lable = tf.constant(validation_lable0,tf.float32)

a_1 = tf.nn.relu(tf.matmul(validation_data,theta_1))
a_0 = tf.ones([validation_quantity,1])
a_20 = tf.concat([a_0,a_1],1)
a_2 = tf.sigmoid(tf.matmul(a_20,theta_2))
prediction = tf.round(a_2)

validation = tf.subtract(prediction,validation_lable)
validation = tf.abs(validation)
accuracy =tf.reduce_sum(validation) 
accuracy = (validation_quantity-accuracy)/validation_quantity

#initialize
init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

for i in range(10000):
    #training
    sess.run(train_step,feed_dict={xs:train_data,ys:train_lable})
    
    if i%100 == 0:
        loss = sess.run(J_theta,feed_dict={xs:train_data,ys:train_lable})
        print(i)
        print("J_theta:",loss)
        acc = sess.run(accuracy)
        print(acc)
        print("\n")








# import test sets
test = np.genfromtxt("test_2.csv",dtype = float,delimiter=",")

#reshape test sets
tem1 = test[1:,2:]
ave1 = np.mean(tem1,axis = 0)*np.ones((418,7))
ptp1 = np.ptp(tem1,axis = 0)*np.ones((418,7))
tem1 = np.subtract(tem1,ave1)
tem1 = np.true_divide(tem1,ptp1)
test[1:,2:] = tem1
test = test[1:,2:]

# define tensorflow constant test data
test_data = tf.constant(test,tf.float32)

# predict test data
t_1 = tf.nn.relu(tf.matmul(test_data,theta_1))
t_0 = tf.ones([418,1])
t_20 = tf.concat([t_0,t_1],1)
t_2 = tf.sigmoid(tf.matmul(t_20,theta_2))
prediction_1 = tf.round(t_2)
prediction_1 = tf.to_int32(prediction_1)

# return result
result = sess.run(prediction_1)
No = np.arange(892,1310)
No = No.reshape(418,1)
title = np.array(["PassengerId","Survived"])
title = title.reshape(1,2)
result = np.append(No,result,axis=1)
result = np.append(title,result,axis=0)
np.savetxt("result.csv", result, delimiter=",",fmt = "%s")
# import test sets
test = np.genfromtxt("test_2.csv",dtype = float,delimiter=",")

#reshape test sets
tem1 = test[1:,2:]
ave1 = np.mean(tem1,axis = 0)*np.ones((418,7))
ptp1 = np.ptp(tem1,axis = 0)*np.ones((418,7))
tem1 = np.subtract(tem1,ave1)
tem1 = np.true_divide(tem1,ptp1)
test[1:,2:] = tem1
test = test[1:,2:]

# define tensorflow constant test data
test_data = tf.constant(test,tf.float32)

# predict test data
t_1 = tf.nn.relu(tf.matmul(test_data,theta_1))
t_0 = tf.ones([418,1])
t_20 = tf.concat([t_0,t_1],1)
t_2 = tf.sigmoid(tf.matmul(t_20,theta_2))
prediction_1 = tf.round(t_2)
prediction_1 = tf.to_int32(prediction_1)

# return result
result = sess.run(prediction_1)
No = np.arange(892,1310)
No = No.reshape(418,1)
title = np.array(["PassengerId","Survived"])
title = title.reshape(1,2)
result = np.append(No,result,axis=1)
result = np.append(title,result,axis=0)
np.savetxt("result.csv", result, delimiter=",",fmt = "%s")
