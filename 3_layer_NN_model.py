import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import math
import timeit
import cv2
# read training data from CSV file 
num_training = 10000
num_val = 1000
#num_test = 28000
def get_FACE_data():
    data=np.load('sorted_train_data.npy')
    X_train = data[1000:30000,:268]
    y_train = np.argmax(data[1000:30000,268:],axis=1)
    X_val = data[:1000,:268]
    y_val= np.argmax(data[:1000,268:],axis=1)
    return X_train,y_train,X_val,y_val

X_train,y_train,X_val,y_val = get_FACE_data()

#Reshape the data from 784 to 28x28 since it represents the image width and height
y_train = np.reshape(y_train,-1)
y_val = np.reshape(y_val,-1)
print('Train data shape: ', X_train.shape)
print('Train labels shape: ', y_train.shape)
print('Validation data shape: ', X_val.shape)
print('Validation labels shape: ', y_val.shape)
#cv2.imshow('image',X_train[0,:,:,:])

def run_model(session, predict, loss_val, Xd, yd,
              epochs=1, batch_size=64, print_every=100,
              training=None, plot_losses=False):
    # have tensorflow compute accuracy
    correct_prediction = tf.equal(tf.argmax(predict,1), y)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    # shuffle indicies
    train_indicies = np.arange(Xd.shape[0])

    training_now = training is not None
    
    # setting up variables we want to compute (and optimizing)
    # if we have a training function, add that to things we compute
    variables = [mean_loss,correct_prediction,accuracy]
    if training_now:
        variables[-1] = training
    
    # counter 
    iter_cnt = 0
    for e in np.arange(epochs):
        # keep track of losses and accuracy
        correct = 0
        losses = []
        # make sure we iterate over the dataset once
        for i in np.arange(int(math.ceil(Xd.shape[0]/batch_size))):
            # generate indicies for the batch
            start_idx = (i*batch_size)%Xd.shape[0]
            idx = train_indicies[start_idx:start_idx+batch_size]
            
            # create a feed dictionary for this batch
            feed_dict = {X: Xd[idx,:],
                         y: yd[idx],
                         is_training: training_now }
            # get batch size
            actual_batch_size = yd[idx].shape[0]
            
            # have tensorflow compute loss and correct predictions
            # and (if given) perform a training step
            loss, corr, _ = session.run(variables,feed_dict=feed_dict)
            
            # aggregate performance stats
            losses.append(loss*actual_batch_size)
            correct += np.sum(corr)
            
            # print every now and then
            if training_now and (iter_cnt % print_every) == 0:
                print("Iteration {0}: with minibatch training loss = {1:.3g} and accuracy of {2:.2g}"\
                      .format(iter_cnt,loss,np.sum(corr)/actual_batch_size))
            iter_cnt += 1
        total_correct = correct/Xd.shape[0]
        total_loss = np.sum(losses)/Xd.shape[0]
        print("Epoch {2}, Overall loss = {0:.3g} and accuracy of {1:.3g}"\
              .format(total_loss,total_correct,e+1))
        if plot_losses:
            plt.plot(losses)
            plt.grid(True)
            plt.title('Epoch {} Loss'.format(e+1))
            plt.xlabel('minibatch number')
            plt.ylabel('minibatch loss')
            plt.show()
    return total_loss,total_correct
# clear old variables
tf.reset_default_graph()

# define our input (e.g. the data that changes every batch)
# The first dim is None, and gets sets automatically based on batch size fed in
X = tf.placeholder(tf.float32, [None,268])
y = tf.placeholder(tf.int64, [None])
is_training = tf.placeholder(tf.bool)

#Define Model
def Emotion_model(X,y,is_training):
    W1=tf.get_variable(shape=[268,1000],name = 'W1')
    b1=tf.get_variable(shape=[1000],name = 'b1')
    W2=tf.get_variable(shape=[1000,3000],name = 'W2')
    b2=tf.get_variable(shape=[3000],name = 'b2')
    W3=tf.get_variable(shape=[3000,7],name = 'W3')
    b3=tf.get_variable(shape=[7],name = 'b3')
    hfc1=tf.nn.relu(tf.matmul(X,W1)+b1)
    hfc2=tf.nn.relu(tf.matmul(hfc1,W2)+b2)
    y_out=tf.matmul(hfc2,W3)+b3
    variable_list = [W1,b1,W2,b2,W3,b3]
    return(y_out,variable_list)

y_out,variable_list = Emotion_model(X,y,is_training)

#mean_loss: a TensorFlow variable (scalar) with numerical loss
#optimizer: a TensorFlow optimizer
mean_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf.one_hot(y,7),logits=y_out))
optimizer = tf.train.AdamOptimizer(5e-4)

#batch normalization in tensorflow requires this extra dependency
extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(extra_update_ops):
    opt_op = optimizer.minimize(mean_loss)
# Create an ExponentialMovingAverage object
ema = tf.train.ExponentialMovingAverage(decay=0.9999)

# Create the shadow variables, and add ops to maintain moving averages
# of var0 and var1.
maintain_averages_op = ema.apply(variable_list)

# Create an op that will update the moving averages after each training
# step.  This is what we will use in place of the usual training op.
with tf.control_dependencies([opt_op]):
    train_step = tf.group(maintain_averages_op)    


#Prediction
prediction = tf.argmax(y_out,1)   
#Lets strat a session
sess = tf.Session()
sess.run(tf.global_variables_initializer())
model_saver = tf.train.Saver()  
print('Training')
run_model(sess,y_out,mean_loss,X_train,y_train,300,100,100,train_step,False)
print('Validation')
run_model(sess,y_out,mean_loss,X_val,y_val,1,64)
model_saver.save(sess,"savemodels/Emotion_model2")
sess.close()