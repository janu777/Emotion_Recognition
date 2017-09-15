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


def get_FACE_data():
    #read the data
    train_data = np.load('data.npy')
    train_labels = np.load('labels.npy')
    # Split 1000 examples for validation
    X_train = train_data[10:11010,:]
    y_train = train_labels[10:11010,:]
    X_val = train_data[:10,:]
    y_val= train_labels[:10,:]
    return X_train,y_train,X_val,y_val

X_train,y_train,X_val,y_val = get_FACE_data()

#Reshape the data from 784 to 28x28 since it represents the image width and height
X_train = np.reshape(X_train,(-1,48,48,1))
X_val = np.reshape(X_val,(-1,48,48,1))
y_train = np.reshape(y_train,(-1, 7))
y_val = np.reshape(y_val,(-1,7))
print('Train data shape: ', X_train.shape)
print('Train labels shape: ', y_train.shape)
print('Validation data shape: ', X_val.shape)
print('Validation labels shape: ', y_val.shape)
#cv2.imshow('image',X_train[0,:,:,:])

def run_model(session, predict, loss_val, Xd, yd,
              epochs=1, batch_size=64, print_every=100,
              training=None, plot_losses=False):
    # have tensorflow compute accuracy
    correct_prediction = tf.equal(tf.argmax(predict,1), tf.argmax(y,1))
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
X = tf.placeholder(tf.float32, [None, 48, 48, 1 ])
y = tf.placeholder(tf.int64, [None,7])
is_training = tf.placeholder(tf.bool)


#########################################################################################################################
#Define Model
def Emotion_model(X,y,is_training):
    Wconv1=tf.get_variable(shape=[5,5,1,64],name = 'Wconv1')
    bconv1=tf.get_variable(shape=[64],name = 'bconv1')
    Wconv2=tf.get_variable(shape=[5,5,64,64],name = 'Wconv2')
    bconv2=tf.get_variable(shape=[64],name = 'bconv2')
    Wconv3=tf.get_variable(shape=[5,5,64,128],name = 'Wconv3')
    bconv3=tf.get_variable(shape=[128],name = 'bconv3')
    W1=tf.get_variable(shape=[12*12*128,3072],name = 'W1')
    b1=tf.get_variable(shape=[1024],name = 'b1')
    W2=tf.get_variable(shape=[1024,7],name = 'W2')
    b2=tf.get_variable(shape=[7],name = 'b2')
    hconv1=tf.nn.relu(tf.nn.conv2d(X,Wconv1,strides=[1,1,1,1],padding='SAME')+bconv1)
    hbn1=tf.contrib.layers.batch_norm(hconv1,center=True, scale=True,is_training=is_training,scope='hbn1')
    hpool1=tf.nn.max_pool(hbn1,ksize=[1,1,1,1],strides=[1,2,2,1],padding='SAME')
    hconv2=tf.nn.relu(tf.nn.conv2d(hpool1,Wconv2,strides=[1,1,1,1],padding='SAME')+bconv2)
    hpool2=tf.nn.max_pool(hconv2,ksize=[1,1,1,1],strides=[1,2,2,1],padding='SAME')
    hconv3_flat=tf.reshape(hconv3,(-1,12*12*128))
    hfc1=tf.nn.relu(tf.matmul(hconv3_flat,W1)+b1)
    y_out=tf.matmul(hfc1,W2)+b2
    variable_list = [Wconv1,bconv1,Wconv2,bconv2,W1,b1,W2,b2]
    return(y_out,variable_list)

y_out,variable_list = Emotion_model(X,y,is_training)

###########################################################################################################################
#mean_loss: a TensorFlow variable (scalar) with numerical loss
#optimizer: a TensorFlow optimizer
mean_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=y_out))
optimizer = tf.train.AdamOptimizer(1e-3)

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
# step.
with tf.control_dependencies([opt_op]):
    train_step = tf.group(maintain_averages_op)    
#prediction
prediction = tf.argmax(y_out,1)   

####################################################################################################################################################
#Start a session
sess = tf.Session()
sess.run(tf.global_variables_initializer())
model_saver = tf.train.Saver()  
print('Training')
run_model(sess,y_out,mean_loss,X_train,y_train,50,64,100,train_step,False)
print('Validation')
run_model(sess,y_out,mean_loss,X_val,y_val,1,64)
model_saver.save(sess,"/home/linux/Documents/DLExercises/Emotion_Recognition/saved_models/Emotion_model1")
sess.close()
         