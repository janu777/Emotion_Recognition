import sys
import cv2
import tensorflow as tf
import numpy as np
sess = tf.Session()
saver = tf.train.import_meta_graph('saved_models/Emotion_model2.meta')
saver.restore(sess, 'saved_models/Emotion_model2')
graph = tf.get_default_graph()
K = tf.placeholder(tf.float32, [None, 28, 28, 1 ])
L = tf.placeholder(tf.int64, [None])
is_training = tf.placeholder(tf.bool)
def Emotion_model(K,L,is_training):
		Wconv1=sess.run(graph.get_tensor_by_name('Wconv1:0'))
		bconv1=sess.run(graph.get_tensor_by_name('bconv1:0'))
		Wconv2=sess.run(graph.get_tensor_by_name('Wconv2:0'))
		bconv2=sess.run(graph.get_tensor_by_name('bconv2:0'))
		W1=sess.run(graph.get_tensor_by_name('W1:0'))
		b1=sess.run(graph.get_tensor_by_name('b1:0'))
		W2=sess.run(graph.get_tensor_by_name('W2:0'))
		b2=sess.run(graph.get_tensor_by_name('b2:0'))
		hconv1=tf.nn.relu(tf.nn.conv2d(K,Wconv1,strides=[1,1,1,1],padding='SAME')+bconv1)
		hpool1=tf.nn.max_pool(hconv1,ksize=[1,1,1,1],strides=[1,2,2,1],padding='SAME')
		hconv2=tf.nn.relu(tf.nn.conv2d(hpool1,Wconv2,strides=[1,1,1,1],padding='SAME')+bconv2)
		hpool2=tf.nn.max_pool(hconv2,ksize=[1,1,1,1],strides=[1,2,2,1],padding='SAME')
		hpool2_flat=tf.reshape(hpool2,(-1,7*7*128))
		hfc1=tf.nn.relu(tf.matmul(hpool2_flat,W1)+b1)
		y_out=tf.matmul(hfc1,W2)+b2
		return(y_out)
y_out = Emotion_model(K,L,is_training)    
prediction = tf.argmax(y_out,1)		
EMOTIONS = ['ANGRY', 'DISGUSTED', 'FEARFUL', 'HAPPY', 'SAD', 'SURPRISED', 'NEUTRAL']
print('Hello There! \nType 1 to specify video path \nType 2 to use Webcam\n')
choice = input()
if choice==1:
	print('Type in your path name:')
	cap = cv2.VideoCapture(raw_input())
else:	
	cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('HAAR/haarcascade_frontalface_default.xml')
i = 0   
while(True):
	ret, frame = cap.read()
	resized_face = np.zeros((28,28),dtype=np.float32)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray,scaleFactor = 1.3,minNeighbors = 5)			
	for (x,y,w,h) in faces:
		cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
		roi_gray = gray[y:y+h, x:x+w]
		resized_face = cv2.resize(roi_gray, (28, 28), interpolation = cv2.INTER_CUBIC) / 255.
		yshape = np.zeros((1,),dtype=np.int64)
		#print(resized_face.dtype)
		resized_face = resized_face.astype(np.float32)
		resized_face = np.reshape(resized_face,(-1,28,28,1))
		predicted_label = sess.run(prediction,feed_dict={K: resized_face,L: yshape,is_training: False})
		#predicted_label = a.testin(resized_face,yshape,False)
		cv2.putText(frame, EMOTIONS[int(predicted_label)], (x - 10, y - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
		#print(resized_face.shape)
		#print(predicted_label)
	#cv2.put
	cv2.imshow('frame',frame)
	c = cv2.waitKey(1)
	if c!=-1:
		cv2.imwrite("f"+str(i)+".jpg",frame)
	i=i+1	 	
cap.release()
cv2.destroyAllWindows()