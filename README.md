# Emotion_Recognition

# About Dataset:

Dataset Name: fer2013

Image Resolution: 48x48x3

No of samples: 35887

Categories: (0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral)

Download Link: https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge

# Description:

1.Downsample the images to 28x28x1 resolution

2.CNN_Model: (conv->relu->Batch_Norm->pool)x2->FC_Layer->relu->FC_Layer->Output

3.Used 31000 samples for training

4.1000 samples for testing

5.Number of epochs: 50

6.Training Accuracy: 99.5%

7.Test Accuracy: 41%

# Performance Output:
![f2207](https://user-images.githubusercontent.com/29327349/30575523-f9469080-9d1e-11e7-8d8f-764a16f7aa75.jpg)
![f2737](https://user-images.githubusercontent.com/29327349/30575540-2294cdee-9d1f-11e7-88b4-bfef616dfbcb.jpg)
![f2846](https://user-images.githubusercontent.com/29327349/30575543-2f6b5ee8-9d1f-11e7-99f6-28ed41145ecc.jpg)
![f4232](https://user-images.githubusercontent.com/29327349/30575551-3ced8438-9d1f-11e7-861a-e487d5c44914.jpg)
![f6406](https://user-images.githubusercontent.com/29327349/30575557-4907f910-9d1f-11e7-9bd3-5be6915b05f5.jpg)
![f7250](https://user-images.githubusercontent.com/29327349/30575565-56c4d5fa-9d1f-11e7-9fe3-c2eaef7b74bd.jpg)
References:
van Gent, P. (2016). Emotion Recognition With Python, OpenCV and a Face Dataset. A tech blog about fun things with Python and embedded electronics.
