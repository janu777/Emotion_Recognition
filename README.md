# Emotion_Recognition

# About Dataset:

Dataset Name: fer2013

Image Resolution: 48x48x3

No of samples: 35887

Categories: (0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral)

Download Link: https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge

# Dependencies:

1.https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml

2.TensorFlow: https://www.tensorflow.org/install/install_linux

3.Python

# Description:

1.Downsample the images to 28x28x1 resolution

2.CNN_Model: (conv->relu->Batch_Norm->pool)x2->FC_Layer->relu->FC_Layer->Output

3.Used 31000 samples for training

4.1000 samples for testing

5.Number of epochs: 50

6.Training Accuracy: 99.5%

7.Test Accuracy: 41%

# Usage:

python main.py

# Performance Output:
![f110](https://user-images.githubusercontent.com/29327349/30577575-cc2c0252-9d2c-11e7-8664-cac6c92349dd.jpg)
![f244](https://user-images.githubusercontent.com/29327349/30577576-cf73c1f2-9d2c-11e7-8666-980b3bfed617.jpg)
![f1414](https://user-images.githubusercontent.com/29327349/30577578-d2b87aba-9d2c-11e7-8ba0-c7a6db366ed2.jpg)
![f3252](https://user-images.githubusercontent.com/29327349/30577581-d5abae5e-9d2c-11e7-911e-b8d932e0ea1b.jpg)
![f339](https://user-images.githubusercontent.com/29327349/30577585-da63b630-9d2c-11e7-96fd-29bfec0980f9.jpg)
![f924](https://user-images.githubusercontent.com/29327349/30577587-df45977c-9d2c-11e7-96f9-c2b00dda3fe7.jpg)

# References:

van Gent, P. (2016). Emotion Recognition With Python, OpenCV and a Face Dataset. A tech blog about fun things with Python and embedded electronics.
