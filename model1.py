import cv2
import glob
import random
import math
import numpy as np
import dlib
import itertools
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.externals import joblib
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
clf1 = RandomForestClassifier(n_estimators=100)
clf2 = SVC(kernel='linear', probability=True, tol=1e-3)#, verbose = True) #Set the classifier as a support vector machines with polynomial kernel
clf4 = GradientBoostingClassifier(n_estimators=10, learning_rate=1.0,max_depth=None, random_state=0)
clf5 = BaggingClassifier(KNeighborsClassifier(),max_samples=0.5, max_features=0.8)
clf6 = tree.DecisionTreeClassifier()
data = {} #Make dictionary for all values
data=np.load('sorted_train_data.npy')
training_data = data[:30000,:268]
training_labels = data[:30000,268:]
prediction_data = data[30000:33000,:268]
prediction_labels= data[30000:33000,268:]
npar_train = np.array(training_data) #Turn the training set into a numpy array for the classifier
npar_trainlabs = np.array(np.argmax(training_labels,axis=1))
clf3 = KNeighborsClassifier(n_neighbors=16)
clf3.fit(npar_train, npar_trainlabs)
npar_pred = np.array(prediction_data)
npar_predlabs = np.array(np.argmax(prediction_labels,axis=1))
pred_lin = clf3.score(npar_pred, npar_predlabs)
print("linear: ", pred_lin)
joblib.dump(clf3, '/home/linux/Documents/DLExercises/ER2/savemodels/KNN.pkl') 