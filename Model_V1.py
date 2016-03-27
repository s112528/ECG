#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Only py3 / so that 2 / 3 = 0.66..
from __future__ import division
# Only py3 string encoding
from __future__ import unicode_literals
# Only py3 print
from __future__ import print_function

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


# Added libraries
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap

# Scikit-learn libraries 
from sklearn.ensemble import RandomForestRegressor
from sklearn import linear_model
from sklearn.svm import SVR


" We first get back the useful data"
# Parameters to adjust if required  !!!
nb_subjects = 23
nb_interval = 25 #nb of intervals per session
nb_features = 16

X_Session1 =np.zeros((nb_subjects,nb_interval,nb_features))
X_Session4 =np.zeros((nb_subjects,nb_interval,nb_features))
X_Session7 =np.zeros((nb_subjects,nb_interval,nb_features))
X_raw = np.zeros((nb_subjects, 3*nb_interval, nb_features)) # X_raw is a table of size nb subjects, the nb of intervals in the 3 sessions per subject, the number of features
X_shuffled =  np.zeros((nb_subjects, 3*nb_interval, nb_features)) 

y_Session1 =np.zeros((nb_subjects,nb_interval))
y_Session4 =np.zeros((nb_subjects,nb_interval))
y_Session7 =np.zeros((nb_subjects,nb_interval))
y_raw = np.zeros((nb_subjects, 3*nb_interval)) # y_raw is a table of size nb subjects, the nb of intervals in the 3 sessions per subject
y_shuffled = np.zeros((nb_subjects, 3*nb_interval))

subject = np.array([8, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20, 21, 22, 23, 25, 26, 27, 29, 31, 32, 33, 34, 35])
session = np.array([1, 4, 7])

path_begin_X = 'C:/Users/Florent/Documents/ULG/2ème Master/Master thesis/Data/ExcelFilesFeaturesFFT/Sujet'
path_begin_y = 'C:/Users/Florent/Documents/ULG/2ème Master/Master thesis/Data/y_model/Sujet'
path_end= '.xls'


for i in range(0,len(subject)):
	for j in range(0,len(session)):
		curr_path_X = path_begin_X + str(subject[i]) + '-' + str(session[j]) + path_end
		curr_path_y = path_begin_y + str(subject[i]) + '-' + str(session[j]) + path_end
		df_X = pd.read_excel(curr_path_X)
		val_X = df_X.values
		df_y = pd.read_excel(curr_path_y)
		val_y = df_y.values
		if session[j] == 1:
			X_Session1[i,:,:] = val_X
			y_Session1[i,:] = np.transpose(val_y)
		elif session[j] == 4:
			X_Session4[i,:,:] = val_X
			y_Session4[i,:] = np.transpose(val_y)
		else :
			X_Session7[i,:,:] = val_X
			y_Session7[i,:] = np.transpose(val_y)

	X_raw[i,:,:] = np.concatenate((X_Session1[i,:,:], X_Session4[i,:,:], X_Session7[i,:,:]), axis=0)
	y_raw[i,:] = np.concatenate((y_Session1[i,:], y_Session4[i,:], y_Session7[i,:]), axis=0)



" We buid the model"

ind_subject_shuffled = np.random.permutation(len(subject))
subject_shuffled = subject[ind_subject_shuffled]
X_shuffled[:,:,:] = X_raw[ind_subject_shuffled,:,:]
y_shuffled[:,:] = y_raw[ind_subject_shuffled,:]


nb_taken = 3
X_test = np.zeros((nb_taken*3*nb_interval,nb_features))
X_train = np.zeros(( (len(subject)-nb_taken)*3*nb_interval,nb_features))
X_test = np.concatenate((X_shuffled[0:nb_taken,:,:]), axis = 0)
y_test = np.concatenate((y_shuffled[0:nb_taken,:]), axis = 0)

X_train = np.concatenate((X_shuffled[nb_taken:len(subject),:,:]), axis = 0)
y_train = np.concatenate((y_shuffled[nb_taken:len(subject),:]), axis = 0)

features_names = ['mean_RR', 'SDNN', 'mean_HR', 'mean_HRI', 'RMSSD', 'NN50', 'pNN50', 'SD1', 'SD2', 'LF_power', 'HF_power', 'LFoverHF', 'LF_norm', 'HF_norm', 'LF_f_max', 'HF_f_max']

# We keep a part of the features only to test if better or not
#X_test = X_test[:,[1,4,5,7,8,11,12,13]] # We take SDNN, RMSSD, NN50, SD1, SD2, LH/HF, LF_norm, HF_norm
#X_train = X_train[:,[1,4,5,7,8,11,12,13]] # We take SDNN, RMSSD, NN50, SD1, SD2, LH/HF, LF_norm, HF_norm

# We shuffle the sesions and the subjects in the train 
ind_train_shuffled = np.random.permutation(len(X_train))  
X_train = X_train[ind_train_shuffled]
y_train = y_train[ind_train_shuffled]






##################
# Random Forest  #
##################
forest = RandomForestRegressor(n_estimators=250) 
forest.fit(X_train, y_train)
y_predicted_forest = forest.predict(X_test)
score_forest = forest.score(X_test, y_test)
print("FOREST")
print("The score obtained is equal to : ", score_forest) 

# The mean square error
print("Residual sum of squares: %.2f" % np.mean((forest.predict(X_test) - y_test) ** 2))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % forest.score(X_test, y_test)) 




# To know the importance of the features
importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(X_train.shape[1]):
	print("%d. feature %d = %s (%f)" % (f + 1, indices[f], features_names[indices[f]], importances[indices[f]]))


# Plot the feature importances of the forest
#plt.figure()
#plt.title("Feature importances")
#plt.bar(range(X_train.shape[1]), importances[indices], color="r", yerr=std[indices], align="center")
#plt.xticks(range(X_train.shape[1]), indices)
#plt.xlim([-1, X_train.shape[1]])
#plt.show()


#####################
# Linear regression #
##################### 
regr = linear_model.LinearRegression()
regr.fit(X_train,y_train)
score_regr = regr.score(X_test, y_test)
y_predicted_lin = regr.predict(X_test)


print("LINEAR REGRESSION")
print("The score obtained is equal to : ", score_regr) 
# The mean square error
print("Residual sum of squares: %.2f" % np.mean((regr.predict(X_test) - y_test) ** 2)) # http://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html#example-linear-model-plot-ols-py
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % regr.score(X_test, y_test)) 


##############
# SVM (poly) #
##############

#svm_poly = SVR(kernel = 'polynomial', degree=2)
#svm_poly.fit(X_train, y_train)
#score_svm_poly = svm_poly.score(X_test, y_test) 

print("SVM REGRESSION")
for i in range(1,6):
	print("degree :",i)
	svm_poly = SVR(kernel = 'poly', degree=i)
	svm_poly.fit(X_train, y_train)
	score_svm_poly = svm_poly.score(X_test, y_test)
	print("The score obtained is equal to : %.2f" %score_svm_poly) 
	# The mean square error
	print("Residual sum of squares: %.2f" % np.mean((svm_poly.predict(X_test) - y_test) ** 2)) # http://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html#example-linear-model-plot-ols-py
	# Explained variance score: 1 is perfect prediction
	print('Variance score: %.2f' % svm_poly.score(X_test, y_test))   
