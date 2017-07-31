#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn.metrics import accuracy_score
from sklearn import svm

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###

#########################################################

def my_svm(features_train, features_test, labels_train, labels_test, kernel, C):
# the classifier
	clf = svm.SVC(kernel=kernel, C=C)
	print "kerner:", kernel
	print "C:",C
#Reduce the Training set, to make algo work faster.
	#features_train=features_train[:len(features_train)/100]
	#labels_train=labels_train[:len(labels_train)/100] 
	
   # train
	t0 = time()
	clf.fit(features_train, labels_train)
	print "\ntraining time:", round(time()-t0, 3), "s"

# predict
	t0 = time()
	pred = clf.predict(features_test)
	print "predicting time:", round(time()-t0, 3), "s"

	accuracy = accuracy_score(pred, labels_test)
	print '\naccuracy = {0}'.format(accuracy)
	print pred[10]
	print pred[26]
	print pred[50]
	
	#There are over 1700 test events--how many are predicted to be in the Chris(1) class?Use the RBF kernel, C=10000., and the full training set.
	#How many Chris emails predicted?
	print "Chris # of mails-", sum(pred)
	
	#How many Sara emails predicted?
	print "Sara # of mails-", 1700-sum(pred)
	
	return pred

# Call the my_svm function
pred = my_svm(features_train, features_test, labels_train, labels_test, 'rbf', 10000.0)	