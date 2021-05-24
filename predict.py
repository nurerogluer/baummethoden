import pandas as pd
import os
import pickle
from sklearn.metrics import classification_report, confusion_matrix
#from train import x_test,y_test,classifier,attribute_names
from sklearn.tree import export_graphviz
from sklearn.metrics import accuracy_score
#from subprocess import check_call
#Export as dot file##
#export_graphviz(classifier, out_file='tree_bills.dot', class_names = ['0','1'], feature_names = attribute_names[0:4])
data = pd.read_csv('data\predict_datas.csv', header=None)
#Export dot to png 
x=data.loc[:,data.columns != 'class']
print(x)
############################
# load Decision Tree model

with open('classifier_decision_tree.pkl', 'rb') as f:
    loaded_classifier = pickle.load(f)
y_predDT = loaded_classifier.predict(x)
print(' ')
print('----------------------')
print('Decision Tree',y_predDT)
print('----------------------')
print(' ')

with open('random_forest_classifier.pkl', 'rb') as f:
    loaded_RFclassifier = pickle.load(f)
y_predRF = loaded_RFclassifier.predict(x)
print(' ')
print('----------------------')
print('Random Forest',y_predRF)
print('----------------------')
print(' ')



#with open('k_fold.pkl', 'rb') as f:
#    loaded_kfold = pickle.load(f)
#y_predKF = loaded_kfold.predict(x)
#print(' ')
#print('----------------------')
#print('K_Fold',y_predKF)
#print('----------------------')
#print(' ')