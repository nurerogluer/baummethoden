#Import Pandas library
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.tree import DecisionTreeClassifier
import pickle 
from sklearn.metrics import classification_report, confusion_matrix  
from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
#from pathlib import Path


#List with attribute names (it is optional to do this but it gives a better understanding of the data for a human reader)
attribute_names = ['variance_wavelet_transformed_image', 'skewness_wavelet_transformed_image', 'curtosis_wavelet_transformed_image', 'entropy_image', 'class']

#Read csv-file
data = pd.read_csv('data/data_banknote_authentication.csv', names=attribute_names, header=None)


#Shuffle data


data = data.sample(frac=1)

#Shows pytho
# the first 5 rows of the data
data.head()
#'class'-column
y_variable = data['class']

#all columns that are not the 'class'-column -> all columns that contain the attributes
x_variables = data.loc[:, data.columns != 'class']

#splits into training and test data
x_train, x_test, y_train, y_test = train_test_split(x_variables, y_variable, test_size=0.2)

# shapes of our data splits

#Create a classifier object 
classifier = DecisionTreeClassifier() 

#Classfier builds Decision Tree with training data
classifier = classifier.fit(x_train, y_train) 

#Shows importances of the attributes according to our model 
classifier.feature_importances_


#filename='classifier_decision_tree.pickle'

#f = open(filename, 'wb')
#pi.dump(classifier, f)
#f.close()
with open('classifier_decision_tree.pkl', 'wb') as f:
    pickle.dump(classifier, f)

#prediciton decisiontree
y_pred = classifier.predict(x_test) 
#Create the matrix that shows how often predicitons were done correctly and how often theey failed.
conf_mat = confusion_matrix(y_test, y_pred)
#The diagonal ones are the correctly predicted instances. The sum of this number devided by the number of all instances gives us the accuracy in percent.
accuracy = (conf_mat[0,0] + conf_mat[1,1]) /(conf_mat[0,0]+conf_mat[0,1]+ conf_mat[1,0]+conf_mat[1,1])
#Create the matrix that shows how often predicitons were done correctly and how often theey failed.

print('Accuracy Decision Tree: ' + str(round(accuracy,4)))
print('Confusion matrix Decision Tree:')
print(conf_mat)  
print('classification report Decision Tree:')
print(classification_report(y_test, y_pred)) 

print('fertig trainiert Decision Tree')


#################################################
#k_fold object


#k_fold = KFold(n_splits=5, shuffle=True, random_state=0)

#scores reached with different splits of training/test data 
#k_fold_scores = cross_val_score(classifier, x_variables, y_variable, cv=k_fold, n_jobs=1)

#arithmetic mean of accuracy scores 
#mean_accuracy = np.mean(k_fold_scores)

#print(round(mean_accuracy, 4))

################################################################################################

#tree parameters which shall be tested
tree_para = {'criterion':['gini','entropy'],'max_depth':[i for i in range(1,20)], 'min_samples_split':[i for i in range (2,20)]}

#GridSearchCV object
grd_clf = GridSearchCV(classifier, tree_para, cv=5)

#creates differnt trees with all the differnet parameters out of our data
grd_clf.fit(x_variables, y_variable)

#best paramters that were found
best_parameters = grd_clf.best_params_  
print(best_parameters)  

#new tree object with best parameters
model_with_best_tree_parameters = grd_clf.best_estimator_

#k_fold object
k_fold = KFold(n_splits=5, shuffle=True, random_state=0)

#scores reached with different splits of training/test data 
k_fold_scores = cross_val_score(model_with_best_tree_parameters, x_variables, y_variable, cv=k_fold, n_jobs=1)

#arithmetic mean of accuracy scores 
mean_accuracy_best_parameters_tree = np.mean(k_fold_scores)

print(round(mean_accuracy_best_parameters_tree, 4))








#arithmetic mean of accuracy scores 
mean_accuracy = np.mean(k_fold_scores)
print(' ')
print('----------------------')
print('       k_fold         ')
print('----------------------')
print('accuracy k_fold',round(mean_accuracy, 4))

#with open('k_fold.pkl', 'wb') as f:
#    pickle.dump(k_fold, f)



#RandomForestClassifier object
random_forest_classifier = RandomForestClassifier(n_estimators=10)

#list with accuracies with different test and training sets of Random Forest
accuracies_rand_forest = cross_val_score(random_forest_classifier, x_variables, y_variable, cv=k_fold, n_jobs=1)

##arithmetic mean of the list with the accuracies of the Random Forest
accuracy_rand = np.mean(accuracies_rand_forest)

#prediciton decisiontree

random_forest_classifier = random_forest_classifier.fit(x_train, y_train) 
y_pred_rand = random_forest_classifier.predict(x_test) 

#Create the matrix that shows how often predicitons were done correctly and how often theey failed.
conf_mat_rand = confusion_matrix(y_test, y_pred_rand)
#The diagonal ones are the correctly predicted instances. The sum of this number devided by the number of all instances gives us the accuracy in percent.
accuracy_rand= (conf_mat_rand[0,0] + conf_mat_rand[1,1]) /(conf_mat_rand[0,0]+conf_mat_rand[0,1]+ conf_mat_rand[1,0]+conf_mat_rand[1,1])
print('Accuracy Random Forest: ' + str(round(accuracy_rand,4)))
print('Confusion matrix Random Forest:')
print(conf_mat_rand)  
print('classification report Random Forest:')
print(classification_report(y_test, y_pred)) 

print('fertig trainiert Random Forest')


with open('random_forest_classifier.pkl', 'wb') as f:
    pickle.dump(random_forest_classifier, f)

#filename='random_forest_classifier.pickle'  
#f = open(filename, 'wb')
#pickle.dump(random_forest_classifier, f)
#f.close()


print(' ')
print('-----------------------------')
print('       Random Forest         ')
print('-----------------------------')
print('Accuracy Random Forest ' + str(round(accuracy_rand,4)))
print('Old accuracy k_fold: ' + str(round(mean_accuracy,4)))
print('Accuracy Decision Tree: ' + str(round(accuracy,4)))
#print('Best tree accuracy: ' + str(round(mean_accuracy_best_parameters_tree,4)))
print('-----------------------------')
print('-----------------------------')