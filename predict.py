from sklearn.metrics import classification_report, confusion_matrix
from train import x_test,y_test,classifier,attribute_names
from sklearn.tree import export_graphviz
from subprocess import check_call
#Export as dot file
export_graphviz(classifier, out_file='tree_bills.dot', class_names = ['0','1'], feature_names = attribute_names[0:4])

#Export dot to png 

#check_call(['dot','-Tpng','tree_bills.dot','-o','tree_bills.png'])

#prediciton decisiontree
y_pred = classifier.predict(x_test) 


#Create the matrix that shows how often predicitons were done correctly and how often theey failed.
conf_mat = confusion_matrix(y_test, y_pred)

#The diagonal ones are the correctly predicted instances. The sum of this number devided by the number of all instances gives us the accuracy in percent.
accuracy = (conf_mat[0,0] + conf_mat[1,1]) /(conf_mat[0,0]+conf_mat[0,1]+ conf_mat[1,0]+conf_mat[1,1])

print('Accuracy: ' + str(round(accuracy,4)))
print('Confusion matrix:')
print(conf_mat)  
print('classification report:')
print(classification_report(y_test, y_pred)) 