# https://www.kaggle.com/rupals/classification-svm-decision-trees-boosting
#import statements
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, learning_curve, KFold
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler
import random
from sklearn.svm import SVC
import sklearn.metrics as sk
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import itertools
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

#data
data=pd.read_csv(r'/home/ram/Downloads/kaggle/iris.data.csv')
print(data.info())
print(list(data.columns))
print(data.isnull().sum())

y=data['Iris-setosa']
x=data.drop(columns='Iris-setosa')
# linear svm kernal
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size =.33,random_state=0)
svm = SVC(gamma='auto', kernel='linear', probability=True)
svm.fit(x_train, y_train)
y_pre = svm.predict(x_test)
accuracy = accuracy_score(y_test,y_pre)
print("SVM-linear: decision tree:Accuracy on test data for a given model is {}".format(accuracy))

# confussion matrix
print("confussion matrix")
print(confusion_matrix(y_test,y_pre))
# classification report
# classification report

print("classification report")
print(classification_report(y_test,y_pre))


# polynomial svm kernal
svm = SVC(gamma='auto', kernel='poly', probability=True)
svm.fit(x_train, y_train)
y_pre = svm.predict(x_test)
accuracy =accuracy_score(y_test,y_pre)
print("SVM-poly: decision tree:Accuracy on test data for a given model is {}".format(accuracy))

# confussion matrix
print("confussion matrix")
print(confusion_matrix(y_test,y_pre))
# classification report
# classification report
print("classification report")

print(classification_report(y_test,y_pre))
# rbf svm kernal

svm = SVC(gamma='auto', kernel='poly', probability=True)
svm.fit(x_train, y_train)
y_pre = svm.predict(x_test)
accuracy = accuracy_score(y_test,y_pre)
print("SVM-rbf: decision tree:Accuracy on test data for a given model is {}".format(accuracy))

# confussion matrix
print("confussion matrix")
print(confusion_matrix(y_test,y_pre))
# classification report
# classification report
print("classification report")

print(classification_report(y_test,y_pre))