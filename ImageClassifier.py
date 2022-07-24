import sys
print ('python: {}'.format(sys.version))

import pandas
from pandas import read_csv
from pandas.plotting import scatter_matrix
from sklearn.model_selection import train_test_split
from matplotlib import pyplot
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model  import LogisticRegression
from sklearn.tree  import DecisionTreeClassifier
from sklearn.neighbors  import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm  import SVC
from sklearn import model_selection
from sklearn.ensemble import VotingClassifier

#Loading the data
url="https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width','class']
dataset=read_csv(url,names=names)

#Exploring the dimension of the dataset
print(dataset.shape)

#Taking a peek at the data
print(dataset.head(20))

#Statistical summary
print(dataset.describe())

#Class distribution
print(dataset.groupby('class').size())

#Univariate plot - box and whisker plots
dataset.plot(kind='box',subplots=True,layout=(2,2),sharex=False,sharey=False)
pyplot.show

#Histogram of the variable
dataset.hist()
pyplot.show()

#Multiariate plots
scatter_matrix(dataset)
pyplot.show()

#Creating validation dataset
#Splitting the dataset
array= dataset.values
X=array[:,0:4]
y=array[:,4]
X_train,X_validation,Y_train,Y_validation= train_test_split(X,y,test_size=0.2,random_state=1)

#Logistic Regression
#Linear Discriminant Analysis
# K-Nearest Neighbors
#Classification and Regression
#Gaussian Naive Bayes
# Support Vector Machine

#Building models
models=[]
models.append(('LR',LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN',KNeighborsClassifier()))
models.append(('NB',GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))

#Evaluate the created models
results=[]
names=[]

for name, model in models:
    kfold= StratifiedKFold(n_splits=10,random_state=1,shuffle=True)
    cv_results=cross_val_score(model,X_train,Y_train,cv=kfold,scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    print('%s: %f (%f)' % (name, cv_results.mean() ,cv_results.std()))

#Compare our models
pyplot.boxplot(results,labels=names)
pyplot.title('Algorithm Comparision')
pyplot.show()

#Make predictions on svm
model= SVC(gamma='auto')
model.fit(X_train, Y_train)
predictions=model.predict(X_validation)

#Evaluate our predictions
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))