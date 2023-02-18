import sklearn
from sklearn import datasets
from sklearn import svm, metrics
from sklearn.neighbors import KNeighborsClassifier



cancer = datasets.load_breast_cancer()

x =  cancer.data
y=  cancer.target 

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2)



classes = ["malignant","benign"]


# Support vector classification

# ONe can use multiple kernels
# Linear
# poly 
# sigmoid 
# etc

clf = svm.SVC(kernel="linear")

clf.fit(x_train,y_train)

y_pred = clf.predict(x_test)


acc = metrics.accuracy_score(y_test,y_pred)


print(acc)


# K nearest neighbours.. Test performanc

clf = KNeighborsClassifier(n_neighbors=15)

clf.fit(x_train,y_train)

y_pred = clf.predict(x_test)


acc = metrics.accuracy_score(y_test,y_pred)


print(acc)


# For this dataset, svm is way performant.
