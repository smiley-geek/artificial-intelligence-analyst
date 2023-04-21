import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
import matplotlib.pyplot as plt



from sklearn import svm, metrics
from sklearn.neighbors import KNeighborsClassifier



# Prepare the data

# Read the data
data= pd.read_csv("sober.csv",sep=",")

# Separate the columns
data = data[["behavior_sexualRisk" ,"behavior_eating" ,"behavior_personalHygine" ,"intention_aggregation" ,"intention_commitment" ,"attitude_consistency" ,"attitude_spontaneity" ,"norm_significantPerson" ,"norm_fulfillment" ,"perception_vulnerability" ,"perception_severity" ,"motivation_strength" ,"motivation_willingness" ,"socialSupport_emotionality" ,"socialSupport_appreciation" ,"socialSupport_instrumental" ,"empowerment_knowledge" ,"empowerment_abilities" ,"empowerment_desires" ,"ca_cervix"]]

# Identify the label column
predict = "ca_cervix"

# Remove the label column from the training data
x = np.array(data.drop(columns=[predict],axis=1))
# The label column
y = np.array(data[predict])

# Split the data into test and training 

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x,y,test_size=0.1)


# Linear Model

print("\nLINEAR REGRESSION\n")

linearModel = None
acc = 0

# Train the model 5 times and return the best model


for _ in range(5):    
    trainLinearModel = linear_model.LinearRegression()
    trainLinearModel.fit(x_train,y_train)
    current_acc = trainLinearModel.score(x_test,y_test)
        
    if current_acc > acc:
        acc = current_acc
        linearModel = trainLinearModel

            
                
print("Linear model has an accuracy of ", acc*100, end="%\n\n")

# Some predictions

def linearPredictions():
    predictions = linearModel.predict(x_test)

    for x in range(len(predictions)):

        print(f'Prediction: {predictions[x]}\n x values {x_test[x]}\n actual answer {y_test[x]}\n\n')
        
# linearPredictions()



# SVM

print("\nSVM\n")

# Identify the classes  
classes = ["0","1"]


# Support vector classification

# One can use multiple kernels
# Linear
# poly 
# sigmoid 
# etc


svmModel = None
acc = 0

# Train the model 5 times and return the best model

for _ in range(5):       
    trainSvm = svm.SVC(kernel="linear")
    trainSvm.fit(x_train,y_train)
    y_pred = trainSvm.predict(x_test)
    current_acc = metrics.accuracy_score(y_test,y_pred)
    
    if current_acc > acc:
        acc = current_acc
        svmModel = trainSvm

print("SVM model has an accuracy of ", acc*100, end="%\n\n")

def svmPredictions():
    predictions = svmModel.predict(x_test)

    for x in range(len(predictions)):

        print(f'Prediction: {predictions[x]}\n x values {x_test[x]}\n actual answer {y_test[x]}\n\n')
        
# svmPredictions()



# K nearest neighbours

print("\nK NEAREST NEIGHBOURS\n")

knnModel = None
acc = 0

# Train the model 5 times and return the best model

for _ in range(5):        
    trainKnn = KNeighborsClassifier(n_neighbors=1)
    trainKnn.fit(x_train,y_train)
    y_pred = trainKnn.predict(x_test)
    current_acc = metrics.accuracy_score(y_test,y_pred)    
    if current_acc > acc:
        acc = current_acc
        knnModel = trainSvm
        

print("KNN model has an accuracy of ", acc*100, end="%\n\n")

def knnPredictions():
    predictions = trainKnn.predict(x_test)

    for x in range(len(predictions)):
        print(f'Prediction: {predictions[x]}\n x values {x_test[x]}\n actual answer {y_test[x]}\n\n')
   
# knnPredictions()     


# Summary
# Svm is performant when the kernel is linear or rbf
# Knn is performant when n_neighbors is between 3 - 7 
