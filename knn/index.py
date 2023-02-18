import pandas as pd
import numpy as np
import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier

from sklearn import linear_model, preprocessing

data = pd.read_csv("car.csv")
print(data.head())

pre = preprocessing.LabelEncoder()


buying = pre.fit_transform(list(data["buying"]))
maint = pre.fit_transform(list(data["maint"]))
door = pre.fit_transform(list(data["door"]))
persons = pre.fit_transform(list(data["persons"]))
lug_boot = pre.fit_transform(list(data["lug_boot"]))
safety = pre.fit_transform(list(data["safety"]))
cls = pre.fit_transform(list(data["class"]))




predict = "class"

x = list(zip(buying,maint,door,persons,lug_boot,safety))
y  = list(cls)

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.1)


model = KNeighborsClassifier(n_neighbors=7)

model.fit(x_train,y_train )


acc = model.score(x_test,y_test)

predict = model.predict(x_test)

print(acc)

names = ['unacc',"acc","good","vgood"]

for x in range(len(predict)):
    if predict[x]  != y_test[x]:
        print("Ha!")
    print("Predicted data: ", predict[x], "data: " , x_test[x], "Actual data: ",y_test[x] )
   