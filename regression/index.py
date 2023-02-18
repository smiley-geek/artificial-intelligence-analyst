import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import pickle
from matplotlib import style


data= pd.read_csv("student-mat.csv",sep=";")
data = data[["G1", "G2","G3","studytime","failures","absences"]]


predict = "G3"

x = np.array(data.drop([predict],1))
y = np.array(data[predict])

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x,y,test_size=0.1)


best = 0

# for _ in range(10000):    

#     linear = linear_model.LinearRegression()

#     linear.fit(x_train,y_train)

#     acc = linear.score(x_test,y_test)

    
#     if acc > best:
#         print(best,acc)
#         best = acc
        
#         # Save
#         with open("studentmodel.pickle","wb") as f:
#             pickle.dump(linear,f)
                

    
pickle_in = open("studentmodel.pickle","rb")


linear = pickle.load(pickle_in)


predictions = linear.predict(x_test)

# for x in range(len(predictions)):
#     print(predictions[x],x_test[x],y_test[x])
    
    
p="G1"    
    
style.use("ggplot")
plot = "failures"
plt.scatter(data[plot], data["G3"])
plt.legend("PP",loc=4)
plt.xlabel(plot)
plt.ylabel("Final Grade")
plt.show()




