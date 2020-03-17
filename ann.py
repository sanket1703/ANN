import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3: 13].values
y = dataset.iloc[:, 13].values

# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X =X[:,1:]
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting classifier to the Training set
# Create your classifier here
import keras 
from keras.models import Sequential
from keras.layers import Dense
# Predicting the Test set results

# we are making a classifier 



#number of neurons we have in the first input layer is the 
#number of independent features in the test set 

classifier.add(Dense(units = 6, kernel_initializer = 'uniform',activation = 'relu',input_dim = 11 ))
classifier.add(Dense(units = 6, kernel_initializer = 'uniform',activation = 'relu'))
classifier.add(Dense(units = 1, kernel_initializer = 'uniform',activation = 'sigmoid'))
classifier.compile(optimizer = 'adam',loss = 'binary_crossentropy',metrics = ['accuracy'])
classifer.fit(X_train,y_train, batch_size = 10, nb_epoch)
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
