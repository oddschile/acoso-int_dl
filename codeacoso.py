from keras.models import Sequential
from keras.layers import Dense
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
import pandas as pd



epv = pd.read_excel("http://oddschile.net//edadepv.xlsx")



sns.heatmap(epv.corr(), annot=True)



variables = ['Escolaridad', 'Edad', 'ptje_autoestima','ptje_depresion']

X = epv[variables] 
y = epv.acoso_int

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)
X

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)


classifier = Sequential()
classifier.add(Dense(2, activation='relu', kernel_initializer='random_normal', input_dim=4))
classifier.add(Dense(2, activation='relu', kernel_initializer='random_normal'))
classifier.add(Dense(1, activation='sigmoid', kernel_initializer='random_normal'))

classifier.compile(optimizer ='adam',loss='binary_crossentropy', metrics =['accuracy'])

classifier.fit(X_train, y_train, epochs=150, batch_size=10)

_, accuracy = classifier.evaluate(X_train, y_train)
print('Accuracy: %.2f' % (accuracy*100))


eval_model=classifier.evaluate(X_train, y_train)
eval_model


y_pred=classifier.predict(X_test)
y_pred =(y_pred>0.5)

y_pred=classifier.predict(X_test)
y_pred =(y_pred>0.5)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)


 

predictions = classifier.predict(new_epv)

rounded = [round(x[0]) for x in predictions]

predictions = classifier.predict_classes(new_epv)
