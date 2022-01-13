import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

dataset = pd.read_csv('heart.csv')

X = dataset.iloc[:, :13]

y = dataset.iloc[:, -1]


from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 8)
classifier.fit(X, y)
pickle.dump(classifier, open('model2.pkl','wb'))
model = pickle.load(open('model2.pkl','rb'))


