import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
# pip install scikit-learn

dataset = datasets.load_iris()

model = GaussianNB()
model.fit(dataset.data,dataset.target)

expected = dataset.target
predicted = model.predict(dataset.data)

print(metrics.classification_report(expected,predicted))
print(metrics.confusion_matrix(expected,predicted))
