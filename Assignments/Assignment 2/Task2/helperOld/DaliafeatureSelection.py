# from sklearn.datasets import load_iris
# from sklearn.pipeline import Pipeline
# from sklearn.grid_search import GridSearchCV
# from sklearn.linear_model import LogisticRegression
# from sklearn.feature_selection import SelectKBest, f_classif
import pandas as pd
import numpy as np
# import math
# import sys
# import os
# import random
# import time
# 
# iris = load_iris()
# X, y = iris.data, iris.target
# kbest = SelectKBest(f_classif)
# pipeline = Pipeline([('kbest', kbest), ('lr', LogisticRegression())])
# grid_search = GridSearchCV(pipeline, {'kbest__k': [1,2,3,4], 'lr__C': np.logspace(-10, 10, 5)})
# newData=grid_search.fit(X, y)
# print ( newData)


from sklearn.feature_selection import VarianceThreshold
X = [[0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 1, 1], [0, 1, 0], [0, 1, 1]]
print (X)
sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
array=sel.fit_transform(X)
print (array)
newData=pd.DataFrame(array)
print (newData)