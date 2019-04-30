import numpy as np
from sklearn.model_selection import cross_val_score, train_test_split
X = np.array([[1, 3], [3, 7], [2, 4], [4, 8],[43, 84],[32, 73]])
y = np.array([0, 1, 1, 1,1,0])
#stratSplit = StratifiedShuffleSplit(y, 1, test_size=0.5,random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
#StratifiedShuffleSplit(y, n_iter=1, test_size=0.5)
# for train_idx,test_idx in stratSplit:
#     X_train=X[train_idx]
#     y_train=y[train_idx]
print(X_train[1])
print(y_train[1])
# print (stratSplit)
