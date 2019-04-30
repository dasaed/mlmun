# -*- coding: utf-8 -*-

# Authors: Dalia Ibrahim, Carlos Salcedo
import itertools
import pandas as pd
import numpy as np
import math
import sys
import time

def FoldSplitter(kfolds):
    #print("### Splitting Training Data  ###")
    mydata = pd.read_csv(sys.argv[1], sep='\t', header=None,)
    numcolMydata = mydata.shape[1] #columns
    #print("Number of Columns: "+str(numcolMydata))
    numrowMydata = mydata.shape[0] #rows
    #print("Number of Rows: "+str(numrowMydata))
    cvfolds = kfolds
    #print("Number of CV Folds: "+str(cvfolds))
    partSize = numrowMydata/cvfolds
    #print("Length of each fold = "+str(partSize))
    class0= mydata.loc[mydata.iloc[:,-1]==0] # filter by class 0
    class1= mydata.loc[mydata.iloc[:,-1]==1] # filter by class 1
    class0 = class0.sample(frac=1).reset_index(drop=True) # Shuffle values of class0
    numclass0 = class0.shape[0]
    partSizeClass0 = numclass0/cvfolds
    #print("Number of Class 0: "+str(class0.shape[0]))
    #print("Number of Class 0 per Fold: "+str(partSizeClass0))
    class1 = class1.sample(frac=1).reset_index(drop=True) # Shuffle values of class1
    numclass1 = class1.shape[0]
    partSizeClass1 = numclass1/cvfolds
    #print("Number of Class 1: "+str(class1.shape[0]))
    #print("Number of Class 1 per Fold: "+str(partSizeClass1))
    leftovers = pd.DataFrame()
    leftoversC0 = class0[(partSizeClass0*cvfolds): ]
    leftoversC1 = class1[(partSizeClass1*cvfolds): ]
    leftovers = pd.concat([leftoversC0,leftoversC1],ignore_index=True, axis=0)
    theFolds = {}
    for i in range(cvfolds):
        fold = 'fold'+str(i)
        class0start = i * partSizeClass0
        class0end = (i * partSizeClass0) + partSizeClass0
        class1start = i * partSizeClass1
        class1end = (i * partSizeClass1) + partSizeClass1
        part1 = class0[class0start:class0end]
        part2 = class1[class1start:class1end]
        if i == cvfolds-1:
            tempart = [part1,part2,leftovers]
        else:
            tempart = [part1,part2]
        fullpart = pd.concat(tempart,ignore_index=True, axis=0)
        theFolds[fold] = fullpart
        #print('############################### Iteration '+str(i)+' ###############################')
        #print(fullpart)
        print(theFolds.keys())
        #print(part1.head(2))
        #print(part2.tail(2))
        #print(fullpart.head(2))
        #print(fullpart.tail(2))
        #print('#######################################################################################')
        #fold = pd.concat(class0[class0start:class0end],class1[class1start:class1end])
    # include left overs in last fold.
    print('#######################################################################################')
    #print(theFolds['fold9'])
    #print(str(leftovers))
    return (theFolds)

#############################################################################
def SplitData(dictOfFolds, iterNum):
    testfold = 'fold'+str(iterNum)
    testPart = dictOfFolds.get(testfold,'Something is wrong, the iterNum does not correspond to a valid fold')
    testDf = pd.DataFrame()
    testDf = testDf.append(testPart) #Create my Test DataFrame fold
    #dictOfFolds.pop(testfold) # Delete Test Dataframe from total folds
    trainkeys = dictOfFolds.keys()
    print(trainkeys)
    trainDf = pd.DataFrame() #create Training/Learing Dataframe
    print("#######################################################################################################")
    print("Entering Learning Fold Concatenation: ")
    for foldKey, foldValue in dictOfFolds.items():
        if foldKey == testfold:
            continue
        newdf = pd.DataFrame(foldValue)
        trainDf = pd.concat([trainDf,newdf],ignore_index=True)
        print(foldKey)
    #export_csv = trainDf.to_csv(r'export_dataframe.csv', index=None, header=False)  # Don't forget to add '.csv' at the end of the path
    xlearning = trainDf.iloc[:, :int(trainDf.shape[0]-2)] # all columns except the last one
    ylearning = trainDf.iloc[:, int(trainDf.shape[0]-2):] # last column
    print("Train Data Frame Full:")
    print(trainDf.head(3))
    print("Train Data Frame NO CLASS")
    xtraining = trainDf.iloc[:, :int(trainDf.shape[1]-1)]#all rows, length of column -2
    print(xtraining.head(3))
    print("Train Data Frame ONLY the class")
    ytraining = trainDf.iloc[:, int(trainDf.shape[1]-1):]
    print(ytraining.head(3))
    print("Test Data Frame Full:")
    print(testDf.head(3))
    print("Test Data Frame NO CLASS")
    xtesting = testDf.iloc[:, :int(testDf.shape[1]-1)]#all rows, length of column -2
    print(xtesting.head(3))
    print("Test Data Frame ONLY the Class:")
    ytesting = testDf.iloc[:, int(testDf.shape[1]-1):]
    print(ytesting.head(3))
    return xtraining, xtesting, ytraining, ytesting
    #return X_train, X_test, y_train, y_test
#############################################################################
#foldDict = FoldSplitter(N) #Partitions the data into n folds
foldDict = FoldSplitter(10) #Partitions the data into n folds
xl0,xt0,yl0,yt0 = SplitData(foldDict,7)
xl1,xt1,yl1,yt1 = SplitData(foldDict,8)
xl2,xt2,yl2,yt2 = SplitData(foldDict,9)
print("Final Test")
print(foldDict.keys())
print("Fold 7 Xlearning length = "+str(xl0.shape[0]))
print("Fold 7 Xtraining length = "+str(xt0.shape[0]))
print("Fold 7 Ylearning length = "+str(yl0.shape[0]))
print("Fold 7 Ytesting length = "+str(yt0.shape[0]))
print("Fold 8 Xlearning length = "+str(xl1.shape[0]))
print("Fold 8 Xtraining length = "+str(xt1.shape[0]))
print("Fold 8 Ylearning length = "+str(yl1.shape[0]))
print("Fold 8 Ytesting length = "+str(yt1.shape[0]))
print("Fold 9 Xlearning length = "+str(xl2.shape[0]))
print("Fold 9 Xtraining length = "+str(xt2.shape[0]))
print("Fold 9 Ytesting length = "+str(yl2.shape[0]))
print("Fold 9 Ytesting length = "+str(yt2.shape[0]))
print("The algorithm is working, but the last fold will always have the leftover samples after the data has been divided.")
