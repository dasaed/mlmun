# -*- coding: utf-8 -*-

# Authors: Dalia and Carlos

from matplotlib import pyplot
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import auc
from sklearn.metrics import average_precision_score
import pandas as pd
import numpy as np
import math
import sys
import os
import random
import time
#sys.path.append(os.path.abspath('feature_selector.py'))
#import feature_selector
# 
# from .feature_selector import FeatureSelector
# from setuptools import setup
# 
# setup(name="feature_selector",
#         version="N/A",
#         description="FeatureSelector is a class for removing features for a dataset intended for machine learning",
#         url="https://github.com/WillKoehrsen/feature_selector",
#         author="Will Koehrsen",
#         author_email="wjk@case.edu",
#         license="N/A",
#         packages=["feature_selector"],
#         install_requires=[
#             "lightgbm>=2.1.1",
#             "matplotlib>=2.1.2",
#             "seaborn>=0.8.1",
#             "numpy>=1.14.5",
#             "pandas>=0.23.1",
#             "scikit-learn>=0.19.1"
#             ],
#         zip_safe=False)
        
# run as:
#> python Basic_KNN.py TrainingData 

#=================================
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
    leftoversC0 = class0[(int(partSizeClass0*cvfolds)): ]
    leftoversC1 = class1[(int(partSizeClass1*cvfolds)): ]
    leftovers = pd.concat([leftoversC0,leftoversC1],ignore_index=True, axis=0)
    theFolds = {}
    for i in range(cvfolds):
        fold = 'fold'+str(i)
        class0start = i * partSizeClass0
        class0end = (i * partSizeClass0) + partSizeClass0
        class1start = i * partSizeClass1
        class1end = (i * partSizeClass1) + partSizeClass1
        part1 = class0[int(class0start):int(class0end)]
        part2 = class1[int(class1start):int(class1end)]
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
    #print("Test Data Frame ONLY the Class:")
    ytesting = testDf.iloc[:, int(testDf.shape[1]-1):]
    #print(ytesting.head(3))
   # print(type(ytesting))
   # input("Stop again")
    return xtraining, xtesting, ytraining, ytesting

#=============================    
def Normalization(Data):
    for column in Data:
        minvalue=Data[column].min()
        maxvalue=Data[column].max()
        Data[column]=Data[column].apply(lambda x: (x - minvalue)/(maxvalue-minvalue))
    return Data
#================================================
### FeatureSelection function copied from https://bit.ly/2J4WkIw
def FeatureSelection(trainingdata):
    
    
    ### Find correlation features ##
    corr_matrix = trainingdata.corr().abs()
    
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    print ("uper")
    print (upper)
    # Find index of feature columns with correlation greater than 0.65
    to_drop = [column for column in upper.columns if any(upper[column] > 0.65)]
    print ( "After Selection")
    print (to_drop)
    #trainingdata.drop(trainingdata.columns[to_drop], axis=1)
    trainingdata=trainingdata.drop(columns=to_drop)
   
    print (trainingdata.shape)
    input("stop1")
    #print ( corr_matrix)
    #input("stop2")
    #================================drop low variance features===================
    #sel = VarianceThreshold(threshold=0.00009)
    sel = VarianceThreshold(threshold=0.9)
    SelectedFeatures=sel.fit_transform(trainingdata)
    print (SelectedFeatures.shape)
    input("after variance")
    trainingdata=pd.DataFrame(SelectedFeatures)
    print (trainingdata)
    return trainingdata
#=========================================================

def SplitData2(trainingdata,Kfold, iterationNum):
    #trainingdata=trainingdata.sample(frac=1)  # shuffel
    
    numObservation = trainingdata.shape[0]
    #print (numObservation)
    elemnetPerBlock=int (numObservation/Kfold)
    print (elemnetPerBlock)
    #for i in range(Kfold):
    
    TestStart=iterationNum*elemnetPerBlock
    Testend = TestStart+(elemnetPerBlock)
    print (TestStart , Testend)
    testpart= trainingdata[TestStart:Testend]
    
    #before testing Part 
    TrainStart=0
    Trainend=TestStart
    TrainingPart = trainingdata[TrainStart:Trainend]
    #the remaining Blocks
    TrainStart=Testend
    Trainend=numObservation
    TrainingPart= pd.concat([TrainingPart,trainingdata[TrainStart:Trainend]])
    #print ( "testing part")
    #print (testpart)
    # print ( "trainingpart")
    # print (TrainingPart)
    # print ( "number of total col" , trainingdata.shape[1])
    X_train = TrainingPart.ix[:, 0:(trainingdata.shape[1]-1)]
    y_train = TrainingPart.iloc[:,-1]
    
    X_test = testpart.ix[:, 0:(trainingdata.shape[1]-1)]
    y_test = testpart.iloc[:,-1]
    
    return  X_train, X_test, y_train, y_test
#===========Cross validation=============  
def CrossValidation(trainingdata, Kfold): 

    
    #trainingdata=Normalization(trainingdata)
    trainingdata=FeatureSelection(trainingdata)
    NumofselectedFeatures= trainingdata.shape[1]-1
    print(NumofselectedFeatures)
    input("FF")
    #NumofselectedFeatures= 19
    ResultGrid=pd.DataFrame(pd.np.empty((7, NumofselectedFeatures)) *0) #every row represent differnt K vs differnt numbers of features
    
    print (trainingdata.shape[1]-1)
    print (trainingdata)
   # input("Line 239 here")
    #NumofObservations=len(trainingdata)  
    #n_neighbors=list(np.arange(1, NumofObservations, 2)) # all possible K for KNN
    n_neighbors=list(np.arange(3, 17, 2)) # 7 possible K for KNN
    #AvgAccuracylist=list() # all accurcny forall  possible K for KNN
    
    
    foldDict = FoldSplitter(Kfold)
    print(foldDict.keys())
    for iteration in range(Kfold):
        print(iteration)
        X_train, X_test, y_train, y_test =SplitData(foldDict,iteration)
        #X_train, X_test, y_train, y_test = SplitData(trainingdata,Kfold, iteration)#train_test_split(trainingdata, trainingdata.iloc[:,-1], test_size=0.4, random_state=0)
       
        for P in range( NumofselectedFeatures):
            NewX_train=X_train.iloc[:,0:P+1]   #small Number of featues in TrainingSet
            NewX_test=X_test.iloc[:,0:P+1]  #small Number of featues in testSet
            #print (NewX_train, NewX_test, y_train, y_test)
            
            for K in n_neighbors:
                knn = KNeighborsClassifier(n_neighbors=K)
                knn.fit(NewX_train, y_train.values.ravel())
                PredictedOutput = knn.predict(NewX_test)
                print (accuracy_score(y_test, PredictedOutput))
                probs = knn.predict_proba(NewX_test)
                #use probabilities for class=1
                probs = probs[:, 1]
                #calculate Roc_AUC
                aucValue = roc_auc_score(y_test, probs)
               
                #print(classification_report(y_test, PredictedOutput))
            
            #==========loss function=========
                # i = 0
                # NumofLoss=0
                # while  i <len(PredictedOutput):
                #     
                #     if PredictedOutput[i] != y_test.iloc[i,0]:
                #         NumofLoss +=1
                #     i+=1
                #     
                # if ( NumofLoss<leastLoss):
                #     leastLoss =NumofLoss
                #     bestK= n_neighbors.index(K);
                #     bestnumFeature=P;
                
           
                row_index = n_neighbors.index(K);
                col_index=P;
                ResultGrid.iloc[row_index,col_index]=(ResultGrid.iloc[row_index,col_index]+aucValue)/2

        #return n_neighbors,AvgAccuracylist 
    print ( ResultGrid) 
    
    return n_neighbors,ResultGrid,X_train, X_test, y_train, y_test
def PlotRoc_AucCurve(SelectedNeighbors,ResultGrid,X_train, X_test, y_train, y_test):
    
    #Find top 3 AUC values and their index
    top_n = 2
    topbestmodels=[[-1,-1,-1], [-1,-1,-1], [-1,-1,-1]]
    for k in range(3):
        for i, row in ResultGrid.iterrows():   
            top = row.nlargest(top_n).index
            for topCol in top:
                if ( ResultGrid.loc[i, topCol] >topbestmodels[k][0]):            
                    topbestmodels[k][0]=ResultGrid.loc[i, topCol]
                    topbestmodels[k][1]=i
                    topbestmodels[k][2]=topCol            
            ResultGrid.loc[topbestmodels[k][1], topbestmodels[k][2]]=-1 

    print (topbestmodels)
    print(ResultGrid)
    input("stopResultGrid")
    for k in range(3):
        print( "inside Roc Curve")
        print(X_test)
        NewX_train=X_train.iloc[:,0:topbestmodels[k][2]+1]   #small Number of featues in TrainingSet
        NewX_test=X_test.iloc[:,0:topbestmodels[k][2]+1]  #small Number of featues in testSet
        #knn = KNeighborsClassifier(n_neighbors=bestK)
        knn = KNeighborsClassifier(n_neighbors=SelectedNeighbors[topbestmodels[k][1]])
        knn.fit(NewX_train, y_train)
        
        
        probs = knn.predict_proba(NewX_test)
        #use probabilities for class=1
        probs = probs[:, 1]
        #calculate Roc_AUC
        aucValue = roc_auc_score(y_test, probs)
        print('AUC = %.3f' % aucValue)
        # calculate roc curve
        fpr, tpr, thresholds = roc_curve(y_test, probs)
        # plot no skill
        # plot the roc curve for the model
        selectedK=SelectedNeighbors[topbestmodels[k][1]]
        SelectFeatures=topbestmodels[k][2]
        LableTxt= "K = " + str(selectedK)+  " ,#P = " +str(SelectFeatures)+ " ,AUC= "+str(round(topbestmodels[k][0],2))
        pyplot.plot(fpr, tpr, marker='.',label=LableTxt)
        # show the plot
    
    pyplot.legend(loc='upper left')
       
    pyplot.title("Combined Statistics")    
    pyplot.xlabel("Length of list (number)")
    pyplot.ylabel("Time taken (seconds)")    
    pyplot.plot([0, 1], [0, 1], linestyle='--')
    pyplot.show()
        
    input("aftercurve")
    
   ######################################################## 
    
    
     # predict class values
    PredictedOutput = knn.predict(NewX_test)
    # calculate precision-recall curve
    print(y_test.shape)
    print(probs.shape)
    input("plot")
    print(probs)
    precision, recall, thresholds = precision_recall_curve(y_test, probs)
    #probs = probs[:, 1]

    
    # calculate precision-recall curve
    precision, recall, thresholds = precision_recall_curve(y_test, probs)
    # calculate F1 score
    f1 = f1_score(y_test, PredictedOutput)
    # calculate precision-recall AUC
    auc = auc(recall, precision)
    # calculate average precision score
    ap = average_precision_score(y_test, probs)
    print('f1=%.3f auc=%.3f ap=%.3f' % (f1, auc, ap))
    # plot no skill
    pyplot.plot([0, 1], [0.5, 0.5], linestyle='--')
    # plot the roc curve for the model
    pyplot.plot(recall, precision, marker='.')
    # show the plot
    pyplot.show()
    return
    
    
def PlotKVsError ():
    print ( "accuracy list" , AvgAccuracylist)
    print ( "list k = " ,   n_neighbors[:len(AvgAccuracylist)] )

    print ( "max accuracy " , max(AvgAccuracylist))
    print ( "best k = " ,   n_neighbors [AvgAccuracylist.index(max(AvgAccuracylist))])

    pyplot.plot(n_neighbors[:len(AvgAccuracylist)], AvgAccuracylist)
    pyplot.xlabel('n_neighbors')
    pyplot.ylabel('AvgAccuracylist')
    pyplot.show()
    return 
 #================main=================
 
#help(precision_recall_curve(x,y))

#print (sys.argv)

if len(sys.argv) <= 1:
    path="/home/mun/Dropbox/MachineLearning/MachineLearning_Shared/Assignments/Assignment 2/Task2/A2_t2_dataset.tsv"
else:
    path=sys.argv[1]

    
trainingdata = pd.read_csv( path, sep='\t', header=0) 


#print("### Running KNN.py - Authors: Dalia and Carlos ###")
Kfold =5
n_neighbors,ResultGrid,X_train, X_test, y_train, y_test = CrossValidation(trainingdata,Kfold )
PlotRoc_AucCurve(n_neighbors,ResultGrid,X_train, X_test, y_train, y_test)

print ( "results")
print (ResultGrid.min().min())


PlotKVsError (n_neighbors)

