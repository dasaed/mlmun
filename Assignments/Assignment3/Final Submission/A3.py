# -*- coding: utf-8 -*-

# Authors: Dalia and Carlos

from matplotlib import pyplot
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from pprint import pprint
from sklearn import svm
from sklearn import metrics
import pandas as pd
import numpy as np
import math
import sys
import os
import random
import time
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

font = {'family': 'serif',
        'color':  'darkred',
        'weight': 'normal',
        'size': 16,
        }

def Normalization(Data):
    for column in Data:
        minvalue=Data[column].min()
        maxvalue=Data[column].max()
        Data[column]=Data[column].apply(lambda x: (x - minvalue)/(maxvalue-minvalue))
    return Data

#=============Feature selection using Random Forest=============================================
def selectKImportance(model, X, k=5):
     return X[:,model.feature_importances_.argsort()[::-1][:k]]
     

#=================================================
def FeatureSelection_RandomForest (trainingdata,testingdata,NumOfImportantFeatures): 
    
    print ( "Feature Ordering..")
    RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
        max_depth=None, max_features='auto', max_leaf_nodes=None,
        min_samples_leaf=1, min_samples_split=2,
        min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1,
        oob_score=False, random_state=None, verbose=0,
        warm_start=False) 
    model = RandomForestClassifier()
    model.fit(trainingdata.iloc[:,:trainingdata.shape[1]-1],trainingdata.iloc[:,-1].values.ravel())
  
    importances = model.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]
   
    #print ( indices)
    #print ( indices.shape)
    #print (trainingdata.shape)
    #input ("DSFDSF")
    trainingdata =trainingdata.loc[:,indices]
    testingdata  = testingdata.loc[:,indices]

    return trainingdata,testingdata      
 
####################################Our cross validation ###############################################
def FoldSplitter(kfolds,trainingdata):
    #print("### Splitting Training Data  ###")
    mydata = trainingdata
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
        
    return (theFolds)

#############################################################################
def SplitData(dictOfFolds, iterNum):
    testfold = 'fold'+str(iterNum)
    testPart = dictOfFolds.get(testfold,'Something is wrong, the iterNum does not correspond to a valid fold')
    testDf = pd.DataFrame()
    testDf = testDf.append(testPart) #Create my Test DataFrame fold
    #dictOfFolds.pop(testfold) # Delete Test Dataframe from total folds
    trainkeys = dictOfFolds.keys()
    #print(trainkeys)
    trainDf = pd.DataFrame() #create Training/Learing Dataframe
    #print("#######################################################################################################")
    #print("Entering Learning Fold Concatenation: ")
    for foldKey, foldValue in dictOfFolds.items():
        if foldKey == testfold:
            continue
        newdf = pd.DataFrame(foldValue)
        trainDf = pd.concat([trainDf,newdf],ignore_index=True)
        #print(foldKey)
    #export_csv = trainDf.to_csv(r'export_dataframe.csv', index=None, header=False)  # Don't forget to add '.csv' at the end of the path
    xlearning = trainDf.iloc[:, :int(trainDf.shape[0]-2)] # all columns except the last one
    ylearning = trainDf.iloc[:, int(trainDf.shape[0]-2):] # last column
    # print("Train Data Frame Full:")
    # print(trainDf.head(3))
    # print("Train Data Frame NO CLASS")
    xtraining = trainDf.iloc[:, :int(trainDf.shape[1]-1)]#all rows, length of column -2
    # print(xtraining.head(3))
    # print("Train Data Frame ONLY the class")
    ytraining = trainDf.iloc[:, int(trainDf.shape[1]-1):]
    # print(ytraining.head(3))
    # print("Test Data Frame Full:")
    # print(testDf.head(3))
    # print("Test Data Frame NO CLASS")
    xtesting = testDf.iloc[:, :int(testDf.shape[1]-1)]#all rows, length of column -2
    # print(xtesting.head(3))
    #print("Test Data Frame ONLY the Class:")
    ytesting = testDf.iloc[:, int(testDf.shape[1]-1):]
    #print(ytesting.head(3))
   # print(type(ytesting))
   # input("Stop again")
    return xtraining, xtesting, ytraining, ytesting
###################################################################################

def SVMLocalCV  (trainingdata,testingdata ):
    
    #trainingdata=FeatureSelection(trainingdata)
  
    MiscalssificationsList=[1,3,4,5,6,7,10]
    foldDict = FoldSplitter(5, trainingdata)
    #print ("\t 2-Spliting Data to 5 Folds Done.")
    #print(foldDict.keys())
    AreaUnderCuvr=[0,0,0,0,0]
    bestC=-1
    bestap=-1
    bestYtest=-1
    for iteration in range(5):
        #print(iteration)
        X_train, X_test, y_train, y_test =SplitData(foldDict,iteration)
        for CItem in MiscalssificationsList:
            #print(CItem)
            clf = svm.SVC(probability=True,kernel='linear', C=CItem)
            # print(X_train)
            # print(y_train)
            clf.fit(X_train,y_train.values.ravel())
            predictions = clf.predict(X_test)
            probs = clf.predict_proba(X_test)
            probs = probs[:, 1]
            ap = average_precision_score(y_test, probs) 
            # print("average_precision_score=== ")
            # print (ap )
            AreaUnderCuvr=AreaUnderCuvr+ap
            if ( bestap <ap):
                bestap=ap
                bestC= CItem
                bestYtest = y_test
                bestpredictions =predictions
       
        precision, recall, thresholds = precision_recall_curve(bestYtest, bestpredictions)
        labeltxt= "SVM, AUCPR= " + str(bestap)+ ", C=" + str(bestC)
        pyplot.plot(recall, precision, marker='o',label=labeltxt)

        pyplot.legend(loc='upper left')
        
        pyplot.title("Precision-Recall")    
        pyplot.xlabel("Recall")
        pyplot.ylabel("Precision") 
    avg=AreaUnderCuvr/35 
    #print (avg) 
    return  bestC, avg     
    
##############################################
def RandomForestLocalCV  (trainingdata,testingdata ):
  
   
    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
        # Number of features to consider at every split
    max_features = ['auto', 'log2']    # NONE like bagging , 'None's
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True]
    
    # Create the random grid
    random_grid = {'n_estimators': n_estimators,
                'max_features': max_features,
                'max_depth': max_depth,
                'min_samples_split': min_samples_split,
                'min_samples_leaf': min_samples_leaf,
                'bootstrap': bootstrap}
        
    #pprint(random_grid)
    
    ###############
    foldDict = FoldSplitter(5, trainingdata)
    print ("Run Cross Validation For Random Forest.")
    #print(foldDict.keys())
    avgAreaUnderCuvr=0
    tempBestModel = RandomForestClassifier()
    bestAp=0
    avgPrecision=0
    avgRecall=0
    Avgaccuracy=0
    arrayOfAUC=[]
    start = time.time()
    for iteration in range(5):
        #print(iteration)
        X_train, X_test, y_train, y_test =SplitData(foldDict,iteration)

        ########### calculate forest grid
        rf =RandomForestClassifier()
        # Random search of parameters, using 5 fold cross validation, 
        # search across 10 different combinations, and use all available cores
        rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 15, cv = 2, verbose=2, random_state=42, n_jobs = -1)
       
        rf_random.fit(X_train,y_train.values.ravel())
        # print(y_train)
        predictions = rf_random.predict(X_test)
        probs = rf_random.predict_proba(X_test)
        #print ( probs)
        probs = probs[:, 1]
        ap = average_precision_score(y_test, probs)
        arrayOfAUC.append(ap)
        #pprint(rf_random.best_params_)
        #print ( ap)
        
        if ( bestAp < ap):
            tempBestModel = rf_random
            bestAp = ap
            bestprob=probs
        avgAreaUnderCuvr= avgAreaUnderCuvr+ ap   
    
        #print (AreaUnderCuvr) 
        precision, recall, thresholds = precision_recall_curve(y_test, predictions)
        Avgaccuracy = Avgaccuracy+accuracy_score(y_test, predictions)
        avgPrecision = avgPrecision+precision
        avgRecall = avgRecall+recall    
    
        labeltxt="RandomForest,  AUCPR = " + str(round(ap,2))
        pyplot.plot(recall, precision, marker='o',label=labeltxt)     
        pyplot.legend(loc='upper left')
        pyplot.title("Precision-Recall")    
        pyplot.xlabel("Recall")
        pyplot.ylabel("Precision")
        
    end = time.time()
    print  (" Random Forest Execution time !!!")  
    print(end - start)
    
      
    Avgaccuracy = Avgaccuracy/5 
    # print ( Avgaccuracy)
    # input( "Avgaccuracy")           
    avgRecall= avgRecall/5
    avgPrecision=avgPrecision/5
    avgAUC = avgAreaUnderCuvr/5
    
    
    std=np.std(arrayOfAUC)
    mean= np.mean(arrayOfAUC)
    
    stdtxt= "std = " + str(round(std,2))
    Meantxt= "Mean = " + str(round(mean,2))
    pyplot.text(0.01, 0.79, stdtxt, fontdict=font)
    pyplot.text(0.01, 0.73, Meantxt, fontdict=font)

    pyplot.plot(avgRecall, avgPrecision, marker='s',label='Average, AVG-AUCPR = '+str(round(avgAUC,2)),linewidth=3.0) 
        
    pyplot.legend(loc='upper left')
    pyplot.title("Precision-Recall")    
    pyplot.xlabel("Recall")
    pyplot.ylabel("Precision")
    print( "#########################")
   
        
    
    return tempBestModel ,  avgAUC
######################Logisstic regrishion  #################################
def LogissticRegressionLocalCV  (trainingdata,testingdata ):
    print ("Run Cross Validation For Logisstic Regression.")
    foldDict = FoldSplitter(5, trainingdata)
    avgPrecision=0
    avgRecall=0
    avgAreaUnderCuvr=0
    tempBestprob = -1
    bestAp=0
    arrayOfAUC=[]
    start = time.time()
    for iteration in range(5):
        print(iteration)
        X_train, X_test, y_train, y_test =SplitData(foldDict,iteration)

        ########### calculate forest grid
        log_class = LogisticRegression()
        log_class.fit(X_train,y_train.values.ravel())
        predictions = log_class.predict(X_test)
        probs=log_class.predict_proba(X_test)
        probs = probs[:, 1]
        ap = average_precision_score(y_test, probs)
        arrayOfAUC.append(ap)
        # print("average_precision_score:")
        # print (ap) 
        if ( bestAp <ap):
            tempBestprob = probs
        avgAreaUnderCuvr= avgAreaUnderCuvr+ ap   
        
        #print (AreaUnderCuvr) 
        precision, recall, thresholds = precision_recall_curve(y_test, predictions)
        #print(precision)
        #print(recall)
        #input("Friendly pause")
        labeltxt="LogissticRegression  AUCPR = " + str(round(ap,2))
        pyplot.plot(recall, precision, marker='o',label=labeltxt)     
        pyplot.legend(loc='upper left')
        pyplot.title("Precision-Recall")    
        pyplot.xlabel("Recall")
        pyplot.ylabel("Precision")
        avgPrecision = avgPrecision+precision
        avgRecall = avgRecall+recall    
    
    
    end = time.time()
    print ("time LOG!!!")
    print(end - start)
    
    avgRecall= avgRecall/5
    avgPrecision=avgPrecision/5
    print(avgRecall)
    print(avgPrecision)
    std=np.std(arrayOfAUC)
    mean= np.mean(arrayOfAUC)
    
    stdtxt= "std = " + str(round(std,2))
    Meantxt= "Mean = " + str(round(mean,2))
    pyplot.text(0.01, 0.79, stdtxt, fontdict=font)
    pyplot.text(0.01, 0.73, Meantxt, fontdict=font)
    
    avgAUC = avgAreaUnderCuvr/5
    pyplot.plot(avgRecall, avgPrecision, marker='s',label='Average , AVG-AUCPR = '+str(round(avgAUC,2)),linewidth=1.0)     
    pyplot.legend(loc='upper left')
    pyplot.title("Precision-Recall")    
    pyplot.xlabel("Recall")
    pyplot.ylabel("Precision")
    
    print( "#########################")    
    
    return  avgAUC
 #######################################################################
def KNNLocalCV(trainingdata,testingdata): 
    print ("Run Cross Validation For KNN.")
    Kfold=5
    NumofselectedFeatures= trainingdata.shape[1]-1
    #print(NumofselectedFeatures)
    
    ResultGrid=pd.DataFrame(pd.np.empty((14, NumofselectedFeatures)) *0) #every row represent differnt K vs differnt numbers of features
  
    n_neighbors=list(np.arange(3, 30, 2)) # 14 possible K for KNN
    
    foldDict = FoldSplitter(5, trainingdata)
    arrayOfAUC=[]
   
    #print(foldDict.keys())
    start = time.time()
    for iteration in range(Kfold):
        #print(iteration)
        X_train, X_test, y_train, y_test =SplitData(foldDict,iteration)
        
        for P in range( NumofselectedFeatures):
            NewX_train=X_train.iloc[:,0:P+1]   #small Number of featues in TrainingSet
            NewX_test=X_test.iloc[:,0:P+1]  #small Number of featues in testSet
            
            for K in n_neighbors:
                knn = KNeighborsClassifier(n_neighbors=K)
                knn.fit(NewX_train, y_train.values.ravel())
                PredictedOutput = knn.predict(NewX_test)
                #print (accuracy_score(y_test, PredictedOutput))
                probs = knn.predict_proba(NewX_test)
                #use probabilities for class=1
                probs = probs[:, 1]
                
                aucValue = average_precision_score(y_test, probs)
                
                #print(classification_report(y_test, PredictedOutput))
            
                row_index = n_neighbors.index(K);
                col_index=P;
                ResultGrid.iloc[row_index,col_index]=ResultGrid.iloc[row_index,col_index]+aucValue
    
        #return n_neighbors,AvgAccuracylist 
    end = time.time()
    print ("timeKNN!!!")  
    print(end - start)
     
    # calculating average AUC
    ResultGrid=ResultGrid/ Kfold
   
    
        #Find top 3 AUC values and their index
    top_n = 2
    topbestmodels=[[-1,-1,-1], [-1,-1,-1], [-1,-1,-1],[-1,-1,-1], [-1,-1,-1]]
    for k in range(5):
        for i, row in ResultGrid.iterrows():   
            top = row.nlargest(top_n).index
            for topCol in top:
                if ( ResultGrid.loc[i, topCol] >topbestmodels[k][0]):            
                    topbestmodels[k][0]=ResultGrid.loc[i, topCol]
                    topbestmodels[k][1]=i
                    topbestmodels[k][2]=topCol            
        ResultGrid.loc[topbestmodels[k][1], topbestmodels[k][2]]=np.NAN 

    avgPrecision=0
    avgRecall=0
    avgAP=0
   
    for k in range(5):
    
        NewX_train=X_train.iloc[:,0:topbestmodels[k][2]+1]   #small Number of featues in TrainingSet
        NewX_test=X_test.iloc[:,0:topbestmodels[k][2]+1]  #small Number of featues in testSet
        knn = KNeighborsClassifier(n_neighbors=n_neighbors[topbestmodels[k][1]])
        knn.fit(NewX_train, y_train.values.ravel())
    
        probs = knn.predict_proba(NewX_test)
        #use probabilities for class=1
        probs = probs[:, 1]
        PredictedOutput = knn.predict(NewX_test)
        ap = average_precision_score(y_test, probs) 
        arrayOfAUC.append(ap)
        avgAP=avgAP+ap;
        precision, recall, thresholds = precision_recall_curve(y_test, PredictedOutput)
        avgPrecision = avgPrecision+precision
        avgRecall = avgRecall+recall    
        
        selectedK=n_neighbors[topbestmodels[k][1]]
        SelectFeatures=topbestmodels[k][2]+1
        labeltxt= "KNN AUCPR = " +str(round(ap,2))+"K = " + str(selectedK)+  " ,#P = " +str(SelectFeatures)
        pyplot.plot(recall, precision, marker='o',label=labeltxt)     
        pyplot.legend(loc='upper left')
        pyplot.title("Precision-Recall")    
        pyplot.xlabel("Recall")
        pyplot.ylabel("Precision")    
   
    avgAP=avgAP/5
    avgPrecision=avgPrecision/5
    avgRecall=avgRecall/5
    
    std=np.std(arrayOfAUC)
    mean= np.mean(arrayOfAUC)
    
    stdtxt= "std = " + str(round(std,2))
    Meantxt= "Mean = " + str(round(mean,2))
    pyplot.text(0.01, 0.79, stdtxt, fontdict=font)
    pyplot.text(0.01, 0.73, Meantxt, fontdict=font)
    
    pyplot.plot(avgRecall, avgPrecision, marker='s',label='Average, AVG-AURPR= '+ str(round(avgAP,2)),linewidth=3.0)     
    pyplot.legend(loc='upper left')
    pyplot.title("Precision-Recall")    
    pyplot.xlabel("Recall")
    pyplot.ylabel("Precision")
    
    print( "#########################")    
    return n_neighbors,topbestmodels
                    
#================main=================
 


if len(sys.argv) < 2:
    trainingpath='/home/dalia/Dropbox/MachineLearning/MachineLearning_Shared/Assignments/Assignment3/A3_training_dataset.tsv'
    testingpath = '/home/dalia/Dropbox/MachineLearning/MachineLearning_Shared/Assignments/Assignment3/A3_test_dataset.tsv'
else:    
   trainingpath=sys.argv[1]
   testingpath=sys.argv[2]


trainingdata = pd.read_csv( trainingpath, sep='\t', header=None) 
testingdata=  pd.read_csv( testingpath, sep='\t', header=None)


NumOfImportantFeatures=2   
trainingdata, testingdata= FeatureSelection_RandomForest(trainingdata,testingdata, NumOfImportantFeatures)
   
# print ( trainingdata.shape[1])
# print ( testingdata.shape[1])
# print ( testingdata.head(10))


# For any classifier  
# 1- Do feature Selection step 
# 2- Tune parameters using Cross validation 
# 3- select best model 
# 4- run the best model and store the data to compare the results.
#####################################################3


# * Random forest   for non linear decision boundery
 
RF_BestModel,AUCPR_RF=RandomForestLocalCV  (trainingdata,testingdata )
pyplot.show()
# 2- Logisstic Regression for  linear decision boundery
AUCPR_LOGR=LogissticRegressionLocalCV (trainingdata,testingdata )
pyplot.show()
#  
# # 3- SVM 
#BestC,AUCPR_SVM =SVMLocalCV  (trainingdata,testingdata )
# 
# #==============
SelectedNeighbors,topbestmodels = KNNLocalCV(trainingdata,testingdata)
# 
AUCPR_KNN=topbestmodels[0][0]
# #==============
pyplot.show()
# 
# # RandomForest =1  
# # Logisstic =2 
# # KNN = 3
# 
# 
SelectedModel =0
if ( AUCPR_RF > AUCPR_KNN):
    if ( AUCPR_RF>AUCPR_LOGR):
        #Store RF
        SelectedModel=1
    else:
        #store logR
        SelectedModel=2
else:
      if ( AUCPR_KNN >AUCPR_LOGR):
      #store knn
        SelectedModel=3  
      else:
          #stor log 
          SelectedModel=2

print ("SelectedModel")
print ( SelectedModel)
# #======================RUN THE BEST MODELS================================
foldDict = FoldSplitter(5, trainingdata)
X_train, X_test, y_train, y_test =SplitData(foldDict,0)

# ======================RF==============================
RF_BestModel.fit(X_train,y_train.values.ravel())

predictions = RF_BestModel.predict(X_test)
probs = RF_BestModel.predict_proba(X_test)
probs = probs[:, 1]
ap = average_precision_score(y_test, probs)
print( "the best model for Random Forest is ")
pprint(RF_BestModel.best_params_)
  
precision, recall, thresholds = precision_recall_curve(y_test, predictions)
labeltxt="RandomForest,  AUCPR = " + str(round(ap,2))
pyplot.plot(recall, precision, marker='o',label=labeltxt)   
# 
# ======================LogisticRegression==============================
log_class = LogisticRegression()
log_class.fit(X_train,y_train.values.ravel())
predictions = log_class.predict(X_test)
probs=log_class.predict_proba(X_test)
probs = probs[:, 1]
ap = average_precision_score(y_test, probs) 
precision, recall, thresholds = precision_recall_curve(y_test, predictions)
#
labeltxt="LogissticRegression  AUCPR = " + str(round(ap,2))
pyplot.plot(recall, precision, marker='o',label=labeltxt)     

# 
# ======================KNN==============================
#     
NewX_train=X_train.iloc[:,0:topbestmodels[0][2]+1]   #small Number of featues in TrainingSet
NewX_test=X_test.iloc[:,0:topbestmodels[0][2]+1]  #small Number of featues in testSet
knn = KNeighborsClassifier(n_neighbors=SelectedNeighbors[topbestmodels[0][1]])
knn.fit(NewX_train, y_train.values.ravel())

probs = knn.predict_proba(NewX_test)
#use probabilities for class=1
probs = probs[:, 1]
PredictedOutput = knn.predict(NewX_test)
ap = average_precision_score(y_test, probs) 
precision, recall, thresholds = precision_recall_curve(y_test, PredictedOutput)

selectedK=SelectedNeighbors[topbestmodels[0][1]]
SelectFeatures=topbestmodels[0][2]+1
labeltxt= "KNN AUCPR = " +str(round(ap,2))+"K = " + str(selectedK)+  " ,#P = " +str(SelectFeatures)
pyplot.plot(recall, precision, marker='o',label=labeltxt) 
####################################### 
pyplot.legend(loc='upper left')
pyplot.title("Precision-Recall")    
pyplot.xlabel("Recall")
pyplot.ylabel("Precision")    
pyplot.show()
#    
# 


print ( "Test using the test File ")
# ##########################################OUTPUT########################################################         
if ( SelectedModel==1):   # RF
    
    RF_BestModel.fit(trainingdata,trainingdata.iloc[:,-1].values.ravel())
    RF_probs = RF_BestModel.predict_proba(testingdata)
    RF_probs = RF_probs[:, 1]
    ToSavepd=pd.DataFrame(RF_probs)
    ToSavepd.to_csv('output.tsv',sep='\t', header=0,index=False)    
    
elif(SelectedModel==2): # log
    log_class = LogisticRegression()
    log_class.fit(trainingdata,trainingdata.iloc[:,-1].values.ravel())
    predictions = log_class.predict(testingdata)
    probs=log_class.predict_proba(testingdata)
    probs = probs[:, 1]
    # print ("count ")
    # print ( np.bincount( predictions)) 
    # input ( "stop")
    ToSavepd=pd.DataFrame(probs)
    ToSavepd.to_csv('output.tsv',sep='\t', header=0,index=False)     
elif(SelectedModel==3):  #knn
    
    trainingdata=trainingdata.iloc[:,0:topbestmodels[0][2]+1]   #small Number of featues in TrainingSet
    testingdata=testingdata.iloc[:,0:topbestmodels[0][2]+1]  #small Number of featues in testSet
    #knn = KNeighborsClassifier(n_neighbors=bestK)
    knn = KNeighborsClassifier(n_neighbors=SelectedNeighbors[topbestmodels[0][1]])
    knn.fit(trainingdata, trainingdata.iloc[:,-1].values.ravel())

    KNN_probs = knn.predict_proba(testingdata)
    #use probabilities for class=1
    KNN_probs = KNN_probs[:, 1]
    ToSavepd=pd.DataFrame(KNN_probs)
    ToSavepd.to_csv('output.tsv',sep='\t', header=0,index=False)   


print ( "The Output File  is generated. ")

#================================================
# print ("count ")
# print ( np.bincount( predictions)) 
# input ( "stop")
