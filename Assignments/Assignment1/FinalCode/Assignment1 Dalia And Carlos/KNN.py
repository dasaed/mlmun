# -*- coding: utf-8 -*-

# Authors: Dalia and Carlos

from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import pandas as pd
import numpy as np
import math
import sys
import random
import time

# run as:
#> python Basic_KNN.py TrainingData TestData K
#=========================================================
#===========Cross validation=============  
def CrossValidation(trainingdata, K,DistanceType,Nearest_neighbour):  #Weighted =1 tiebreaker=2
    Kfold=5
    Accuracy=-1
    for iteration in range(Kfold):
        #X_train, X_test, y_train, y_test = train_test_split(trainingdata.loc[:, trainingdata.columns != 'Class'], trainingdata['Class'], test_size=0.4, random_state=0)
        trainingdata=trainingdata.sample(frac=1)  # shuffel
        X_train, X_test, y_train, y_test = train_test_split(trainingdata, trainingdata['Class'], test_size=0.4, random_state=0)
    
        #ClassicalKNN(X_train,X_test,K,1) correct
        ClassicalKNN(trainingdata,X_test,K,DistanceType,Nearest_neighbour)
        output = pd.read_csv( "output.csv", sep='\t') 
        #print (output)
        PredictedOutput=output['Class']  # this should be compared with y_test
        
        i = 0
        correct=0
        NotCorrect=0
       
    
        print(classification_report(y_test, PredictedOutput))
      
        while  i <len(PredictedOutput):
            
            if PredictedOutput[i] == y_test.iloc[i]:
                correct +=1
            else:
                NotCorrect+=1
        
            i+=1
        
        AccuracyTemp = correct/(correct+NotCorrect)
        Accuracy = (AccuracyTemp +AccuracyTemp)/2
       
        print ( "Accuracy = %f /n",Accuracy)
    return 2
#=============================    
def Normalization(Data):
    for column in Data:
        minvalue=Data[column].min()
        maxvalue=Data[column].max()
        Data[column]=Data[column].apply(lambda x: (x - minvalue)/(maxvalue-minvalue))
    return Data

 #=========== Distance Funcs=============
def EculideanDistance(inputInstance,testInstance):
    Summation =0
    #for column in testInstance:
    for i in range(8):
        d=inputInstance[1][i] - testInstance[1][i]
        Summation= Summation+ math.pow(d ,2)
    Distance = [math.sqrt(Summation),inputInstance[1][9]]
    return Distance

 #========= Calc Nearest Neighbor option 2 ==========#
def Calculate_Nearest_neighbour(topMatched,K):
    #print("######### TopMatched ######### \n"+str(topMatched))
    if ( topMatched.iloc[0, 0] == 0 ):# if distance is 0
        OutputClass=topMatched.iloc[0, 1];
        FinalOutput=[OutputClass, 1.0]
        return FinalOutput
    topMatched['Count'] = topMatched.groupby('Class')['Class'].transform('count')
    tieBreakers = pd.DataFrame(columns=['tieBreaker']);
    count = 0
    while tieBreakers.shape[0] < topMatched.shape[0]:
        myNum = random.randint(0, topMatched.shape[0]*3)
        if myNum not in tieBreakers.values:
            tieBreakers.loc[len(tieBreakers)] = myNum
        count+=1
    topMatched.reset_index(drop=True, inplace=True)
    tieBreakers.reset_index(drop=True, inplace=True)
    topMatches = pd.concat([topMatched, tieBreakers], axis=1, names=['Distance', 'Class', 'Count', 'tieBreaker'])
    topCount = topMatches['Count'].max()
    Probability = float(topCount) / float(K)
    topMatch = topMatches[ (topMatches['Count'] == topCount ) ]
    topDecision = topMatch.sort_values(by=['tieBreaker'],ascending=True).head(1)
    FinalOutput = [topDecision.iloc[0,1],Probability]
    #print(FinalOutput)
    return FinalOutput

 #======================== 
def Calculate_Nearest_neighbourWeighted(topMatched,K,DistanceType):
    #print("topMatcheddddddddd")
    #print (topMatched)
    MaxConditionalProbability=-1
    OutputClass=-1
    
    for j in range(K):  #loop on classes
        sum =0
        allweights=0
        for i in range(K):
            if (DistanceType==1):
                allweights =K
            else:
                  if (topMatched.iloc[i, 0] !=0):
                    allweights =allweights+(1/(topMatched.iloc[i, 0]**2))
                   
            
            if ((topMatched.iloc[j, 1] ==topMatched.iloc[i, 1]) ):# i!=j
                if(DistanceType ==1):
                    sum = sum+1
                elif(DistanceType ==2):
                    if (topMatched.iloc[i, 0] !=0):
                        sum = sum+(1/(topMatched.iloc[i, 0]**2))
                        
                    else:
                        sum=K
        if (allweights ==0):
            allweights=sum;    # all weights ==0 so the probabity should equal 100
        Probability=sum/allweights
       
        if (Probability > MaxConditionalProbability):
            MaxConditionalProbability=Probability
            OutputClass=topMatched.iloc[j, 1];
    FinalOutput=[OutputClass,round(MaxConditionalProbability,2)]
    #print (FinalOutput)
    return FinalOutput
  #======================================================
def ClassicalKNN(trainingdata,testingdata,K,typeofDistance,Nearest_neighbour):
    FinalOutput=pd.DataFrame(columns=['Class','conditional_prob']);
    for row_inTest in testingdata.iterrows():
        AllDistancePerTest=pd.DataFrame(columns=['Distance','Class'])
        for row_inTrain in trainingdata.iterrows():
            resultlist=EculideanDistance(row_inTrain, row_inTest)
                #EuclideanDistance returns [ distance.float , class.string]
            AllDistancePerTest.loc[len(AllDistancePerTest), :]=resultlist
       # print("Row test vs Row Train")
        #print(row_inTest)
        #print(row_inTrain)
        BotKPerTest=AllDistancePerTest.sort_values(by=['Distance'],ascending=True).head(K)
        #print(BotKPerTest)
        #input("Stop")
        #TopKPerTest =
        # [ 'Distance' , 'Class' ]
        # [    1.2          a
        # [    2.1          b
        # [    2.2          b
        #majority vote
        tempOutput = Calculate_Nearest_neighbour(BotKPerTest,K)
        FinalOutput.loc[len(FinalOutput), :]=tempOutput
        #print('FinalOutput')
        #print(FinalOutput)
    OutPutDate = time.strftime("%Y%m%d-%H%M%S")
    print('FinalOutput')
    print(FinalOutput)
    FinalOutput.to_csv('output.csv',sep='\t')

    return "Check output file"
 
 #================main=================
#print (sys.argv)

trainingdata = pd.read_csv( sys.argv[1], sep='\t', header=0) 
testingdata=  pd.read_csv( sys.argv[2], sep='\t', header=0)

if len(sys.argv) <= 3:
#    print(len(sys.argv))
    print("Using K = 3")
    K=3
else:
    K=int(sys.argv[3])

# NormalizedTrainingData=Normalization(trainingdata.drop('Class', axis=1))
# print ("The Training Data is Normalized\n")
# NormalizedTrainingData = pd.concat([NormalizedTrainingData, trainingdata['Class']], axis=1)
# NormalizedTestData=Normalization(testingdata); 
# print ("The testing Data is Normalized\n")   
# 
# print ("****KNN Running**999**\n")   
# #ClassicalKNN(NormalizedTrainingData,NormalizedTestData,K,1)


print("### Running KNN.py - Authors: Dalia and Carlos ###")
ClassicalKNN(trainingdata,testingdata,K,1,1)
print("#########")
#CrossValidation(trainingdata, K,1,1)

