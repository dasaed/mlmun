# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import numpy as np
import math
import sys
import random
import time


#=========================================================
def Normalization(Data):
    for column in Data:
        minvalue=Data[column].min()
        maxvalue=Data[column].max()
        Data[column]=Data[column].apply(lambda x: (x - minvalue)/(maxvalue-minvalue))
    return Data


 #===========Cross validation=============  
def CrossValidation(trainingdata, K,DistanceType,Nearest_neighbour):  #Dalia =1 Carols=2
    Kfold=3
    Accuracy=-1
    for iteration in range(Kfold):
        X_train, X_test, y_train, y_test = train_test_split(trainingdata.loc[:, trainingdata.columns != 'Class'], trainingdata['Class'], test_size=0.4, random_state=0)
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
 #===========Distance Funcs============= 
def EculideanDistance(inputInstance,testInstance):
    Summation =0
    #for column in testInstance:
    for i in range(8):

        d=inputInstance[1][i] - testInstance[1][i]
        Summation= Summation+ math.pow(d ,2) 
    Distance = [math.sqrt(Summation),inputInstance[1][9]]

    return Distance
 

#========= Calc Nearest Neighbor option 2 ====
def Calculate_Nearest_neighbourRandom(topMatched,K,DistanceType):
    #print("########## CALCULATE NEAREST NEIGHBOR ########")
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
        #print("Number: "+str(myNum)  + "   Iteration: "+str(count))
    topMatched.reset_index(drop=True, inplace=True)
    tieBreakers.reset_index(drop=True, inplace=True)
    topMatches = pd.concat([topMatched, tieBreakers], axis=1, names=['Distance', 'Class', 'Count', 'tieBreaker'])
    #print(topMatches)
    topCount = topMatches['Count'].max()
    Probability = float(topCount) / float(K)
    topMatch = topMatches[ (topMatches['Count'] == topCount ) ]
    #print(topMatch)
    #test = list(topMatch.columns.values)
    #print(test)
    topDecision = topMatch.sort_values(by=['tieBreaker'],ascending=True).head(1)
    #print(topDecision)
    FinalOutput = [topDecision.iloc[0,1],Probability]
    #print(FinalOutput)
    #input("Stop!!!!!!!")
    return FinalOutput
# =================================================
def Calculate_Nearest_neighbour(topMatched,K,DistanceType):
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

def ClassicalKNN(trainingdata,testingdata,K,DistanceType,Nearest_neighbour):
    FinalOutput=pd.DataFrame(columns=['Class','conditional_prob']);
    for row_inTest in testingdata.iterrows():
        AllDistancePerTest=pd.DataFrame(columns=['Distance','Class'])
       
        for row_inTrain in trainingdata.iterrows():
            #print ( "inside loop\n")
            #print ( row_inTrain ) 
            #print ( "row_inTest\n")
            #print (  row_inTest)
            
            resultlist=EculideanDistance(row_inTrain, row_inTest)
            # print ( "resultlist\n")
            # print ( resultlist)
            AllDistancePerTest.loc[len(AllDistancePerTest), :]=resultlist
     
        # d sort And get top k 5
        #print ( "AllDistancePerTest\n")
        #print ( AllDistancePerTest) 
        TopKPerTest=AllDistancePerTest.sort_values(by=['Distance'],ascending=True).head(K)
        #print ("after sort and get top k \n" )
        #print ( AllDistancePerTest)
        #print (TopKPerTest)
        
        #majority vot  or weight 
        if ( Nearest_neighbour ==1): #dalia
            tempOutput = Calculate_Nearest_neighbour(TopKPerTest,K,DistanceType)
        elif ( Nearest_neighbour ==2): #Carlos
            tempOutput = Calculate_Nearest_neighbourRandom(TopKPerTest,K,DistanceType)
        
        FinalOutput.loc[len(FinalOutput), :]=tempOutput
        #print('FinalOutput')
        #print(FinalOutput)
        #break

        #conditional probability
        #output
    print('FinalOutput')
    print(FinalOutput)
    FinalOutput.to_csv('output.csv',sep='\t')
        
    return 23
 
 #================main=================
trainingdata = pd.read_csv( sys.argv[1], sep='\t', header=0) 

testingdata=  pd.read_csv( sys.argv[2], sep='\t', header=0) 
K=int(sys.argv[3])


# Normalize Files ( Training and Testing ) 

# NormalizedTrainingData=Normalization(trainingdata.drop('Class', axis=1))
# print ("The Training Data is Normalized\n")
# NormalizedTrainingData = pd.concat([NormalizedTrainingData, trainingdata['Class']], axis=1)
# NormalizedTestData=Normalization(testingdata); 
# print ("The testing Data is Normalized\n")   
# 
# print ("KNN Running.....\n")   
# #Run KNN   
# ClassicalKNN(NormalizedTrainingData,NormalizedTestData,K,1,1)  #Dalia 1 withoutweighted=1 normaliez




#===================Testing Code##########################



# 
# 
# 
# 
choice = input("""Press 1 Classical KNN ( ecledain distance , neighrest nabour dalia) \n
Press 2 Classical KNN ( ecledain distance , neighrest nabour Carlos)\n
Press 3 Classical KNN ( ecledain distance , neighrest nabour Dalia) And weighted  \n   
Press 4 Classical KNN ( ecledain distance , neighrest nabour Carlos) And weighted  Not work\n 
Press 5 NormalizationMin KNN ( ecledain distance , neighrest nabour Dalia) And weighted\n
  """)
  
print("let us start")
if int(choice) == 1:
    CrossValidation(trainingdata, K,1,1)    # sum , dalia 1 Call 
    
    print ("call knn\n")
elif int(choice) == 2:
    
    #print (trainingdata.drop('Class', axis=1))
    NormalizedTrainingData=Normalization(trainingdata.drop('Class', axis=1))
    NormalizedTrainingData = pd.concat([NormalizedTrainingData, trainingdata['Class']], axis=1)
    NormalizedTestData=Normalization(testingdata); 
    #print ( NormalizedTrainingData)
    #print ( NormalizedTestData)
    
    #CrossValidation(NormalizedTrainingData, K,1,2) #Sum 1 , Carlos 2 Call
    CrossValidation(NormalizedTrainingData, K,1,2) #Sum 1 , Carlos 2 Call  

elif int(choice) == 3:
    CrossValidation(trainingdata, K,2,1)    # weighted , dalia 1

elif int(choice) == 4:
    CrossValidation(trainingdata, K,2,2)    #Calos 1 weighted
    print ("Not weights\n")
elif int(choice) == 5:     # normalize min
    #normalization then knn
    print (trainingdata.drop('Class', axis=1))
    NormalizedTrainingData=Normalization(trainingdata.drop('Class', axis=1))
    NormalizedTrainingData = pd.concat([NormalizedTrainingData, trainingdata['Class']], axis=1)
    NormalizedTestData=Normalization(testingdata); 
    print ( NormalizedTrainingData)
    print ( NormalizedTestData)
    
    CrossValidation(NormalizedTrainingData, K,1,1)    # notweighted =2 , Dalia 1 normalized
elif int(choice) == 6:     # normalize min
    #normalization then knn
    print (trainingdata.drop('Class', axis=1))
    NormalizedTrainingData=Normalization(trainingdata.drop('Class', axis=1))
    NormalizedTrainingData = pd.concat([NormalizedTrainingData, trainingdata['Class']], axis=1)
    NormalizedTestData=Normalization(testingdata); 
    print ( NormalizedTrainingData)
    print ( NormalizedTestData)
    
    CrossValidation(NormalizedTrainingData, K,2,1)    # weighted =2 , Dalia 1 normalized
elif int(choice) == 7:
#     #call knn && weight knn66
    #print (trainingdata.drop('Class', axis=1))
    NormalizedTrainingData=Normalization(trainingdata.drop('Class', axis=1))
    NormalizedTrainingData = pd.concat([NormalizedTrainingData, trainingdata['Class']], axis=1)
    NormalizedTestData=Normalization(testingdata); 
    #print ( NormalizedTrainingData)
    #print ( NormalizedTestData)
    ClassicalKNN(NormalizedTrainingData,NormalizedTestData,K,2,1)  #Dalia 1 weighted=2 normaliez
    #ClassicalKNN(NormalizedTrainingData,NormalizedTestData,K,1,1)  #Dalia 1 withoutweighted=1 normaliez
elif int(choice) == 8:
    NormalizedTrainingData=Normalization(trainingdata.drop('Class', axis=1))
    NormalizedTrainingData = pd.concat([NormalizedTrainingData, trainingdata['Class']], axis=1)
    NormalizedTestData=Normalization(testingdata); 
    #print ( NormalizedTrainingData)
    #print ( NormalizedTestData)
else:
    print ("Wrong choice!!")
  
#======================================================        


