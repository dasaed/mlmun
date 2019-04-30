# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import math
import sys
import random
import time

# run as:
#> python Basic_KNN.py TrainingData TestData K

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
def ClassicalKNN(trainingdata,testingdata,K,typeofDistance):
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
        print(BotKPerTest)
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
    FinalOutput.to_csv(OutPutDate+'output.csv',sep='\t')

    return "Check output file"
 
 #================main=================
print (sys.argv)

trainingdata = pd.read_csv( sys.argv[1], sep='\t', header=0) 
testingdata=  pd.read_csv( sys.argv[2], sep='\t', header=0)
K=int(sys.argv[3])
print (trainingdata)
print("### Running Classical KNN ###")
ClassicalKNN(trainingdata,testingdata,K,1)

