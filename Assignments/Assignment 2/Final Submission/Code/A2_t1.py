# -*- coding: utf-8 -*-

# Authors: Dalia Ibrahim, Carlos Salcedo and Shengkai Geng

import pandas as pd
import numpy as np
import math
import sys
import time

#####################################################################
#import pandas as pd
#import numpy as np
#df = pd.DataFrame({'col1' : ['A', 'A', 'B', np.nan, 'D', 'C'],  'col2' : [2, 1, 9, 8, 7, 4], 'col3': [0, 1, 9, 4, 2, 3], })
#print("The original set")
#print (df)
#df1=df.sort_values(by=['col2'], ascending=False)
#df1=df.sort_values('col2')
#print ( "AAfter sort \n")
#print (df1)
#####################################################################
#####################################################################
#sampledf = pd.DataFrame( {'c1':[50,25,12], 'c2':[20,45,20], 'c3':[15,10,60]} )
#print(sampledf)
# sampledf
#           c1  c2  c3
#   c1      50  20  15
#   c2      25  45  10
#   c3      12  20  60

FinalOutput=pd.DataFrame(columns=['Class','Precision','Recall','Specificity','FDR']);
def metricCalculator(data):
    """
    Data should be in the form of:
        Class   C1  C2      ...     Cn
        C1      #   #       ...     #
        C2      #   #       ...     #
        ...    ...  ...     ...     ...
        Cn      #   #       ...     #
    * Rows are the true Class
    * Columns are the predicted class
    * # = represents int or float, preferably float
    :param data:
    :return:
    # sampledf
    #           c1  c2  c3
    #   c1      50  20  15
    #   c2      25  45  10
    #   c3      12  20  60
    # Precision = TP / (TP + FP)
    # Recall = TP / RP
    # Specificity = TN / RN
    # FDR = False Discovery Rate = FP / P
    # Class         Precision               Recall             Specificity                                              FDR
    # c1           50/(50+20+15)         50/(50+25+12)     1-[((50+20+15)-50) / ((20+45+20)+(15+10+60))]            (50+20+15)-50)/(50+20+15)
    # c1                0.59                  0.57                 0.79                                                  0.41
    # c1 [i]    ([i+1]/sum(row[i]))     [i+1]/col[i+1]     1-[(sum(row(i))-row[i] )/[sum(allCol) - sum(col(i)]      (sum(row(i))-row[i])/sum(col(i))
        #FinalOutput.loc[len(FinalOutput), :]=tempOutput
    """
    FinalOutput=pd.DataFrame(columns=['Class','Precision','Recall','Specificity','FDR']);
    tempdf = data.drop([0],axis=1).drop([0],axis=0)#axis = 1 =  columns, axis = 0 = rows
    tempdf = tempdf.astype(int)
    totalSamples = tempdf.values.sum()
    counter = 0 # Tells me what row I am on
    accuracy = 0
    for classRow in data.iterrows():
        if counter == 0:
            counter+=1
            #row 0 = header or column title
            continue
        #print("#### Row "+str(counter)+" ####")
        #print(classRow)
        #print("Index of the Current Row with respect to the Matrix "+str(classRow[0]))
        theClass = classRow[1][0]
        truePositive = classRow[1][counter]
        accuracy += int(truePositive)
        #tpfpClass = sum(classRow[1]-classRow[1][0])
        sumRow=0
        #print("Size of Row: "+str(data.shape[0]-1))
        #tpfpClass = sum(classRow[1] for i in range(1,classRow[data.shape[0]-1]))
        #print("The class is: "+theClass)
        #print("The TP is: "+truePositive)
        ############ Calculating Presicion ##############
        for i in range(1,data.shape[0]):
            #print(classRow[1][i])
            sumRow=sumRow + int(classRow[1][i])
        #print("TP + FP = "+str(sumRow))
        precision=float(truePositive)/float(sumRow)
        #print("Precision = "+str(precision))
        #################################################

        ############ Calculating Recall ##############
        tempdf = data[counter].drop(0,axis=0)
        tempdf = tempdf.astype(int)
        realPositive = tempdf.values.sum()
        #print(realPositive)
        recall = float(truePositive)/float(realPositive)
        #print("Recall = "+str(recall))
        #################################################
        ############ Calculating Specificity ##############
        tempdf = data.drop([0,counter],axis=1).drop([0],axis=0)#axis = 1 =  columns, axis = 0 = rows
        tempdf = tempdf.astype(int)
        #print(tempdf)
        realNegatives=tempdf.values.sum()
        falsePositives = float(sumRow) - float(truePositive)
        #print("Real Negatives = "+str(realNegatives))
        specificity = 1.0 - (float(falsePositives) / float(realNegatives))
        #print("Specificity = "+str(specificity))
        ###################################################
        ##### Calculating False Discovery Rate (FDR) ######
        fdr = (float(sumRow)-float(truePositive)) / float(sumRow)
        #print("FDR = "+str(fdr))
        #print("*************************************************")
        FinalOutput.loc[len(FinalOutput), :]=[theClass,round(precision,2),round(recall,2),round(specificity,2),round(fdr,2)]
        #OutPutDate = time.strftime("%Y%m%d_%H%M%S")
        #OutPutDate = time.strftime("%Y%m%d_%H%M")
        #FinalOutput.to_csv(OutPutDate+'output.csv',sep='\t')
        counter +=1
    accuracy = round(((float(accuracy)/float(totalSamples)) * 100),2)
    print("Accuracy = "+str(accuracy)+"%")
    print(FinalOutput)
    return

 #================main=================

print("### Performance Metric Calculations - Dalia, Carlos and Shengkai  ###")
mydata = pd.read_csv( sys.argv[1], sep='\t', header=None,)
#mydata = pd.read_csv( sys.argv[1], sep='\t', header=None,)
mydata.loc[0:0,[0,0]] = 'Classes'
metricCalculator(mydata)
