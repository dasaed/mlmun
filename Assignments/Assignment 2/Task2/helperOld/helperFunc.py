import pandas as pd
import numpy as np
import math
import sys
import random
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

Trainingdata = pd.read_csv("/home/mun/Dropbox/MachineLearning/MachineLearning_Shared/Assignments/Assignment 2/Task2/A2_t2_dataset.tsv", sep='\t', header=0)

#print ( Trainingdata.iloc[:,0:1] )
#newdf= pd.DataFrame(pd.np.empty((5, 5)) * 1)
newdf= pd.DataFrame(np.arange(60).reshape(10,6))
print ( newdf)
dp = newdf/2
print ( dp)


# print (newdf.max().max())
# print(newdf.idxmax(axis=1))

# top_n = 2
# topbestmodels=[[-1,-1,-1], [-1,-1,-1], [-1,-1,-1]]
# for k in range(3):
#     for i, row in newdf.iterrows():   
#         top = row.nlargest(top_n).index
#         for topCol in top:
#             if ( newdf.loc[i, topCol] >topbestmodels[k][0]):            
#                 topbestmodels[k][0]=newdf.loc[i, topCol]
#                 topbestmodels[k][1]=i
#                 topbestmodels[k][2]=topCol            
#         newdf.loc[topbestmodels[k][1], topbestmodels[k][2]]=-1 
# 
# print(topbestmodels)

# 
# 
# 
# for topCol in top:
#         if ( newdf.loc[i, topCol] >topList1[0]):
#             topList3[0] = topList2[0]
#             topList2[0] =topList1[0]
#             topList1[0]=newdf.loc[i, topCol]
#             topList1[1]=i
#             topList1[2]=topCol
#         elif ( newdf.loc[i, topCol] >topList2[0]):
#             topList3[0] = topList2[0]
#             topList2[0]=newdf.loc[i, topCol]
#             topList2[1]=i
#             topList2[2]=topCol
#         elif ( newdf.loc[i, topCol] >topList3[0]):
#             topList3[0]=newdf.loc[i, topCol]
#             topList3[1]=i
#             topList3[2]=topCol
# 
# 
#     print (top)










# top10 = list()
# 
# def process(col):
#     top10.append(col.sort_values(ascending=False).head(2))
# 
# 
# newdf.apply(process,axis=0)
# print ( top10)
# print( type ( top10))
# print( "&&&&&&&&&&&&&&&&&&&&")

#top3= np.argsort(newdf,axis=1).iloc[:,-3:]  
#print ( top3)

# print (newdf.min().min())
# print ( newdf.idxmin())
# print ( newdf.nsmallest(1,newdf[:,:]))


#newdf=newdf.rename(index=np.arange(3, 15, 2))
#unique_counts = newdf.data.nunique()
# array = newdf.values
# 
# lengh = newdf.shape[1]
# print(lengh)
# print((array))
# print ("X=")
# 
# X = array[:,0:lengh-1]
# print (X)
# print ("Y=")
# 
# Y = array[:,lengh-1]
# print(Y)
# print ("$$$$$$$$$$")
# # feature extraction
# test = SelectKBest(score_func=chi2, k=4)
# fit = test.fit(X, Y)
# # summarize scores
# np.set_printoptions(precision=3)
# print(fit.scores_)
# features = fit.transform(X,Y)
# # summarize selected features
# print(features[0:5,:])
#     
#     
#     
# newdf.iloc[3,2]=500
# # newdf[0][2]=5
# # newdf[0][3]=55
# print (newdf)
#print ( unique_counts)
######################Func spliting#############################
# Trainingdata = pd.DataFrame(np.arange(60).reshape(20,3))
# print(Trainingdata)
# #print ( len(Trainingdata))
# #print (Trainingdata.shape[0])
# n_neighbors=list(np.arange(1, Trainingdata.shape[0], 2))
# print (n_neighbors)
# u=4
# m=6
# print (Trainingdata[u:m])
# Kfold=5
# numObservation = Trainingdata.shape[0]
# print (numObservation)
# elemnetPerBlock=int (numObservation/Kfold)
# print (elemnetPerBlock)
# for i in range(Kfold):
#     print("iteration = ",i)
#     TestStart=i*elemnetPerBlock
#     Testend = TestStart+(elemnetPerBlock)
#     print (TestStart , Testend)
#     testpart= Trainingdata[TestStart:Testend]
#     
#     #before testing Part 
#     TrainStart=0
#     Trainend=TestStart
#     TrainingPart = Trainingdata[TrainStart:Trainend]
#     #the remaining Blocks
#     TrainStart=Testend
#     Trainend=numObservation
#     TrainingPart= pd.concat([TrainingPart,Trainingdata[TrainStart:Trainend]])
#     print ( "testing part")
#     print (testpart)
#     print ( "trainingpart")
#     print (TrainingPart)
#     print ("*******************")
#     
#     
# plt.plot([1,2,3,4],[1,2,3,4])
# plt.xlabel('n_neighbors')
# plt.ylabel('AvgAccuracylist')
# plt.show()
# # slices = [Trainingdata[i::Kfold] for i in range(Kfold)]
# # print ("slices")
# # print(slices)
# # for i in range(Kfold):
# #     print("iteration = %d",i)
# #     validation = slices[i]
# #     training = [Trainingdata
# #                 for s in slices if s is not validation
# #                 for Trainingdata in s]
# #     print (validation)
# #     print(training)
# #     print("**********")
#     #yield training, validation