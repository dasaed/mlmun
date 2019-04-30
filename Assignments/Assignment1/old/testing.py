#=========== test via Validation Set Approach =========
import pandas as pd
import numpy as np
#tdRows=int(trainingdata.shape[0])
#print("Number of rows in TD: "+str(tdRows))
#tdsize = int(math.floor(tdRows/5))
#print("Size of Validation data: "+str(tdsize))
#print ("first part\n")
#tdp1 = trainingdata.head(tdsize*3)
#print("##############################################################")
#print(tdp1)
#tdp2 = trainingdata[(tdsize*3):(tdsize*4)]
#print("##############################################################")
#print(tdp2)
#tdp3 = trainingdata[tdsize*4:]
#print("##############################################################")
#print(tdp3)



k=3
def valSetApproach(k):
    df1 = pd.DataFrame( {'col1': ['A', 'A', 'B', 'A', 'D', 'C'], 'col2': [2, 1, 9, 8, 7, 4], 'col3': [0, 1, 9, 4, 2, 3], })
    df2 = pd.DataFrame( {'col1': ['B', 'A', 'A', 'A', 'D', 'C'], 'col2': [2, 1, 9, 8, 7, 4], 'col3': [0, 1, 9, 4, 2, 3], })
    print("######### Set 1 ##########")
    print(df1)
    print("######### Set 2 ###########")
    print(df2)
    #outputSize=int(trainingdata.shape[0])
    if (df1.equals(df2)):
        return "Algorithm is matching the test results by 100%"
    else:
        print("Where the columns are different")
        dfdiff = df2.where(df2 != df1)
        print(dfdiff)
        #errores=dfdiff.loc[ dfdiff['col1'] == 'NaN' ]
        #print(errores)
        #print("Errors Found: "+str(errores))
        print("Where the columns are similar")
        dfsim = df2.where(df2 == df1)
        print(dfsim)

        print("Number of Errors:  \n"+str(dfsim.isna().sum()))
        #print("Merged Sets")
        #dfmer=pd.merge(df1,df2, on=['col1'],how='outer')
        #print(dfmer)
valSetApproach(k)