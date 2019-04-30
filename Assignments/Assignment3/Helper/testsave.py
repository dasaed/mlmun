import numpy as np
import pandas as pd
 
 
 
 
a = np.arange(15).reshape(3, 5)
print ( a)
my=pd.DataFrame(a)
my = my.rename_axis(None)
#my2 = my.iloc[:,1:]
#npmy2.savetxt('testOut.tsv', a, delimiter='\t')
my.to_csv('/home/mun/Dropbox/MachineLearning/MachineLearning_Shared/Assignments/Assignment3/output1.tsv',sep='\t', header=0,index=False)