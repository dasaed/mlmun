def NormalizationByRow(Data):

    Transpose=Data.transpose()
    print(Transpose)
    for column in Data:
        
        mean=np.mean(Transpose[column])
        std=Transpose[column].std()
        print (mean)
        print ( std)
        Transpose[column]=Transpose[column].apply(lambda x: (x - mean)/(std))
        print (Transpose[column])
        break
        
    return Data


def ValidationSetApproach(trainingdata, K,DistanceType):
    randomTrainingData=trainingdata.sample(frac=1)
    print ( "randomTrainingData")
    print (randomTrainingData)
    tdRows=int( randomTrainingData.shape[0])
    print("Number of rows in TD: "+str(tdRows))
    tdsize = int(math.floor(tdRows/5))
    print ("first part\n")
    TrainPart =  randomTrainingData.head(tdsize*3)
    print("##############################################################")
    print(TrainPart)
    ValidationPart =  randomTrainingData[(tdsize*3):(tdsize*4)]
    print("##############################################################")
    print(ValidationPart)
    TestPart =  randomTrainingData[tdsize*4:]
    print("##############################################################")
    print(TestPart)
    
    ClassicalKNN(TrainPart,ValidationPart,K,DistanceType)  # output in csv
    output = pd.read_csv( "output.csv", sep='\t') 
    print (output)
    df1=output['Class']
    #df2=trainingdata.DataFrame(columns=['Class'])
    df2=ValidationPart['Class']
    #df1 = output['Class']

    df1Size=int(df1.shape[0])
    df2=ValidationPart['Class']
    df2Size=int(df2.shape[0])
    print ("df1 size = \n")
    print (df1Size)
    print ("df2 size = \n")
    print (df2Size)
    print ("df1 \n")
    print (df1)
    print ("df2 \n")
    print (df2)
    print (list (df2))
    if (df1.equals(df2)):
        print( "Algorithm is matching the test results by 100%")
    else:
        dfsim = df2.where(df2 == df1)
        print("Number of Errors:"+str(dfsim.isna().sum()))
        
    return 23
#===============================Calculate performance metric======
def precisionandRecall (labels,predictions):
    cm = confusion_matrix(labels, predictions)
    recall = np.diag(cm) / np.sum(cm, axis = 1)
    precision = np.diag(cm) / np.sum(cm, axis = 0)
    
    print ("recall")
    print (recall)
    
    print("precision")
    print(precision)
    return

def precision(y_true, y_pred):
    i = set(y_true).intersection(y_pred)
    len1 = len(y_pred)
    if len1 == 0:
        return 0
    else:
        return len(i) / len1


def recall(y_true, y_pred):
    i = set(y_true).intersection(y_pred)
    return len(i) / len(y_true)


def f1(y_true, y_pred):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    if p + r == 0:
        return 0
    else:
        return 2 * (p * r) / (p + r)
