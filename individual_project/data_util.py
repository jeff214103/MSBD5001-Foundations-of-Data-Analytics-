import pandas as pd
import os


class DataUtil:
    def __init__(self,path="dataset"):
        self.path = path

    def featureFilter(self,df):
        df = df.drop(['id'], axis=1)
        df = df.dropna(how='any', axis = 0)
        return df

    # Return features, label
    def readTrainData(self,train="train.csv"):
        df = pd.read_csv(os.path.join(self.path,train))
        df = self.featureFilter(df)
        return df.iloc[:,:-1],df.iloc[:,-1:]

    def readTestData(self,test="test.csv"):
        df = pd.read_csv(os.path.join(self.path,test))
        df = self.featureFilter(df)
        return df


    def outputFile(self,arr,output="submission.csv"):
        df = pd.DataFrame(arr, columns=['label'])
        df.index.name = "id"
        df.to_csv(output)