import keras as k
import numpy as np

class DataGenerator(Sequence):
    def __init__(train, label_col, preprocessor, batch_size, num_classes, input_size, nchannels):
        self.train = train
        self.label_col = label_col
        self.batch_size = batch_size
        self.preprocessor = preprocessor
        self.num_classes = num_classes
        self.input_size = input_size
        self.nchannels = nchannels

    def  __len__():
        return len(self.train)/self.batch_size
        
    
    def __get_item__(self, idx):
        batch_x = self.train[idx*self.batch_size : (idx+1)*self.batch_size]
        labels = self.train[[self.label_col]]
        X = np.empty((self.batch_size, *(self.input_size), self.nchannels))
        y = np.empty(self.batch_size, self.num_classes)
        for i, row in enumerate(batch_x):
            X[i,] = self.preprocessor.process(row.filename)
            y[i,] =  labels.iloc[i]
        return X,y