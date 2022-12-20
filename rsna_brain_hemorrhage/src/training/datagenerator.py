import tensorflow as tf
from keras.utils import Sequence
import numpy as np
from random import shuffle
import math


class DataGenerator(Sequence):
    def __init__(
                    self,
                    data,
                    label_col,
                    preprocessor,
                    modelConfig,
                    augConfig,
                    data_type):
        self.data = data
        self.label_col = label_col
        self.batch_size = modelConfig.get("batch_size")
        self.preprocessor = preprocessor
        self.num_classes = modelConfig.get("num_classes")
        self.input_size = modelConfig.get("input_size")
        self.nchannels = modelConfig.get("nchannels")
        self.shuffle = modelConfig.get("shuffle")
        self.image_idx = self.data.index.values
        self.current_epoch = 0             
        self.modelConfig = modelConfig
        self.augConfig = augConfig
        self.data_type = data_type
        self.on_epoch_end()

    def __len__(self):
        return math.ceil(self.data.shape[0]/self.batch_size)

    def __getitem__(self, idx):
        batch_x = self.data.iloc[idx*self.batch_size : (idx+1)*self.batch_size]
        labels = self.label_col.iloc[idx*self.batch_size : (idx+1)*self.batch_size]
        X = np.empty((self.batch_size, *(self.input_size), self.nchannels))
        y = np.empty((self.batch_size, self.num_classes))
        j = 0
        for i, row in batch_x.iterrows():
            X[j,] = self.preprocessor.process(
                                                row,
                                                self.modelConfig,
                                                self.augConfig,
                                                " ", False)
            y[j,] =  self.label_col.iloc[j]
            j += 1
        return np.array(X), np.array(y)
    
    def __next__(self):
        return self

    def __iter__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)

    def __call__(self):
        return self

    def on_epoch_end(self):
        if(self.shuffle):
            shuffle(self.image_idx)
        self.current_epoch += 1 