import abc

import tensorflow as tf
from tensorflow import keras
from keras.applications.resnet import preprocess_input as preprocess_resnet_50
from keras.applications.vgg16 import preprocess_input as preprocess_vgg_16
from keras.layers import GlobalAveragePooling2D, Dense, Activation, concatenate, Dropout
from keras.initializers import glorot_normal, he_normal
from keras.regularizers import l2
import keras.optimizers as optimizer
from keras.models import Model, load_model
from keras.utils import Sequence
import keras.callbacks as callbacks
from logging import Logger
from typing import List
import keras.metrics as metrics

import src.loss.focalloss as loss


class Builder:
    @abc.abstractmethod
    def buildmodel(self):
        pass
    
    @abc.abstractmethod
    def compile_model(self):
        pass
    
    @abc.abstractmethod
    def learn(self):
        pass

    @abc.abstractmethod
    def predict(self, test_generator):
        pass


class Resnet50Builder(Builder):
    def __init__(
                    self,
                    train_generator,
                    valid_generator,
                    test_generator,
                    modelConfig
                ):
        super(Resnet50Builder, self).__init__()
        self.train_generator = train_generator
        self.test_generator = test_generator
        self.valid_generator = valid_generator
        self.modelConfig = modelConfig
        self.model = None
  
    def _get_model(self):
        net = tf.keras.applications.ResNet50(include_top=False, weights='imagenet', classes=6)
        for layer in net.layers:
            layer.trainable = False
        return net

    def _getCallbacks(self, logger: Logger):
        callbacks_list = self.modelConfig.get("callbacks")
        callback_objects = []
        if callbacks_list and len(callbacks_list) > 0:
            for callback in callbacks_list:
                callbackConfig = self.modelConfig.get(callback)
                params = callbackConfig.get("params")
                if params:
                    obj = getattr(callbacks, callbackConfig.get("class"))(**params)
                else:
                    obj = getattr(callbacks, callbackConfig.get("class"))
                callback_objects.append(obj)
                logger.info(f"Callback [{obj}] added in callbacks list")
        return callback_objects
    
    def _getOptimizer(self, logger: Logger):
        try:
            optimizerName = self.modelConfig.get("optimizer")
            optimizerParams = self.modelConfig.get(optimizerName)
            if optimizerParams and len(optimizerParams) > 0:
                optimizerObj = getattr(optimizer, optimizerName)(**optimizerParams)
            else:
                optimizerObj = getattr(optimizer, optimizerName)(lr=self.modelConfig.get("learning_rate"))
            return optimizerObj
        except Exception as ex:
            logger.error(f"Error in retrieving optimizer object. {ex}")
    
    def _getLossFunction(self, logger: Logger):
        loss_func = self.modelConfig.get("loss_function")
        loss_obj = getattr(loss, loss_func)(**self.modelConfig.get(loss_func))
        logger.info(f"Initializing loss function : {loss_obj}")
        return loss_obj
    
    def _getMetrics(self, logger: Logger):
        metricsList = list(self.modelConfig.get("metrics_list"))
        metrices = []
        for metric in metricsList:
            params = self.modelConfig.get(metric)
            logger.info(f"[{metric}] got added in Metrices List with Params [{params}]")
            metricObj = getattr(metrics, metric)
            if params:
                metrices.append(metricObj(**params))
            else:
                metrices.append(metricObj())
        return metrices
    
    def buildmodel(self, logger: Logger):
        base_model = self._get_model()
        logger.info(f"Model [{base_model}] got initialised")
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dropout(0.3)(x)
        x = Dense(100, activation="relu")(x)
        x = Dropout(0.3)(x)
        pred = Dense(self.modelConfig.get("num_classes"),
                     kernel_initializer=he_normal(seed=11),
                     kernel_regularizer=l2(0.05),
                     bias_regularizer=l2(0.05), activation="sigmoid")(x)
        model = Model(inputs=base_model.input, outputs=pred)
        logger.info(f"Model Summary : {model.summary()}")
        self.model = model
    
    def compile_model(self, logger: Logger):
        self.model.compile(optimizer=self._getOptimizer(logger),
                           loss=self._getLossFunction(logger),
                           metrics=self._getMetrics(logger)
                        )
    
    def learn(self, logger: Logger):
        return self.model.fit_generator(generator=self.train_generator,
                    validation_data=self.valid_generator,
                    epochs=self.modelConfig.get("epochs"),
                    callbacks=self._getCallbacks(logger),
                    workers=8)
    
    def predict(self):
        return self.model.predict(self.test_generator, workers=8)