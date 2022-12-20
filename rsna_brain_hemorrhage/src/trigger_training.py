import pandas as pd
import numpy as np
from configparser import ConfigParser
from logging import Logger
import random

from common.commonconfigparser import CommonConfigParser
from common.log.log_utility import intialise_logging
from training.datagenerator import DataGenerator
from imagepreprocess.preprocess_images import PreProcessor
import training.modelbuilder as builder
import tensorflow as tf


def _getModelConfig(modelConfig, config):
    _modelConfig = modelConfig.get(config.getModelName(), str.lower(config.getModelName())+".config")
    try:
        if _modelConfig and not _modelConfig.isspace(): 
            return eval(_modelConfig)
    except Exception as ex:
        raise ("Model configurations are not properly defined. Please check the configurations again.{ex}")


def _getModelAugmentConfig(config: CommonConfigParser, logger: Logger):
    mConfig = ConfigParser()
    augConfig = ConfigParser()
    logger.info(f"Configuration Path : {config.getConfigPath()}")
    logger.info(f"Model Configuration Path : [{config.getModelConfigurationPath()}]")
    logger.info(f"Augmentation Path : [{config.getAugmentationPath()}]")
    mConfig.read(config.getConfigPath() + config.getModelConfigurationPath())
    augConfig.read(config.getConfigPath() + config.getAugmentationPath())
    modelConfig = _getModelConfig(mConfig, config)
    return modelConfig, augConfig


def _get_generators(
                        logger: Logger,
                        config: CommonConfigParser,
                        modelConfig,
                        augConfig
                    ):
    train_data = pd.read_pickle(config.getTrainingData())
    test = pd.read_pickle(config.getTestData())
    folds = train_data.fold.unique()
    valid_fold = random.randint(np.min(folds), np.max(folds))
    train = train_data[train_data.fold != valid_fold]
    train_labels = train[config.getModelLabelNames()]
    valid = train_data[train_data.fold == valid_fold]
    valid_labels = valid[config.getModelLabelNames()]
    test_labels = test[config.getModelLabelNames()]
    train_generator = DataGenerator(train, train_labels, PreProcessor(logger), modelConfig, augConfig, "TRAIN")
    valid_generator = DataGenerator(valid, valid_labels, PreProcessor(logger), modelConfig, augConfig, "VALID")
    test_generator = DataGenerator(test, test_labels, PreProcessor(logger), modelConfig, augConfig, "TEST")
    return train_generator, valid_generator, test_generator


def _getBuilder(
                    logger,
                    train_generator,
                    valid_generator,
                    test_generator,
                    modelConig
                ):
    builder_class = modelConfig.get("builder")
    builder_object = getattr(builder, builder_class)(train_generator, valid_generator, test_generator, modelConfig)
    logger.info(f"Builder {builder_object} initialized")
    return builder_object


if __name__ == "__main__":
    config =  CommonConfigParser('common_config.ini')
    logger = intialise_logging(config)
    logger.info("Starting training for RSNA")
    modelConfig, augConfig = _getModelAugmentConfig(config, logger)
    train_generator, valid_generator, test_generator = _get_generators(logger, config, modelConfig, augConfig)
    builder = _getBuilder(logger, train_generator, valid_generator, test_generator, modelConfig)
    builder.buildmodel(logger=logger)
    builder.compile_model(logger)
    builder.learn(logger)
    predict = builder.predict()
    logger.info(f"Predict : {predict}")
