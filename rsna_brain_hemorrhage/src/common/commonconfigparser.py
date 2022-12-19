from configparser import ConfigParser
import os
from pathlib import Path
from typing import List


class CommonConfigParser:
    def __init__(self, *config_files):
        self.config = ConfigParser()
        # Location of the config file : <project_home>/config/config.ini
        self.config_path = f"{Path(__file__).parent.parent.parent.resolve()}/config/"
        _files = [self.config_path + "common_config/" + lst for lst in config_files]
        print(f"_files : {_files}")
        self.config.read(_files)

    def getConfigPath(self):
        """ Returns base configurations folder path"""
        return self.config_path

    def get_preprocess_path(self):
        """
        Get preprocessing script file for preprocessing data
        """
        return self.config.get('PREPROCESS', 'preprocess.script')

    def get_log_config_path(self):
        """
        Get the path for logging configuration file
        """
        return self.config.get('LOGS', 'log.config.path')

    def get_dataset_name(self):
        """
        Get name of dataset for which model will be trained.
        """
        return self.config.get('DATASET','dataset.name')

    def get_dataset_version(self):
        """
        Get dataset version, to track models trained on which dataset version
        """
        return self.config.get('DATASET','dataset.version')

    def getModelConfigurationPath(self):
        """Returns path for model configurations"""
        return self.config.get("MODEL", "model.configurations.path")

    def getModelName(self)-> List:
        """ Return model config for the input model."""
        return self.config.get("MODEL", "model.name")

    def getDataOutputDir(self):
        """Returns data output directory path"""
        return self.config.get("DATA", "data.output", fallback="output")
    
    def getTrainingData(self):
        """Returns training data path """
        return self.config.get("DATA", "data.train.data")
    
    def getTestData(self):
        """Returns Test Data"""
        return self.config.get("DATA", "data.test.data")
    
    def getAugmentationPath(self):
        """Returns augmentation path """
        return self.config.get("AUGMENTATION", "augmentation.config")

    def getModelLabelNames(self):
        """ Returns label/target columns"""
        labels = self.config.get("MODEL", "model.train.labels")
        if labels and len(labels) > 0:
            labelNames = labels.split(",")
        else:
            labelNames = []
        return labelNames

    def get_application_name(self):
        """Returns application name"""
        return self.config.get("COMMON", "application.name")

    def get_application_version(self):
        """Returns application version"""
        return self.config.get("COMMON", "application.version")
