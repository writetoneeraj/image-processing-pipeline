from configparser import ConfigParser
import os
from pathlib import Path

_config = ConfigParser()

# Location of the config file : <project_home>/config/config.ini
config_path = Path(__file__).parent.parent.resolve()/'conf'/'config.ini'
try:
    with open(config_path) as f:
        _config.read(config_path, encoding='utf-8')
except Exception as ex:
    raise Exception(
        'Failed to read the config file. Stopping the application') from ex

def get_run_models():
    """
    Get model names to run
    """
    models = _config.get('MODELS','models.cnn')
    return models

def get_model_config(model_nm):
    """Return model config for the input model"""
    models_config = _config.get('MODEL','models.config')
    model_config = models_config.get(model_nm)
    return model_config


def get_preprocess_path():
    """
    Get preprocessing script file for preprocessing data
    """
    return _config.get('PREPROCESS', 'preprocess.script')

def get_log_config_path():
    """
    Get the path for logging configuration file
    """
    return _config.get('LOGS', 'log.config')

def get_dataset_name():
    """
    Get name of dataset for which model will be trained.
    """
    return _config.get('DATASET','dataset.name')

def get_dataset_version():
    """
    Get dataset version, to track models trained on which dataset version
    """
    return _config.get('DATASET','dataset.version')

def getModelOutputDir():
    """Returns model outpput directory path"""
    return _config.get('PATH','data.model.output')