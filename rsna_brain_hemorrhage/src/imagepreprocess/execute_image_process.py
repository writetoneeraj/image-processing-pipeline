from src.imagepreprocess import preprocess_images
import pandas as pd
from joblib import Parallel, delayed
import multiprocessing
from keras.utils import Sequence
from configparser import ConfigParser
from src.common.commonconfigparser import CommonConfigParser

from argparse import ArgumentParser

"""
def argumentParser():
    parser = ArgumentParser()
    parser.add_argument("--input")
    parser.add_argument("--train-image-folder")
    parser.add_argument("--train-image-output")
    parser.add_argument("--config")
    return parser.parse_args()
"""

class ImageAugmentation(Sequence):
    def __init__(
                    self,
                    df:pd.DataFrame,
                    config: CommonConfigParser,
                    modelConfig,
                    aug_config
                ):
        self.data = df
        self.config = config
        self.train_image_folder = self.config.getTrainImageDirPath()
        self.train_process_output_folder = self.config.getProcessTrainImageOutput()
        self.modelConfig = modelConfig
        self.aug_config = aug_config
    
    def __len__(self):
        return self.data.shape[0]/self.config.batch_size 

    def _process_image(self, df: pd.DataFrame, batch_size):
        for i in range(0, df.shape[0], batch_size):
            print(i, "    ", i+batch_size)
            retlist = Parallel(
                                n_jobs=multiprocessing.cpu_count()
                            )(delayed(preprocess_images.process)(
                                                                row,
                                                                self.modelConfig,
                                                                self.aug_config,
                                                                self.config.getProcessTrainImageOutput(),
                                                                True
                                                            ) for row in df[i:i+batch_size].iterrows()
                            )
            return retlist

    def __getitem__(self, idx):
        batch_size = self.config.batch_size
        batch = self.data[idx*batch_size:(idx+1)*batch_size]
        retlist = self._process_image(batch, batch_size)

def _getModelAugmentConfig(config: CommonConfigParser):
    mConfig = ConfigParser()
    aug_config = ConfigParser()
    mConfig.read(config.getConfigPath() + config.getModelConfigurationPath()) # log in as logger.debug comment
    aug_config.read(config.getConfigPath() + config.getAugmentationPath())
    modelConfig = _getModelConfig(mConfig, config)
    return modelConfig, aug_config

def _process_image(df: pd.DataFrame, config: CommonConfigParser):
    modelConfig, aug_config = _getModelAugmentConfig(config)
    batch_size = modelConfig.get("batch_size")
    for i in range(0, df.shape[0], batch_size):
        print(i, "    ", i+batch_size)
        retlist = Parallel(
                            n_jobs=multiprocessing.cpu_count()
                        )(delayed(preprocess_images.process)(
                                                                row,
                                                                modelConfig,
                                                                aug_config,
                                                                config.getProcessTrainImageOutput(),
                                                                True
                                                            ) for row in df[i:i+batch_size].iterrows()
                        )
        return retlist

def _getModelConfig(modelConfig, config):
    _modelConfig = modelConfig.get(config.getModelName(), str.lower(config.getModelName())+".config")
    try:
        if _modelConfig and not _modelConfig.isspace(): 
            return eval(_modelConfig)
    except:
        raise ValueError("Model Configurations not properly defines. Please check configuration again")


# if this module is not called directly then also should work. May be as continuos pipeline from processing data.
# Create a else block and move augmentation configuration onwards in the block.
if __name__ == "__main__":
    # initialise logger
    config = CommonConfigParser('common_config.ini')
    train_image_folder = config.getTrainImageDirPath()
    train_process_output_folder = config.getProcessTrainImageOutput()
    train = pd.read_pickle(config.getTrainingDataPath())
    _process_image(train, config)
