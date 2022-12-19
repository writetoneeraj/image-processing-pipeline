import pydicom
import cv2
from src.common.dicomdata import apply_window, rescale_image
from src.common.commonconfigparser import CommonConfigParser
import numpy as np
from src.transforms import augmentation
from matplotlib import pyplot as plt
import albumentations as A


"""
This class preprocess the image before training.
"""
class PreProcessor:
    def __init__(self, logger):
        self.logger = logger

    def transpose_image(self, row, modelConfig, dicom):
        image1 = apply_window(dicom.pixel_array, *(modelConfig.get("brain_window")))
        image2 = apply_window(dicom.pixel_array, row.WindowCenter, row.WindowWidth)
        image3 = apply_window(dicom.pixel_array, *(modelConfig.get("other_window")))
        return(np.array([image1, image2, image3]).transpose(1,2,0))

    def _write_image_to_folder(self,
                                processed_image_output_path: str,
                                filepath: str,
                                image: np.ndarray
                            ):
        filename = filepath.split("/")[-1]
        filename = filename.split('.')[0]
        cv2.imwrite(f"{processed_image_output_path}{filename}.png", image)

    def normalize(self, image, row):
        return (image - row.ct_level)/(row.ct_width-row.ct_level) * 0.5
    
    def process(self,
                row,
                modelConfig,
                augmentation_config,
                processed_image_output_path,
                save_to_file
            ):
        dicom = pydicom.dcmread(row.filepath)
        image = rescale_image(dicom.pixel_array, row.RescaleSlope, row.RescaleIntercept, row.BitsStored, row.PixelRepresentation)
        transforms_list = augmentation_config.get("AUGMENT", "transforms").split(",")
        self.logger.info(f"Transform List to be performed on image : {transforms_list}")
        transforms = []
        if transforms_list is not None and len(transforms_list) > 0:
            for ctrans in transforms_list:
                function_config = augmentation_config.get('AUGMENT', ctrans)
                if function_config and not function_config.isspace():
                    try:
                        ctrans_value = eval(function_config)
                        if hasattr(A, ctrans_value.get('name')):
                            transforms.append(getattr(A, ctrans_value.get('name'))(**ctrans_value.get('params')))
                        else:
                            transforms.append(getattr(augmentation, ctrans_value.get('name'))(**ctrans_value.get('params')))
                    except Exception as ex:
                        self.logger.info(f"Execution of function [{ctrans}] failed. [{ex}]") # replace with logger.info
                        raise(ex)
                else:
                    self.logger.info(f"Function [{ctrans}] not defined. Please check augmentation configuration.") # replace with logger.info
        transform = A.Compose(transforms)
        image = transform(image=image)["image"]
        image = self.transpose_image(row, modelConfig, dicom)
        if save_to_file:
            self._write_image_to_folder(processed_image_output_path, row.filepath, image)
        return image