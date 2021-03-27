import pydicom
from misc.dicomdata import apply_window, rescale_image
import numpy as np
from factory import get_transforms
"""
This class preprocess the image before training.
"""
class PreProcessor:
    def __init__(self, row, model_config):
        self.row = row
        self.model_config = model_config

    
    def transpose_image(self, dicom):
        image1 = apply_window(dicom.pixel_array, *(self.model_config.brain_window()))
        image2 = apply_window(dicom.pixel_array, *(self.model_config.subdural_window()))
        image3 = apply_window(dicom.pixel_array, *(self.model_config.other_window()))
        return(np.array([image1, image2, image3]).transpose(1,2,0))

    def process(self):
        dicom = pydicom.dcm_read(self.row.img_path)
        # change names of columns as per column names
        image = rescale_image(dicom.pixel_array, self.row.slope, self.row.intercept, self.row.bits, self.row.pixel)
        image = self.transpose_image(dicom)
        transpose = get_transforms(self.model_config)
        image = transpose(image=image)['image']
        return image