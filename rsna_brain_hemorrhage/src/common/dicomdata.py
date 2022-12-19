import numpy as np
import pydicom

def get_dicom_value(x, cast=int):
    if type(x) in [pydicom.multival.MultiValue, tuple]:
        return cast(x[0])
    else:
        return cast(x)


def cast(value):
    if type(value) is pydicom.valuerep.MultiValue:
        return tuple(value)
    return value


def get_dicom_raw(dicom):
    return {attr:cast(getattr(dicom,attr)) for attr in dir(dicom) if attr[0].isupper() and attr not in ['PixelData']}


def rescale_image(image, slope, intercept, bits, pixel):
    # In some cases intercept value is wrong and can be fixed
    # Ref. https://www.kaggle.com/jhoward/cleaning-the-data-for-rapid-prototyping-fastai
    if bits == 12 and pixel == 0 and intercept > -100:
        image = image.copy() + 1000
        px_mode = 4096
        image[image>=px_mode] = image[image>=px_mode] - px_mode
        intercept = -100
    return image.astype(np.float32) * slope + intercept


def apply_window(image, center, width):
    if isinstance(center,tuple) or isinstance(width, tuple):
        print(f"CENTER : {center}, WIDTH : {width}")
        center = center[0]
        width = width[0]
    image = image.copy()
    min_value = center - width // 2
    max_value = center + width // 2
    image[image < min_value] = min_value
    image[image > max_value] = max_value
    return image


def get_windowed_ratio(image, center, width):
    # get ratio of pixels within the window
    windowed = apply_window(image, center, width)
    return len(np.where((windowed > 0) & (windowed < 80))[0]) / windowed.size


def create_record(id, img_path):
    """
    Read dicom image, fetch metatdata for all parameters
    Get basic statistics for few parameters and store them all in 
    dataframe.
    """
    dicom = pydicom.dcmread(img_path)
    record = {
        'ID': id,
    }
    record.update(get_dicom_raw(dicom))
    raw = dicom.pixel_array
    slope = float(record['RescaleSlope'])
    intercept = float(record['RescaleIntercept'])
    center = get_dicom_value(record['WindowCenter'])
    width = get_dicom_value(record['WindowWidth'])
    bits= record['BitsStored']
    pixel = record['PixelRepresentation']

    image = rescale_image(raw, slope, intercept, bits, pixel)
    doctor = apply_window(image, center, width)
    brain = apply_window(image, 40, 80)
    record.update({
        'raw_max': raw.max(),
        'raw_min': raw.min(),
        'raw_mean': raw.mean(),
        'raw_std': raw.std(),
        'raw_diff': raw.max() - raw.min(),
        'doctor_max': doctor.max(),
        'doctor_min': doctor.min(),
        'doctor_mean': doctor.mean(),
        'doctor_std': doctor.std(),
        'doctor_diff': doctor.max() - doctor.min(),
        'brain_max': brain.max(),
        'brain_min': brain.min(),
        'brain_mean': brain.mean(),
        'brain_std': brain.std(),
        'brain_diff': brain.max() - brain.min(),
        'brain_ratio': get_windowed_ratio(image, 40, 80),
    })
    return record


def dicomDataframe(train):
    dicom_metadata=[]
    corrupted_ids=[]
    for index, row in train.iterrows():
        try:
            record = create_record(row['ID'], row['filepath'])
            dicom_metadata.append(record)
        except:
            corrupted_ids.append(row['ID'])
            print(f'Corrupted image :' + row['ID'])
            continue
    return dicom_metadata, corrupted_ids