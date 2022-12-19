# First step is to pivot the row wise data in column wise data, get data from dicom images and merge with existing dataframe.
python src/datapreprocess/createdataset.py --input ./data/train.csv --output ./output/processed_data/train_dicom.pkl --img_dir ./data/train_images/
python src/datapreprocess/createdataset.py --input ./data/test.csv --output ./output/processed_data/test_dicom.pkl --img_dir ./data/test_images/

python src/datapreprocess/processdataset.py --input ./output/processed_data/train_dicom.pkl --output ./output/processed_data/train.pkl --brain-diff 8.0
python src/datapreprocess/processdataset.py --input ./output/processed_data/test_dicom.pkl --output ./output/processed_data/test.pkl --brain-diff 8.0
python src/datapreprocess/make_folds.py --input ./output/processed_data/train.pkl --output ./output/processed_data/train_folds.pkl --method group_fold --folds 3 --group PatientID --seed 300

