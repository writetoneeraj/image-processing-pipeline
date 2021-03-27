python preprocess/createdataset.py --input ./data/train.csv --output ./output/processed_data/train_dicom.pkl --img_dir ./data/train_images/
python preprocess/createdataset.py --input ./data/test.csv --output ./output/processed_data/test_dicom.pkl --img_dir ./data/test_images/
python preprocess/make_folds.py --input ./output/processed_data/train_dicom.pkl --output ./output/processed_data/train_folds.pkl --method group_fold --folds 3 --group PatientID --seed 300
