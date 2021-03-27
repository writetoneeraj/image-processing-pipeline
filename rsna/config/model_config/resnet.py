folds=[1,2,3,4,5]
seed=300
group_ID='PatientID'
gpu_required=0
output_dir="./data/output/models_output"

# hyperparameters
learning_rate=0.001
loss_function='FOCALLOGLOSS'
epoch=10
resume_from=None
batch_size=32
num_workers=4
imgsize=(512,512)
pretrained='imagenet'

# Optimiser details
optim = dict(optim='Adam')

# Scheduler details
scheduler = dict(
name='MultiStepLR',
milesones=[1,2,3],
gamma=3/7)

# Augmentation
crop = dict(name='RandomResizedCrop', params=dict(height=imgsize[0], width=imgsize[1], scale=(0.7,1.0), p=1.0))
crop_test = dict(name='RandomResizedCrop', params=dict(height=imgsize[0], width=imgsize[1], scale=(0.75,1.0), p=1.0))
resize = dict(name='Resize', params=dict(height=imgsize[0], width=imgsize[1]))
hflip = dict(name='HorizontalFlip', params=dict(p=0.5,))
vflip = dict(name='VerticalFlip', params=dict(p=0.5,))
contrast = dict(name='RandomBrightnessContrast', params=dict(brightness_limit=0.08, contrast_limit=0.08, p=0.5))
#totensor = dict(name='ToTensor', params=dict(normalize=normalize))
rotate = dict(name='Rotate', params=dict(limit=30, border_mode=0), p=0.7)
rotate_test = dict(name='Rotate', params=dict(limit=25, border_mode=0), p=0.7)
dicomnoise = dict(name='RandomDicomNoise', params=dict(limit_ratio=0.06, p=0.9))
dicomnoise_test = dict(name='RandomDicomNoise', params=dict(limit_ratio=0.05, p=0.7))

window_policy = 4

data = dict(
    train=dict(
        dataset_type='DataGenerator',
        annotations='./processed_data/train_folds8_seed300.pkl',
        imgdir='./data/stage_2_train_images',
        imgsize=imgsize,
        n_grad_acc=1,
        loader=dict(
            shuffle=True,
            batch_size=batch_size,
            drop_last=True,
            num_workers=num_workers,
            pin_memory=False,
        ),
        transforms=[crop, hflip, rotate, dicomnoise],
        dataset_policy=1,
        window_policy=window_policy,
    ),
    valid = dict(
        dataset_type='DataGenerator',
        annotations='./processed_data/train_folds8_seed300.pkl',
        imgdir='./data/stage_2_train_images',
        imgsize=imgsize,
        loader=dict(
            shuffle=False,
            batch_size=batch_size,
            drop_last=False,
            num_workers=num_workers,
            pin_memory=False,
        ),
        transforms=[crop_test, hflip, rotate_test, dicomnoise_test],
        dataset_policy=1,
        window_policy=window_policy,
    ),
    test = dict(
        dataset_type='DataGenerator',
        annotations='./processed_data/test.pkl',
        imgdir='./data/stage_2_test_images',
        imgsize=imgsize,
        loader=dict(
            shuffle=False,
            batch_size=batch_size,
            drop_last=False,
            num_workers=num_workers,
            pin_memory=False,
        ),
        transforms=[crop_test, hflip, rotate_test, dicomnoise_test],
        dataset_policy=1,
        window_policy=window_policy,
    ),
)