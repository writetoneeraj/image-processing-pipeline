from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input as preprocess_resnet_50
from keras.applications.vgg16 import preprocess_input as preprocess_vgg_16
from keras.layers import GlobalAveragePooling2D, Dense, Activation, concatenate, Dropout
from keras.initializers import glorot_normal, he_normal
from keras.regularizers import l2
from keras.models import Model, load_model
from keras.utils import Sequence
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.optimizers import Adam


def define_checkpoints(path):
    model_checkpoint = ModelCheckpoint(filepath=path, monitor='val_loss', verbose=1, mode='min', 
                                       save_best_only=True, save_weights_only=True,period=1)
    reduceonplateu = ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=2, verbose=1, mode="min", min_lr = 1e-8)
    early_stopping = EarlyStopping(monitor="val_loss",min_delta=0.01, mode="min", patience=5, verbose=1, restore_best_weights=True)
    return[model_checkpoint, reduceonplateu, early_stopping]
    
MODELOUT_PATH = '../input/rsna_models/weights/'
class Builder:
    def __init__(self, 
        model_fun,
        metrics_list,
        loss_fun,
        num_classes,
        train_generator,
        valid_generator,
        epochs,
        checkpoint_path = MODELOUT_PATH):
        self.metrics_list = metrics_list
        self.loss_fun = loss_fun
        self.num_classes = num_classes
        self.epochs = epochs
        self.train_generator = train_generator
        self.valid_generator = valid_generator
        self.checkpoint_path = checkpoint_path
        self.model_checkpointpath = self.checkpoint_path + f'weights.{epoch:02d}-{val_loss:.2f}.hdf5'
        self.callbacks=define_checkpoints(self.model_checkpointpath)
        self.base_model=ResNet50(include_top=False, weights='./weights/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5')
        
    def buildmodel(self):
        model = self.model_fun
        x = model.output()
        x = GlobalAveragePooling2D()(x)
        x = Dropout(0.3)(x)
        x = Dense(100, activation="relu")(x)
        x = Dropout(0.3)(x)
        pred = Dense(self.num_classes,
                     kernel_initializer=he_normal(seed=11),
                     kernel_regularizer=l2(0.05),
                     bias_regularizer=l2(0.05), activation="sigmoid")(x)
        self.model = Model(inputs=self.base_model.input, outputs=pred)
    
    def compile_model(self):
        self.model.compile(optimizer=Adam(learning_rate=LR),
                           loss=self.loss_fun, 
                           metrics=self.metrics_list)
    
    def learn(self):
        return self.model.fit_generator(generator=self.train_generator,
                    validation_data=self.valid_generator,
                    epochs=self.epochs,
                    callbacks=self.callbacks,
                    #use_multiprocessing=False,
                    workers=8)

    def predict(self, test_generator):
        return self.model.predict_generator(test_generator, workers=8)