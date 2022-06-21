import os
import datetime

from keras.preprocessing.image import ImageDataGenerator
from keras import layers, models
from classification_models.keras import Classifiers

from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard


def create_datagen(train_data, test_data):
    train_datagen = ImageDataGenerator(
        validation_split=0.2
    )
    test_datagen = ImageDataGenerator()
    train_generator = train_datagen.flow_from_directory(directory=train_data, target_size=(128, 128), color_mode='rgb',
                                                        batch_size=16, class_mode='categorical', shuffle=True,
                                                        seed=942212, subset='training')
    val_generator = train_datagen.flow_from_directory(directory=train_data, target_size=(128, 128), color_mode='rgb',
                                                      batch_size=16, class_mode='categorical', subset='validation',
                                                      seed=942212, shuffle=True)
    test_generator = test_datagen.flow_from_directory(directory=test_data, target_size=(128, 128), color_mode='rgb',
                                                      batch_size=1, class_mode=None, shuffle=False, seed=942212)
    return train_generator, val_generator, test_generator



def run_resnet34(train_data, test_data):
    train_data, val_data, test_data = create_datagen(train_data, test_data)

    ResNet34, preprocess_unit = Classifiers.get('resnet34')
    base_model = ResNet34(input_shape=(128, 128, 3), weights='imagenet', include_top=False)
    x = layers.GlobalAveragePooling2D()(base_model.output)
    output = layers.Dense(4, activation='softmax')(x)
    model = models.Model(inputs=[base_model.input], outputs=[output])
    model.compile(optimizer='SGD', loss='categorical_crossentropy', metrics=['accuracy'])

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='min')
    mcp_save = ModelCheckpoint('./resnet34.py/rn34.hdf5', save_best_only=True, monitor='val_loss', mode='min')
    reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7,
                                       verbose=1, min_delta=1e-4, mode='min')
    tensorboard_callback = TensorBoard(
        log_dir=f'./logs/ResNet34/{datetime.datetime.now().strftime("%Y%m%d%H%M%S")}')

    model.fit(train_data, epochs=20, callbacks=[early_stopping, mcp_save, reduce_lr_loss, tensorboard_callback],
              batch_size=16, validation_data=val_data)


if __name__ == '__main__':
    train_data = '/home/staeros/nas-fedot/datasets/Blood-Cell-Classification/train'
    test_data = '/home/staeros/nas-fedot/datasets/Blood-Cell-Classification/test'
    run_resnet34(train_data, test_data)
