#-------------------------------------------------------------------------------
# Name:        module1
# Purpose:
#
# Author:      Adam Tejkl
#
# Created:     05.08.2021
# Copyright:   (c) Adam Tejkl 2021
# Licence:     <your licence>
#-------------------------------------------------------------------------------
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import json

import os

print("Import of libraries done")

#%%

## specify model number
model_number = "006"
##

main_folder = "D:\DS_ML\modely\model_" + model_number
mosaics_folder = os.path.join(main_folder, "mosaics")
log = []

##num_skipped = 0
##for folder_name in ("NoRill", "Rill", "Sheet"):
##    folder_path = os.path.join(mosaics_folder, folder_name)
##    for fname in os.listdir(folder_path):
##        fpath = os.path.join(folder_path, fname)
##        try:
##            fobj = open(fpath, "rb")
##            is_jfif = tf.compat.as_bytes("JFIF") in fobj.peek(10)
##        finally:
##            fobj.close()
##
##        if not is_jfif:
##            num_skipped += 1
##            # Delete corrupted image
##            os.remove(fpath)
##
##print("Deleted %d images" % num_skipped)

# MTD_ds_20181112_01_0

mtd_name = 'MTD_' + 'ds' + "_" + '20181112_01_0' + ".json"
mtd_file = open(os.path.join(mosaics_folder, mtd_name), "r")
mtd_dict = json.load(mtd_file)
mtd_file.close()

h = mtd_dict["Height"]
w = mtd_dict["Width"]
pixel_number = mtd_dict["Pixel Number"]

#%%

image_size = (h * pixel_number, w * pixel_number)

##image_size = (5, 15)
batch_size = 32

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    mosaics_folder,
    validation_split=0.2,
    subset="training",
    seed=1337,
    image_size=image_size,
    batch_size=batch_size,
    labels="inferred",
    label_mode = "categorical",
    class_names = ['NoRill', 'Rill', 'Sheet'],
    color_mode="rgb",
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    mosaics_folder,
    validation_split=0.2,
    subset="validation",
    seed=1337,
    image_size=image_size,
    batch_size=batch_size,
    labels="inferred",
    label_mode = "categorical",
    class_names = ['NoRill', 'Rill', 'Sheet'],
    color_mode="rgb",
)

print("Training and validation done")

#%%

##import matplotlib.pyplot as plt
##
##plt.figure(figsize=(10, 10))
##for images, labels in train_ds.take(1):
##    for i in range(9):
##        ax = plt.subplot(3, 3, i + 1)
##        plt.imshow(images[i].numpy().astype("uint8"))
##        plt.title(int(labels[i]))
##        plt.axis("off")
##
##print("Plotting done")

#%%

##data_augmentation = keras.Sequential(
##    [
##        layers.experimental.preprocessing.RandomFlip("horizontal"),
##        layers.experimental.preprocessing.RandomRotation(0.1),
##    ]
##)

#%%

def make_model(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)
    # Image augmentation block

##    data_augmentation = keras.Sequential(
##    [
##        layers.experimental.preprocessing.RandomFlip("horizontal"),
##        layers.experimental.preprocessing.RandomRotation(0.1),
##    ]
##    )

##    x = data_augmentation(inputs)
    x = inputs

    # Entry block
    x = layers.experimental.preprocessing.Rescaling(1.0 / 255)(x)
    x = layers.Conv2D(32, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(64, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    for size in [128, 256, 512, 728]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(size, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    x = layers.SeparableConv2D(1024, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.GlobalAveragePooling2D()(x)
    if num_classes == 2:
        activation = "sigmoid"
        units = 1
    else:
        activation = "softmax"
        units = num_classes

    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(units, activation=activation)(x)
    return keras.Model(inputs, outputs)

##image_size = (180, 180)

topgis_model = make_model(input_shape=image_size + (3,), num_classes=3)  # pozor na num_classes musi sedet s poctem class_names
##topgis_model = make_model(input_shape=image_size + (1,), num_classes=5)
##keras.utils.plot_model(topgis_model, show_shapes=True)

epochs = 30
log.append(epochs)

callbacks = [keras.callbacks.ModelCheckpoint("save_at_{epoch}.h5"),]
backup_callback = keras.callbacks.experimental.BackupAndRestore(backup_dir= os.path.join(main_folder, "zaloha"))

topgis_model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="binary_crossentropy", metrics=["accuracy"],)

topgis_model.fit( train_ds, epochs=epochs, callbacks=[callbacks, backup_callback], validation_data=val_ds,)

topgis_model.save(main_folder)
print(log)

def analyse_mosaic(mosaic, image_size, input_model):
    img = keras.preprocessing.image.load_img(mosaic, target_size=image_size)
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create batch axis
    predictions = input_model.predict(img_array)
    score = predictions[0]
    print( "This image is %.2f percent NoRill and %.2f percent Rill." % (100 * (1 - score), 100 * score))

##topgis_model.fit(dataset, epochs= epochs, callbacks=[backup_callback])