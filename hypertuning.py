# %%
# !pip install kaggle

# %%
# !pip uninstall tensorflow -y
# !pip install tensorflow-macos

# %%
#!pip install -q -U keras-tuner

# %%
# imports 
import kagglehub
import pandas as pd 
import numpy as np
import os 
from PIL import Image
import PIL.Image
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import keras_tuner as kt 
from tensorflow.keras.models import Sequential

# %% [markdown]
# ### basic import and cleaning

# %%
# define and collect image paths and labels

# kaggle import 
path = kagglehub.dataset_download("syedalinaqvi/augmented-skin-conditions-image-dataset")
image_folder = os.path.join(path, "Skin_Conditions")
print("Image folder:", image_folder)
print("Subfolders (classes):", os.listdir(image_folder))

# create empty lists to hold image paths and labels
image_paths = []
labels = []

# iterate through each subfolder and collect image paths and labels
for label in sorted(os.listdir(image_folder)):
    label_folder = os.path.join(image_folder, label)

    if os.path.isdir(label_folder):
        for filename in os.listdir(label_folder):
            if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                image_paths.append(os.path.join(label_folder, filename))
                labels.append(label)

# %%
# create dataframe
df = pd.DataFrame({"image_path": image_paths, "labels": labels})

# remove rows w/ missing or invalid paths 
df = df[df["image_path"].apply(os.path.exists)]
df = df.drop_duplicates()

# numbers summary 
print(f"\nTotal images: {len(df)}")
print("Images per class:")
print(df["labels"].value_counts())

# %%
print(tf.__version__)

# %% [markdown]
# ### view images

# %%
acne_images = df[df["labels"] == "Acne"]["image_path"].tolist()
PIL.Image.open(acne_images[3])

# %%
carcinoma_images = df[df["labels"] == "Carcinoma"]["image_path"].tolist()
PIL.Image.open(carcinoma_images[3])

# %% [markdown]
# ## using: https://www.tensorflow.org/tutorials/load_data/images

# %% [markdown]
# ### load data using a keras utility
# these use the `tf.kera.utils.image_dataset_from_directory` utility. this basically helps generate a dataset from image files in a directory. 
# https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image_dataset_from_directory

# %% [markdown]
# #### create a dataset
# defining some parameters for the loader first.

# %%
AUTOTUNE = tf.data.AUTOTUNE
batch_size = 16
img_height = 90
img_width = 90

# added this one 
# num_classes = df["labels"].nunique()

# %% [markdown]
# use a validation split when developing model, using the standard 80% for training and 20% for validation. the `image_folder` variable is where all the images are stored and the image sizes are set to the parameters above.
# 
# **note:** above parameters were pulled directly from tensorflow tutorial, can tune later.

# %%
# train model 

train_ds = tf.keras.utils.image_dataset_from_directory(
  image_folder,
  validation_split=0.2,
  subset="training",
  seed=123, 
  image_size=(img_height, img_width),
  batch_size=batch_size)

# %%
# validate model

val_ds = tf.keras.utils.image_dataset_from_directory(
  image_folder,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

# %%
# verifying class names 

train_class_names = train_ds.class_names
print(train_class_names)

val_class_names = val_ds.class_names
print(val_class_names)

# %% [markdown]
# ### visualize the data 

# %% [markdown]
# visualize images from the set. this isn't super necessary but nice to have to see what the model is actually looking at. 

# %%
plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(train_class_names[labels[i]])
    plt.axis("off")

# %%
plt.figure(figsize=(10, 10))
for images, labels in val_ds.take(1):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(val_class_names[labels[i]])
    plt.axis("off")

# %% [markdown]
# the below code shows what the shape of the tensor is. this is a batch of 32 images in the shape `180x180x3` where the last dimension refers to color channels RGB. 
# 
# the `label_batch` is a tensor with the shape in the 2nd line, corresponding to labels of the 32 images. 

# %%
for image_batch, labels_batch in train_ds:
  print(image_batch.shape)
  print(labels_batch.shape)
  break

# %% [markdown]
# ### standardize data 

# %% [markdown]
# this is where we standardize the data. RGB channel values are in the `[0,255]` range, which isn't ideal for a neural network because we want to make our input values small. to standardize our values to be in `[0,1]` range, we're using `tf.kera.layers.Rescaling` and also note this is where the `(1./255)` comes from.

# %%
normalization_layer = tf.keras.layers.Rescaling(1./255)

# %% [markdown]
# there are 2 ways to use this layer. the first is to apply it to the dataset by calling `Dataset.map` which is what we did below. 
# 
# **note:** need to research if it's normal for it to be exactly 1.0 because the tutorial example was 0.96

# %%
# apply the normalization to the dataset of images while keeping labels unchanged
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))
# normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))

# # also normalizing the val_ds also
# normalized_val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))


# # creates an iterator over both the normalized datasets
# image_batch, labels_batch = next(iter(normalized_ds))
# image_batch_val, labels_batch_val = next(iter(normalized_val_ds))

# # select first image from that batch for both train and val
# first_image = image_batch[0]
# first_val_image = image_batch_val[0]

# # Notice the pixel values are now in `[0,1]`.
# print(np.min(first_image), np.max(first_image))
# print(np.min(first_val_image), np.max(first_val_image))

# %% [markdown]
# the 2nd way to use the layer is to include it inside our model definition to simplify deployment. 

# %% [markdown]
# ### configure dataset for performance

# %% [markdown]
# tutorial says to make sure to "use buffered prefetching so you can yield data from disk without having I/O become blocking". what does this mean? 
# * when you train a NN, the model runs computations AND the data pipeline is loading images, coding, and preparing them. so if data loading is too slow, the model waits for the next batch, wasting time 
# * I/O = input/output, like reading files from my disk 
# * blocking = the training process has to wait for the next batch to be ready
# * buffered prefethcing loads the next batch WHILE the current one is still being processed, speeding up the process

# %% [markdown]
# 2 methods to use when fetching data: 
# * `Dataset.cache` = keep image in memory after they're loaded off disk during first epoch, ensuring dataset doesn't become a bottleneck when training model 
# * `Dataset.prefetch` = overlap data preprocessing and mdoel execution while training

# %% [markdown]
# so the below code uses prefetch to have that overlap. 
# 
# autotune lets tf dynamically tune the best performance settings for my data pipeline.

# %%
# AUTOTUNE = tf.data.AUTOTUNE


# also reducing batch size so we can do hyperparameter tuning (less memory)
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# %% [markdown]
# ### hyperparameter tuning (https://www.tensorflow.org/tutorials/keras/keras_tuner)

# %% [markdown]
# still need to figure this section out 

# %%
# # aida - i think everything was normalized before
# # img_train = train_ds.map(lambda image, label: image)
# # img_val = val_ds.map(lambda image, label: image)

# # label_train = train_ds.map(lambda image, label: label)
# # label_val = val_ds.map(lambda image, label: label)


# train_ds = train_ds.batch(16)
# val_ds = val_ds.batch(16)

# %%
def model_builder(hp):
      num_classes = 6 # acne, carcinoma, etc. etc. 
      model = keras.Sequential()

      model.add(keras.Input(shape=(90, 90, 3)))
      model.add(layers.Rescaling(1./255))

      # add first convolutional layer
      model.add(layers.Conv2D(
            #adding filter 
            filters=hp.Int('conv_1_filter', min_value=16, max_value=64, step=16),
            # adding filter size or kernel size
            kernel_size=hp.Choice('conv_1_kernel', values = [3,5]), #3
            #activation function
            activation='relu',
            padding = 'same'))
      model.add(layers.MaxPooling2D())

      # add second convolutional layer
      model.add(layers.Conv2D(
            #adding filter 
            filters=hp.Int('conv_2_filter', min_value=32, max_value=128, step=32),
            # adding filter size or kernel size
            kernel_size=hp.Choice('conv_2_kernel', values = [3,5]), #3
            #activation function
            activation='relu',
            padding = 'same'))
      model.add(layers.MaxPooling2D())

      # # add third convolutional layer
      # model.add(layers.Conv2D(
      #       #adding filter 
      #       filters=hp.Int('conv_3_filter', min_value=64, max_value=256, step=64),
      #       # adding filter size or kernel size
      #       kernel_size=hp.Choice('conv_3_kernel', values = [3,5]), #3
      #       #activation function
      #       activation='relu',
      #       padding = 'same'))
      # model.add(layers.MaxPooling2D())


      # regularization 
      hp_dropout = hp.Float('dropout', min_value=0.0, max_value=0.5, step=0.1)
      model.add(layers.Dropout(hp_dropout))
    
      # add flatten layer
      model.add(layers.GlobalAveragePooling2D()) 

      hp_units = hp.Int('dense_units', min_value=32, max_value=256, step=32)
      model.add(keras.layers.Dense(units=hp_units, activation='relu'))

      # output layer
      model.add(layers.Dense(num_classes))

      # learning rate for optimizer
      hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

      model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy'])

      return model

# %%
tuner = kt.Hyperband(model_builder,
                     objective='val_accuracy',
                     max_epochs=10,
                     factor=3,
                     directory='my_dir',
                     project_name='intro_to_kt')

# %%
stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

# %%
tuner.search(train_ds, epochs=10, validation_data=val_ds, callbacks=[stop_early])

# Get the optimal hyperparameters
best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]

print(f"""
The hyperparameter search is complete. The optimal number of units in the first densely-connected
layer is {best_hps.get('units')} and the optimal learning rate for the optimizer
is {best_hps.get('learning_rate')}.
""")

# # %% [markdown]
# # ### train a model

# # %% [markdown]
# # * https://www.tensorflow.org/guide/keras/sequential_model 
# # * sequential model is appropriate for a plain stack of layers where each layer has exactly 1 input 1 output --> this might not be what we want
# # * the below code shows the model with convolution blocks, max pooling layer, and a fully connected layer on top of it that is activated by a ReLU activation function
# # * however, this model hasn't been tuned in any way 

# # %%
# num_classes = 6

# model = tf.keras.Sequential([
#   tf.keras.layers.Rescaling(1./255),
#   tf.keras.layers.Conv2D(32, 3, activation='relu'),
#   tf.keras.layers.MaxPooling2D(),
#   tf.keras.layers.Conv2D(32, 3, activation='relu'),
#   tf.keras.layers.MaxPooling2D(),
#   tf.keras.layers.Conv2D(32, 3, activation='relu'),
#   tf.keras.layers.MaxPooling2D(),
#   tf.keras.layers.Flatten(),
#   tf.keras.layers.Dense(128, activation='relu'),
#   tf.keras.layers.Dense(num_classes)
# ])

# # %%
# model.compile(
#   optimizer='adam',
#   loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#   metrics=['accuracy'])

# # %%
# model.fit(
#   train_ds,
#   validation_data=val_ds,
#   epochs=3
# )

# # %% [markdown]
# # ### IGNORE MUCH OF ABOVE 
# # ### image classification tutorial https://www.tensorflow.org/tutorials/images/classification 

# # %% [markdown]
# # * post dataset configuration and standardization, we can create a basic Keras model

# # %%
# num_classes = len(train_class_names)

# model = Sequential([
#   layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
#   layers.Conv2D(16, 3, padding='same', activation='relu'),
#   layers.MaxPooling2D(),
#   layers.Conv2D(32, 3, padding='same', activation='relu'),
#   layers.MaxPooling2D(),
#   layers.Conv2D(64, 3, padding='same', activation='relu'),
#   layers.MaxPooling2D(),
#   layers.Flatten(),
#   layers.Dense(128, activation='relu'),
#   layers.Dense(num_classes)
# ])

# # %% [markdown]
# # Compile the model next. 
# # 
# # We choose the `tf.keras.optimizers.Adam` optimizer and `tf.keras.losses.SparseCategoricalCrossentropy` loss function. To view training and validation accuracy for each training epoch, pass the `metrics` argument to `Model.compile`.

# # %%
# model.compile(optimizer='adam',
#               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#               metrics=['accuracy'])

# # %% [markdown]
# # Model summary 
# # 
# # View all layers of network using Kera `Model.summary` method. 
# # 
# # **research:** how do i actually make sense of this data though? 

# # %%
# model.summary()

# # %% [markdown]
# # ### train the model

# # %% [markdown]
# # train the model for 10 epochs with the Keras `Model.fit` method

# # %%
# epochs=10
# history = model.fit(
#   train_ds,
#   validation_data=val_ds,
#   epochs=epochs
# )

# # %% [markdown]
# # Visualize training results. 
# # 
# # Create plots of loss and accuracy on training and validation sets. 

# # %%
# acc = history.history['accuracy']
# val_acc = history.history['val_accuracy']

# loss = history.history['loss']
# val_loss = history.history['val_loss']

# epochs_range = range(epochs)

# plt.figure(figsize=(8, 8))
# plt.subplot(1, 2, 1)
# plt.plot(epochs_range, acc, label='Training Accuracy')
# plt.plot(epochs_range, val_acc, label='Validation Accuracy')
# plt.legend(loc='lower right')
# plt.title('Training and Validation Accuracy')

# plt.subplot(1, 2, 2)
# plt.plot(epochs_range, loss, label='Training Loss')
# plt.plot(epochs_range, val_loss, label='Validation Loss')
# plt.legend(loc='upper right')
# plt.title('Training and Validation Loss')
# plt.show()

# # %% [markdown]
# # ### Overfitting 
# # 
# # We see trainng accuracy increasing linearly over time, but validation accuracy hovers around 60% and doesn't improve. A difference in accuracy b/t training and validation accuracy is a sign of overfitting. 

# # %% [markdown]
# # One method we can use is data augmentation, which generates additional training data from existing samples by augementing them using random transformations that yield believeable looking images. This is in response to the idea that overfitting generally cocurs when there are a small number of training examples. 

# # %%
# data_augmentation = keras.Sequential(
#   [
#     layers.RandomFlip("horizontal",
#                       input_shape=(img_height,
#                                   img_width,
#                                   3)),
#     layers.RandomRotation(0.1),
#     layers.RandomZoom(0.1),
#   ]
# )

# # %%
# plt.figure(figsize=(10, 10))
# for images, _ in train_ds.take(1):
#   for i in range(9):
#     augmented_images = data_augmentation(images)
#     ax = plt.subplot(3, 3, i + 1)
#     plt.imshow(augmented_images[0].numpy().astype("uint8"))
#     plt.axis("off")

# # %% [markdown]
# # Introduce dropout regularization - need to look into this one more 

# # %%
# model = Sequential([
#   data_augmentation,
#   layers.Rescaling(1./255),
#   layers.Conv2D(16, 3, padding='same', activation='relu'),
#   layers.MaxPooling2D(),
#   layers.Conv2D(32, 3, padding='same', activation='relu'),
#   layers.MaxPooling2D(),
#   layers.Conv2D(64, 3, padding='same', activation='relu'),
#   layers.MaxPooling2D(),
#   layers.Dropout(0.2),
#   layers.Flatten(),
#   layers.Dense(128, activation='relu'),
#   layers.Dense(num_classes, name="outputs")
# ])

# # %%
# model.compile(optimizer='adam',
#               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#               metrics=['accuracy'])

# model.summary()

# epochs = 15
# history = model.fit(
#   train_ds,
#   validation_data=val_ds,
#   epochs=epochs
# )

# # %%
# acc = history.history['accuracy']
# val_acc = history.history['val_accuracy']

# loss = history.history['loss']
# val_loss = history.history['val_loss']

# epochs_range = range(epochs)

# plt.figure(figsize=(8, 8))
# plt.subplot(1, 2, 1)
# plt.plot(epochs_range, acc, label='Training Accuracy')
# plt.plot(epochs_range, val_acc, label='Validation Accuracy')
# plt.legend(loc='lower right')
# plt.title('Training and Validation Accuracy')

# plt.subplot(1, 2, 2)
# plt.plot(epochs_range, loss, label='Training Loss')
# plt.plot(epochs_range, val_loss, label='Validation Loss')
# plt.legend(loc='upper right')
# plt.title('Training and Validation Loss')
# plt.show()

# # %% [markdown]
# # results are better.

# # %% [markdown]
# # ### predict on new data

# # %% [markdown]
# # ### below is not from tensorflow tutorial

# # %% [markdown]
# # ### model evaluation

# # %%
# val_loss, val_accuracy = model.evaluate(val_ds)
# print(f"Final Validation Loss: {val_loss:.4f}")
# print(f"Final Validation Accuracy: {val_accuracy:.4f}")

# # %%
# # labeling and predictions
# y_true = np.concatenate([y.numpy() for x, y in val_ds], axis=0)
# y_pred_logits = np.concatenate([model(x, training=False).numpy() for x, y in val_ds], axis=0)
# y_pred = np.argmax(y_pred_logits, axis=1)

# # classification report
# print(classification_report(y_true, y_pred, target_names=df["labels"].unique()))

# # confusion matrix
# cm = confusion_matrix(y_true, y_pred)
# print("Confusion Matrix:\n", cm)

# # %%




# %%
