"""
*****************************************************************************************
This model was built by: Djaid Walid

__________________________________________________________________________________________________
                                   Contacts                                                      |
__________________________________________________________________________________________________
Github     | https://github.com/waliddj                                                          |
Linkedin   | www.linkedin.com/in/walid-djaid-375777229                                           |
Instagram  | https://www.instagram.com/d.w.science?igsh=MWlnMmNpOTM2OW0xaA%3D%3D&utm_source=qr   |
__________________________________________________________________________________________________

Dataset used to train this model is : Multi Cancer Dataset.
Link to the dataset: https://www.kaggle.com/datasets/obulisainaren/multi-cancer/data
Obuli Sai Naren. (2022). Multi Cancer Dataset [Data set]. Kaggle. https://doi.org/10.34740/KAGGLE/DSV/3415848
*****************************************************************************************
"""

import tensorflow as tf
from tensorflow.keras import Sequential, layers
from tensorflow.keras.layers import Flatten,Dense,MaxPool2D,Conv2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import numpy as np
import pathlib
import os
import kagglehub
from sklearn.model_selection import train_test_split
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Download the dataset from kiggle
path = kagglehub.dataset_download("obulisainaren/multi-cancer")
print("Path to dataset files:", path)

data_dir = 'C:/Users/walid/.cache/kagglehub/datasets/obulisainaren/multi-cancer/versions/3/Multi Cancer/Multi Cancer/Kidney Cancer/'





# ***************** Data Preprocessing **********************************
IMAGE_SIZE = (224,224)
BATCH_SIZE = 32

# Split the train and test data
train_dir = 'C:/Users/walid/.cache/kagglehub/datasets/obulisainaren/multi-cancer/versions/3/Multi Cancer/Multi Cancer/Kidney Cancer/train'
test_dir = 'C:/Users/walid/.cache/kagglehub/datasets/obulisainaren/multi-cancer/versions/3/Multi Cancer/Multi Cancer/Kidney Cancer/test'

# get the class names from the train directory (You can also get the class names from the test directory)
data_path = pathlib.Path(train_dir)
class_names = np.array(sorted([item.name for item in data_path.glob('*')]))
print(class_names)

"""
# Visualize the data 
def view_random_image(target_dir, target_class):
  target_foler = target_dir + target_class
  random_image = random.sample(os.listdir(target_foler),1)
  # Read in the image and plot it using matplotlib
  img = mpimg.imread(target_foler + "/" + random_image[0])
  plt.imshow(img)
  plt.title(f"{target_class}")
  plt.axis("off")
  plt.show()
  print(img.shape)
  return img

img = view_random_image(data_dir,
                        target_class=random.choice(class_names))
"""


# Normalize the data using ImageDataGenerator
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMAGE_SIZE, #resize the images
    batch_size=BATCH_SIZE,
    class_mode='binary'
)
test_data = test_datagen.flow_from_directory(
    test_dir,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary'
)



# ********************** Build the CNN model **********************

model = Sequential([
    Conv2D(filters=10, kernel_size=3, input_shape=(224,224,3), activation='relu'),
    Conv2D(10,3, activation='relu'),
    MaxPool2D(),
    Conv2D(10,3,activation='relu'),
    MaxPool2D(),
    Flatten(),
    Dense(1, activation='sigmoid') #Because we are dealing with binary classification
])

model.compile(
    loss = tf.keras.losses.BinaryCrossentropy(),
    optimizer='adam',
    metrics=['accuracy']
)

history = model.fit(train_data,
                epochs=6,
                steps_per_epoch=len(train_data),
                validation_data=test_data,
                validation_steps=len(test_data))

# ***************** Model evaluation ******************************
### Model evaluation
#### Loss/Accuracy

# Loss/Accuracy
pd.DataFrame(history.history).plot()
plt.xlabel("epochs")
plt.ylabel('loss')
plt.show()

"""
|Metrics|Accuracy|Loss|
|-------|-----|----|
|Train data| 100%| 1.0018e-04|
|Test data| 99.9% | 0.0018|
> **Note:** in the 3rd epoch the model reached a 100% accuracy rate. 
"""
### plot random im

# ************** Save the model ******************************
model.save('C:/Users/walid/Desktop/Kidney_cancer.keras')

# ****************** Plot random xRay images with their predication ********************

md = tf.keras.models.load_model('C:/Users/walid/Desktop/Kidney_cancer.keras')

import random
import pathlib

# Create a function for plotting a random image along with its prediction
def plot_random_prediction(model, data_dir, class_names, img_size=(224, 224)):
    """
    data_dir: Path or str to the test folder that contains one subfolder per class
    class_names: np.array or list of class names in the same order used by the model
    img_size: target size used to train the model
    """
    data_dir = pathlib.Path(data_dir)
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    all_images = [p for p in data_dir.rglob("*") if p.suffix.lower() in exts and p.parent != data_dir]
    if not all_images:
        raise ValueError("No images found under data_dir. Check your path and folder structure.")

    #img_path = random.choice(all_images)
    img_path = 'C:/Users/walid/Desktop/Kedny imges/kt_2_f2.png'

    #true_class = img_path.parent.name
    true_class = 'kidney_tumor'
    # Map true class name to index according to class_names
    class_names = np.array(class_names).astype(str)
    try:
        true_idx = np.where(class_names == true_class)[0][0]
    except IndexError:
        raise ValueError(f"True class '{true_class}' not found in class_names {class_names}.")

    # Load and preprocess image
    img = tf.keras.utils.load_img(img_path, target_size=img_size)
    img_arr = tf.keras.utils.img_to_array(img) / 255.0
    x = np.expand_dims(img_arr, axis=0)

    # Predict
    prob = float(model.predict(x, verbose=0).squeeze())  # sigmoid output in [0,1]
    pred_idx = int(prob >= 0.5)
    pred_class = class_names[pred_idx]

    # Confidence for the predicted class
    confidence = prob if pred_idx == 1 else (1.0 - prob)

    # Plot
    plt.figure()
    plt.imshow(img_arr)
    plt.axis("off")
    correct = (pred_idx == true_idx)
    title_color = "green" if correct else "red"
    plt.title(f"Pred: {pred_class} ({confidence*100:.1f}%) | True: {true_class}", color=title_color)
    plt.show()


plot_random_prediction(md, train_dir, class_names, img_size=IMAGE_SIZE)


