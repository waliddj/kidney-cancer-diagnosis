# kidney-cancer-diagnosis
A convolutional neural network (CNN) binary-classification model for kidney cancer diagnosis.
# Dataset
[Multi Cancer Dataset](https://www.kaggle.com/datasets/obulisainaren/multi-cancer/data)

Citation: Obuli Sai Naren. (2022). Multi Cancer Dataset [Data set]. Kaggle. https://doi.org/10.34740/KAGGLE/DSV/3415848

**DataSet structure:**

| Cancer Type                  | Classes | Images  |
|------------------------------|----------|---------|
| Acute Lymphoblastic Leukemia | 4        | 20,000  |
| Brain Cancer                 | 3        | 15,000  |
| Breast Cancer                | 2        | 10,000  |
| Cervical Cancer              | 5        | 25,000  |
| Kidney Cancer                | 2        | 10,000  |
| Lung and Colon Cancer        | 5        | 25,000  |
| Lymphoma                     | 3        | 15,000  |
| Oral Cancer                  | 2        | 10,000  |

I selected for this model only the Kidney Cancer directory.

As a first step in the modeling process, the data was split after being downloaded into two main directories ```train``` and ```test``` following this structure:
```
\train
  |
  |__\kidney_normal
      |
      |__4000 images
  |
  |__\kidney_tumor
      |
      |__4000 images

\test
  |
  |__\kidney_normal
      |
      |__1000 images
  |
  |__\kidney_tumor
      |  
      |__1000 images
```

```python
import os
import shutil
import random
from sklearn.model_selection import train_test_split

# Paths to your original dataset
base_dir = 'C:/Users/walid/.cache/kagglehub/datasets/obulisainaren/multi-cancer/versions/3/Multi Cancer/Multi Cancer/Kidney Cancer/'
categories = ['kidney_normal', 'kidney_tumor']

# Paths for train/test splits
train_dir = os.path.join(base_dir, 'train')
test_dir = os.path.join(base_dir, 'test')

# Create directories if they don't exist
for split_dir in [train_dir, test_dir]:
    for category in categories:
        os.makedirs(os.path.join(split_dir, category), exist_ok=True)

# Split ratio
test_ratio = 0.2  # 20% for test data

for category in categories:
    folder = os.path.join(base_dir, category)
    images = os.listdir(folder)
    images = [img for img in images if img.lower().endswith(('.png', '.jpg', '.jpeg'))]

    # Train-test split
    train_images, test_images = train_test_split(images, test_size=test_ratio, random_state=42)

    # Copy images
    for img in train_images:
        shutil.move(os.path.join(folder, img), os.path.join(train_dir, category, img))
    for img in test_images:
        shutil.move(os.path.join(folder, img), os.path.join(test_dir, category, img))

print("Dataset successfully split into train and test folders.")
```

# Code architecture
```python
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
```
# Download the data from [Kaggel](https://www.kaggle.com/datasets/obulisainaren/multi-cancer/data)
Downloading the data using ```kaggelhub.dataset_download(()``` function

```python
path = kagglehub.dataset_download("obulisainaren/multi-cancer")
print("Path to dataset files:", path)
```

# Data preprocessing
## Set the Image and batch sizes
```python
IMAGE_SIZE = (224,224)
BATCH_SIZE = 32
```
## Split the train and test data directories
```python
train_dir = 'C:/Users/walid/.cache/kagglehub/datasets/obulisainaren/multi-cancer/versions/3/Multi Cancer/Multi Cancer/Kidney Cancer/train'
test_dir = 'C:/Users/walid/.cache/kagglehub/datasets/obulisainaren/multi-cancer/versions/3/Multi Cancer/Multi Cancer/Kidney Cancer/test'
```
## get the class names from the train directory (You can also get the class names from the test directory)
```python
data_path = pathlib.Path(train_dir)
class_names = np.array(sorted([item.name for item in data_path.glob('*')]))
print(class_names)
```
## Visualize the data 
```python

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
```
<img width="420" height="320" alt="Figure_1" src="https://github.com/user-attachments/assets/ebcc7ae4-9371-44b9-a2f1-138b927decfb" />
<img width="420" height="320" alt="Figure_2" src="https://github.com/user-attachments/assets/746bf9e8-29bf-4175-be81-f1e397bc8671" />
<img width="420" height="320" alt="Figure_3" src="https://github.com/user-attachments/assets/9b5513e0-3928-4b2c-a224-ed33c0790b22" />
<img width="420" height="320" alt="Figure_4" src="https://github.com/user-attachments/assets/7718aca4-7beb-488d-81fb-20e6bb080aed" />

## Normalize the data using ```ImageDataGenerator```
```python

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
```

# Build the CNN model
---
## Model architecture:
- ```Conv2D``` layer *(```input``` layer)* with 10 ```filters``` and ```kernel_size``` = 3, ```input_shape``` = ```(224,224,3)```, and a ```ReLU``` activation method.
- ```Conv2D``` layer with same parameters as ```input``` layer but without the ```input_shape``` value.
- ```MaxPool2D``` layer follwed by another ```Conv2D``` layer, again a ```MaxPool2D``` layer.
- ```Flatten``` layer followed by a ```Dense``` layer *(```output``` layer)* , with a ```sigmoid``` activation method.
- ```epochs``` = ```6```.
- Optimizer: ```Adam()```.
- ```learning_date``` = ```0.001``` *(default value)*
---
## Build the model
```pythpn
model = Sequential([
    Conv2D(filters=10, kernel_size=3, input_shape=(224,224,3), activation='relu'),
    Conv2D(10,3, activation='relu'),
    MaxPool2D(),
    Conv2D(10,3,activation='relu'),
    MaxPool2D(),
    Flatten(),
    Dense(1, activation='sigmoid') #Because we are dealing with binary classification
])
```
## Compile the model
```python
model.compile(
    loss = tf.keras.losses.BinaryCrossentropy(),
    optimizer='adam',
    metrics=['accuracy']
)
```
## Fit the model to the data while keep tracking the model's history for evaluation
```python
history = model.fit(train_data,
                epochs=6,
                steps_per_epoch=len(train_data),
                validation_data=test_data,
                validation_steps=len(test_data))
```

# Model evaluation
## Loss/Accuracy
```python
pd.DataFrame(history.history).plot()
plt.xlabel("epochs")
plt.ylabel('loss')
plt.show()
```

|Metrics|Accuracy|Loss|
|-------|-----|----|
|Train data| 100%| 1.0018e-04|
|Test data| 99.9% | 0.0018|

> **Note:** in the 3rd epoch the model reached a 100% accuracy rate. 

## Plot random predictions from dataset:

<img width="420" height="320" alt="pred_1" src="https://github.com/user-attachments/assets/03aed626-0bb6-4f69-a552-cca95ed9a448" />
<img width="420" height="320" alt="pred_2" src="https://github.com/user-attachments/assets/a66b3071-e460-4b1b-801b-3c97d4263846" />
<img width="420" height="320" alt="pred_3" src="https://github.com/user-attachments/assets/3cfcf221-5fb7-441a-850a-f32fb0155038" />
<img width="420" height="320" alt="pred_4" src="https://github.com/user-attachments/assets/9da19dca-b4c7-4267-81cd-de8af83417bb" />

## Medical Scan kidney cancer prediction:

Images from [Medical Scan Classification Dataset](https://www.kaggle.com/datasets/arjunbasandrai/medical-scan-classification-dataset)

<img width="420" height="320" alt="kt" src="https://github.com/user-attachments/assets/8c8a1685-071d-4d5a-99fe-e6a87999831b" />
<img width="420" height="320" alt="pred_5" src="https://github.com/user-attachments/assets/a453d8db-62cf-4d22-a336-beba9c4d7b83" />

Images from [Google](https://www.google.com/search?sca_esv=92bdb361676e0929&rlz=1C1SQJL_frES1021ES1021&sxsrf=AE3TifOeUygsca-gGXsAAx3rVM1loGDC8A:1762336214487&udm=2&fbs=AIIjpHxU7SXXniUZfeShr2fp4giZ1Y6MJ25_tmWITc7uy4KIeh7Rd2Okf7KLGPdxBrNMaIm3bjKczkW42t0bjubXjlXqwvu7QbcZ7V2iVSmFSoiz8Tc1Ch_WXeS7BT_VuVdajZxCDmNl6KDQErOZ_7pCX_vO5TihHSiszq7vifOl0LMHwYRW2AxjriArJgIQOTGxABpgmx7kw0yFJrdqwxfmzI__CGDc5Q&q=Medical+Scan+kidney+cancer&sa=X&ved=2ahUKEwjqgqiA3tqQAxVV8DQHHSqnO2oQtKgLegQIFRAB&biw=1536&bih=777&dpr=2.5)


<img width="420" height="320" alt="kt_2" src="https://github.com/user-attachments/assets/395a89e3-785e-4227-8185-542d5dd1f9bc" />
<img width="420" height="320" alt="pred_kt2_f2" src="https://github.com/user-attachments/assets/154dfd69-e1ed-46d2-a09f-ce5ab96b6a11" />


<img width="420" height="320" alt="kt_2_f" src="https://github.com/user-attachments/assets/c9abb39a-cf46-45d9-adec-f4e6db62aceb" />
<img width="420" height="320" alt="pred_kt_2" src="https://github.com/user-attachments/assets/4bc88ebf-710e-4537-95de-80b181190e43" />



