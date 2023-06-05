import numpy as np
import os
import shutil
import random
import tensorflow as tf

# Function to remove folders with less than 2 elements
def remove_useless_folder(root):
    removedel = 0
    for folder in os.listdir(root):
        if len(os.listdir(os.path.join(root, folder))) < 2:
            shutil.rmtree(os.path.join(root,folder))
            removedel += 1
    print(f"Removed {removedel} elements")

# Function to split the dataset into train and validation sets
def train_validation_splitter(path, percentage):
    # Create TRAIN and VALIDATION folders
    parent_folder = os.path.dirname(path)
    train_path = os.path.join(parent_folder, "TRAIN")
    os.makedirs(train_path, exist_ok=True)
    validation_path = os.path.join(parent_folder, "VALIDATION")
    os.makedirs(validation_path, exist_ok=True)
    
    counter = 0
    for folder in os.listdir(path):
        counter += 1
        if counter % 500 == 0:
            print((counter / (len(os.listdir(path) * 20)), "%"))
        r = random.randint(1, 100)
        if r < percentage * 10:
            shutil.move(os.path.join(path, folder), validation_path)
        else:
            shutil.move(os.path.join(path, folder), train_path)
    print("DATASET SPLITTED")

# Function to augment the dataset with more "positive" samples
def dataset_enrich(dataset_dir):
    counter = 0
    for DIR in os.listdir(dataset_dir):
        DIR_PATH = os.path.join(dataset_dir, DIR)
        if len(os.listdir(DIR_PATH)) < 2:
            counter += 1
            if counter % 100 == 0:
                print((f"{counter} images augmented"))
            img_path = os.path.join(DIR_PATH, str(os.listdir(DIR_PATH)[0]))
            
            # Function to perform data augmentation
            def data_aug(img):
                data = []
                for i in range(5):
                    img = tf.image.stateless_random_brightness(img, max_delta=0.09, seed=(1,5))
                    img = tf.image.stateless_random_contrast(img, lower=0.6, upper=1, seed=(1,3))
                    img = tf.image.stateless_random_flip_left_right(img, seed=(np.random.randint(100),np.random.randint(100)))
                    img = tf.image.stateless_random_jpeg_quality(img, min_jpeg_quality=90, max_jpeg_quality=100, seed=(np.random.randint(100),np.random.randint(100)))
                    img = tf.image.stateless_random_saturation(img, lower=0.9,upper=1, seed=(np.random.randint(100),np.random.randint(100)))
                    data.append(img)
                return data
            
            img = cv2.imread(img_path)
            augmented_images = data_aug(img)
            
            for i, image in enumerate(augmented_images):
                cv2.imwrite(os.path.join(DIR_PATH, f'{DIR}{i}.jpg'), image.numpy())
    print("DATASET ENRICHED")
