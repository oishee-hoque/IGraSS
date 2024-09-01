# For data preprocessing
import tensorflow as tf
from tensorflow import image as tf_image
from tensorflow import data as tf_data
from tensorflow import io as tf_io
import numpy as np

IMAGE_SIZE = 512
BATCH_SIZE = 4

import keras
class Augment(keras.layers.Layer):
    def __init__(self, seed=42):
        super().__init__()
        # both use the same seed, so they'll make the same random changes.
        self.augment_inputs = keras.Sequential([
                  keras.layers.RandomFlip("horizontal_and_vertical",seed=42),
                  keras.layers.RandomRotation(0.1,seed=42),
                  keras.layers.RandomZoom(0.2, seed=42),  # Zoom in/out 20%
                  keras.layers.RandomBrightness(0.05,value_range=(0, 1), seed=42),  # Adjust contrast by 10%
                  keras.layers.RandomContrast(0.1, seed=42),
                  # keras.layers.RandomTranslation(0.05,0.05,seed=42)
                ])

        
        # Use the same seed to keep inputs and labels transformations consistent
        self.augment_labels = keras.Sequential([
                  keras.layers.RandomFlip("horizontal_and_vertical",seed=42),
                  keras.layers.RandomRotation(0.1,seed=42),
                  keras.layers.RandomZoom(0.2, seed=42),  # Zoom in/out 20%
                  # keras.layers.RandomTranslation(0.05,0.05,seed=42)
                ])

    def call(self, inputs, labels):
        inputs = self.augment_inputs(inputs)
        labels = self.augment_labels(labels)
        return inputs,labels
    
class DataProcessor():
    def __init__(self,IMAGE_SIZE=512,BATCH_SIZE=4,NUM_CLASSES=1):
        self.IMAGE_SIZE = IMAGE_SIZE
        self.BATCH_SIZE = BATCH_SIZE
        self.NUM_CLASSES = NUM_CLASSES
        
    def _load_npy(self,path):
        # Load the .npy file and cast to float32
        npy_data = np.load(path.decode('utf-8'))
        return npy_data.astype(np.uint8)
        
    def read_image(self,image_path, mask=False):
    #     image = tf_io.read_file(image_path)
        if mask:
            mask_data = tf.numpy_function(self._load_npy, [image_path], tf.uint8)
            mask_data.set_shape([None, None, 1])
            image = tf_image.resize(mask_data, [self.IMAGE_SIZE, self.IMAGE_SIZE])
        else:
            image = tf_io.read_file(image_path)
            image = tf_image.decode_png(image, channels=3)
            image.set_shape([None, None, 3])
            image = tf_image.resize(images=image, size=[self.IMAGE_SIZE, self.IMAGE_SIZE])
        return image

    def load_data(self,image_list, mask_list):
        image = self.read_image(image_list)
        mask = self.read_image(mask_list, mask=True)
        return image, mask


    def data_generator(self,image_list, mask_list):
        print("came here")
        dataset = tf_data.Dataset.from_tensor_slices((image_list, mask_list))
        dataset = dataset.map(self.load_data, num_parallel_calls=tf_data.AUTOTUNE)
        dataset = (dataset.batch(self.BATCH_SIZE,                         drop_remainder=True).map(Augment()).prefetch(tf.data.AUTOTUNE))
        return dataset
