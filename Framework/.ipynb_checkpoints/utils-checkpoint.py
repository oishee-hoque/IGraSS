##All Libraries
import sys
# adding Folder_2/subfolder to the system path
sys.path.insert(0, '/scratch/gza5dr/Current_Canal_Experiments/Canal_Detection_Experiments/DeepLab3/KFold_Experiments/utils')

import keras
from keras import layers
from keras import ops

import os
import numpy as np
from glob import glob
import cv2
from scipy.io import loadmat
import matplotlib.pyplot as plt
import json

# For data preprocessing
import tensorflow as tf
from tensorflow import image as tf_image
from tensorflow import data as tf_data
from tensorflow import io as tf_io

#parameters
from keras.layers import Dropout
from keras import regularizers
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
import time


from deeplabV3Model import *
from train_test_Split import *
from data_process import *
random.seed(42)

DATA_DIR = ''
train_images = ''
train_masks = ''

class DatasetHandler():
    
    def __init__(self,DATA_DIR="/scratch/gza5dr/NHDShape/procssed_data/",BATCH_SIZE = 4,NUM_CLASSES = 1,IMAGE_SIZE=512,
                 img_folder_name='image_patches',mask_folder_name='mask_patches'):
        self.DATA_DIR = DATA_DIR
        self.train_images = sorted(glob(os.path.join(self.DATA_DIR, f"{img_folder_name}/*")))
        self.train_masks = sorted(glob(os.path.join(self.DATA_DIR, f"{mask_folder_name}/*")))
        self.BATCH_SIZE = BATCH_SIZE
        self.NUM_CLASSES = NUM_CLASSES
        self.IMAGE_SIZE= IMAGE_SIZE
        self.train_test_splits = []

    def prep_data_splits(self,regions = ['r1','r2', 'r3']):

        train_test_splits = []
        test_region_list = []
        

        for r in regions:
            test_region = [r]
            train_region = [region for region in regions if region not in test_region]
            print(len(self.train_images))
            test_idx = [idx for idx,filename in enumerate(self.train_images) if any(test in filename for test in test_region)]
            train_idx = [idx for idx,filename in enumerate(self.train_images) if any(train in filename for train in train_region)]
            self.train_test_splits.append((train_idx, test_idx))
    
    
    def prep_data_splits_1(self,p=.1):

        train_test_splits = []
        test_region_list = []
        l = len(self.train_images)
        c = int(len(self.train_images)*p)
        train_idx = [idx for idx,filename in enumerate(self.train_images[:-c])]
        test_idx = [idx for idx,filename in enumerate(self.train_images[-c:])]
        self.train_test_splits.append((train_idx, test_idx))

    def save_json_1(self,regions,fpath='/scratch/gza5dr/NHDShape/procssed_data/data_v1_json',n=3):
        data = {}
        for i in range(n):
            data[f'Set_{i}'] = {
                "region": regions[i],
                f"train_set_{i}": self.train_test_splits[i][0],
                f"test_set_{i}": self.train_test_splits[i][1]
            }
        # Saving the data to a JSON file
        with open(fpath, 'w') as json_file:
            json.dump(data, json_file)
            
    def save_json(self,fpath='/scratch/gza5dr/NHDShape/procssed_data/data_v1_json',n=1):
        data = {}
        for i in range(n):
            data[f'Set_{i}'] = {
                f"train_set_{i}": self.train_test_splits[i][0],
                f"test_set_{i}": self.train_test_splits[i][1]
            }
        # Saving the data to a JSON file
        with open(fpath, 'w') as json_file:
            json.dump(data, json_file)

    def load_json(self,fpath):
        #getting the preloaded file names
        with open(fpath, 'r') as json_file:
            data = json.load(json_file)
        return data

    def get_train_test(self,data,d_set=0):
        cur_train_set = data[f'Set_{d_set}'][f'train_set_{d_set}']
        cur_test_set = data[f'Set_{d_set}'][f'test_set_{d_set}']

        train_image_names, train_mask_names = image_mask_list(cur_train_set,self.train_images,self.train_masks)
        test_image_names, test_mask_names = image_mask_list(cur_test_set,self.train_images,self.train_masks)

        return  train_image_names, train_mask_names, test_image_names, test_mask_names

    def prep_data(self,data,n_splits=2,d_set=0):

        ##Preparing the dataset
        dp = DataProcessor(IMAGE_SIZE=self.IMAGE_SIZE,BATCH_SIZE= self.BATCH_SIZE,NUM_CLASSES=self.NUM_CLASSES)
        main_train_dataset = []
        main_val_dataset = []
        main_test_dataset = []

        train_dataset = []
        val_dataset = []
        test_dataset = []

        kfold_train_images,kfold_train_masks, test_image_names, test_mask_names= self.get_train_test(data,d_set=d_set)
        train_index, val_index = Kfold_splits(kfold_train_images,n_splits)
        test_dataset.append(dp.data_generator(test_image_names, test_mask_names,aug=False))
        train_dataset.append(dp.data_generator(kfold_train_images, kfold_train_masks))
        print(len(kfold_train_images))
        print(len(test_image_names))

        # for j in range(n_splits):
        #     train_image_names, train_mask_names = image_mask_list(train_index[j],kfold_train_images,kfold_train_masks)
        #     print(len(train_image_names))
        #     val_image_names, val_mask_names = image_mask_list(val_index[j],kfold_train_images,kfold_train_masks)
        #     train_dataset.append(dp.data_generator(train_image_names, train_mask_names))
        #     val_dataset.append(dp.data_generator(val_image_names, val_mask_names))


        # main_train_dataset.append(train_dataset)
        # main_val_dataset.append(val_dataset)
        # main_test_dataset.append(test_dataset)  


        return train_dataset,val_dataset,test_dataset
    
    def check_dataset(self,data_check,n=4):
        fig,axs = plt.subplots(n,2,figsize=(10, 10))
        for images,labels in data_check:
            for i in range(n):
                image = images[i].numpy().astype("uint8")
                masks = labels[i]
                axs[i][0].imshow(image)
                axs[i][1].imshow(masks)


#####Metrics                
                
def data_prep(y_true, y_pred):
    y_pred = tf.cast(y_pred > 0.3, tf.float32)
    tp = tf.reduce_sum(y_true * y_pred, axis=[0, 1, 2])
    fp = tf.reduce_sum((1 - y_true) * y_pred, axis=[0, 1, 2])
    fn = tf.reduce_sum(y_true * (1 - y_pred), axis=[0, 1, 2])
    return tp, fp, fn



def iou(y_true, y_pred, smooth=1e-6):
    # Assuming y_pred and y_true are of shape (batch_size, height, width, num_classes)
    # and contain binary values (e.g., 0 or 1).
    y_pred = tf.cast(y_pred > 0.3, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
    union = tf.reduce_sum(y_true, axis=[1, 2, 3]) + tf.reduce_sum(y_pred, axis=[1, 2, 3]) - intersection
    # (intersection + smooth) / (union + smooth)
    return tf.reduce_mean(intersection / union)

def precision(y_true, y_pred):
    tp, fp, fn = data_prep(y_true, y_pred)
    precision = tp / (tp + fp + 1e-5)
    return precision

def recall(y_true, y_pred):
    tp, fp, fn = data_prep(y_true, y_pred)
    recall = tp / (tp + fn + 1e-5)
    return recall

    
def f1_score(y_true, y_pred):
    p =  precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    f1_scores = 2 * ((p * r) / (p + r + 1e-5)) 
    return f1_scores

def dice_coef(y_true, y_pred,smooth=1e-6):
    # Ensure the prediction is binary (0 or 1)
    y_pred = tf.cast(y_pred > 0.3, tf.float32)
    # Flatten the tensors to make sure operations are conducted on a 1D basis
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    # Calculate intersection and union
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

def smooth_labels(y_true, label_smoothing=0.1):
    return y_true * (1.0 - label_smoothing) + (label_smoothing / tf.cast(tf.shape(y_true)[1], tf.float32))

def custom_loss_with_label_smoothing(y_true, y_pred, label_smoothing=0.1):
    y_true_smooth = smooth_labels(y_true, label_smoothing)
    return keras.losses.categorical_crossentropy(y_true_smooth, y_pred)

def accuracy(y_true, y_pred):
    """
    Calculate the accuracy of binary class segmentation.
    
    Parameters:
    - y_true: the ground truth tensor.
    - y_pred: the predicted tensor.
    
    Returns:
    - accuracy_score: the accuracy of the segmentation.
    """
    # Ensure the prediction is binary (0 or 1)
    y_pred = tf.cast(y_pred > 0.5, tf.float32)
    
    # Flatten the tensors to convert any 2D image masks into 1D vectors
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    
    # Calculate the number of correct predictions
    correct_predictions = tf.equal(y_true_f, y_pred_f)
    
    # Calculate accuracy
    accuracy_score = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
    
    return accuracy_score

def true_positives(y_true, y_pred, r=2):
    """
    Calculate True Positives within a neighborhood.
    
    Args:
    y_true: Ground truth binary mask
    y_pred: Predicted binary mask
    r: Radius of the neighborhood (kernel size will be 2r+1)
    
    Returns:
    True Positives count
    """
    # Ensure inputs are float32
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    # Add batch and channel dimensions if not present
    if len(y_true.shape) == 2:
        y_true = y_true[tf.newaxis, ..., tf.newaxis]
    if len(y_pred.shape) == 2:
        y_pred = y_pred[tf.newaxis, ..., tf.newaxis]
    
    # Create a max pooling layer
    pool = keras.layers.MaxPool2D(pool_size=(2*r+1, 2*r+1), strides=1, padding='same')
    
    # Apply max pooling to y_true
    y_true_dilated = pool(y_true)
    # print(y_true_dilated[0])
    
    # Element-wise multiplication
    tp = y_true_dilated * y_pred
    
    # Sum all true positives
    return tf.reduce_sum(tp)


def false_positives(y_true, y_pred, r=2):
    """
    Calculate False Positives considering a neighborhood.
    
    Args:
    y_true: Ground truth binary mask
    y_pred: Predicted binary mask
    r: Radius of the neighborhood (kernel size will be 2r+1)
    
    Returns:
    False Positives count
    """
    # Ensure inputs are float32
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    # Add batch and channel dimensions if not present
    if len(y_true.shape) == 2:
        y_true = y_true[tf.newaxis, ..., tf.newaxis]
    if len(y_pred.shape) == 2:
        y_pred = y_pred[tf.newaxis, ..., tf.newaxis]
    
    # Create a max pooling layer
    pool = keras.layers.MaxPool2D(pool_size=(2*r+1, 2*r+1), strides=1, padding='same')
    
    # Apply max pooling to y_true
    y_true_dilated = pool(y_true)
    
    # Calculate false positives
    fp = y_pred * (1 - y_true_dilated)
    
    # Sum all false positives
    return tf.reduce_sum(fp)



def false_negatives(y_true, y_pred, r=2):
    """
    Calculate False Negatives considering a neighborhood.
    
    Args:
    y_true: Ground truth binary mask
    y_pred: Predicted binary mask
    r: Radius of the neighborhood (kernel size will be 2r+1)
    
    Returns:
    False Negatives count
    """
    # Ensure inputs are float32
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    # Add batch and channel dimensions if not present
    if len(y_true.shape) == 2:
        y_true = y_true[tf.newaxis, ..., tf.newaxis]
    if len(y_pred.shape) == 2:
        y_pred = y_pred[tf.newaxis, ..., tf.newaxis]
    
    # Create a max pooling layer
    pool = keras.layers.MaxPool2D(pool_size=(2*r+1, 2*r+1), strides=1, padding='same')
    
    # Apply max pooling to y_pred
    y_pred_dilated = pool(y_pred)
    
    # Calculate false negatives
    fn = y_true * (1 - y_pred_dilated)
    
    # Sum all false negatives
    return tf.reduce_sum(fn)

def true_negatives(y_true, y_pred):
    """
    Calculate True Negatives.
    
    Args:
    y_true: Ground truth binary mask
    y_pred: Predicted binary mask
    
    Returns:
    True Negatives count
    """
    # Ensure inputs are float32
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    # Add batch and channel dimensions if not present
    if len(y_true.shape) == 2:
        y_true = y_true[tf.newaxis, ..., tf.newaxis]
    if len(y_pred.shape) == 2:
        y_pred = y_pred[tf.newaxis, ..., tf.newaxis]
    
    # Calculate true negatives
    tn = (1 - y_true) * (1 - y_pred)
    
    # Sum all true negatives
    return tf.reduce_sum(tn)


def r_data_prep(y_true, y_pred):
    y_pred = tf.cast(y_pred > 0.3, tf.float32)
    tp = true_positives(y_true, y_pred)
    fp = false_positives(y_true, y_pred)
    fn = false_negatives(y_true, y_pred)
    return tp, fp, fn

def r_precision(y_true, y_pred):
    tp, fp, fn = r_data_prep(y_true, y_pred)
    precision = tp / (tp + fp + 1e-5)
    return precision

def r_recall(y_true, y_pred):
    tp, fp, fn = r_data_prep(y_true, y_pred)
    recall = tp / (tp + fn + 1e-5)
    return recall

    
def r_f1_score(y_true, y_pred):
    p =  r_precision(y_true, y_pred)
    r = r_recall(y_true, y_pred)
    f1_scores = 2 * ((p * r) / (p + r + 1e-5)) 
    return f1_scores



def r_accuracy(y_true, y_pred, r=2):
    """
    Calculate Accuracy with neighborhood consideration for TP, FP, and FN.
    
    Args:
    y_true: Ground truth binary mask
    y_pred: Predicted binary mask
    r: Radius of the neighborhood (kernel size will be 2r+1)
    
    Returns:
    Accuracy value
    """
    # Ensure inputs are float32
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    # Add batch and channel dimensions if not present
    if len(y_true.shape) == 2:
        y_true = y_true[tf.newaxis, ..., tf.newaxis]
    if len(y_pred.shape) == 2:
        y_pred = y_pred[tf.newaxis, ..., tf.newaxis]
    
    # Create a max pooling layer
    pool = keras.layers.MaxPool2D(pool_size=(2*r+1, 2*r+1), strides=1, padding='same')
    
    # Calculate TP
    y_true_dilated = pool(y_true)
    tp = tf.reduce_sum(y_true_dilated * y_pred)
    
    # Calculate FP
    fp = tf.reduce_sum(y_pred * (1 - y_true_dilated))
    
    # Calculate FN
    y_pred_dilated = pool(y_pred)
    fn = tf.reduce_sum(y_true * (1 - y_pred_dilated))
    
    # Calculate TN (pixel-wise, no neighborhood consideration)
    tn = tf.reduce_sum((1 - y_true) * (1 - y_pred))
    
    # Calculate Accuracy
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    
    return accuracy


def r_dice_coeff(y_true, y_pred, r=2, smooth=1e-6):
    """
    Calculate Dice Loss with neighborhood consideration.
    
    Args:
    y_true: Ground truth binary mask
    y_pred: Predicted binary mask
    r: Radius of the neighborhood (kernel size will be 2r+1)
    smooth: Smoothing factor to avoid division by zero
    
    Returns:
    Dice Loss value
    """
    # Ensure inputs are float32
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    # Add batch and channel dimensions if not present
    if len(y_true.shape) == 2:
        y_true = y_true[tf.newaxis, ..., tf.newaxis]
    if len(y_pred.shape) == 2:
        y_pred = y_pred[tf.newaxis, ..., tf.newaxis]
    
    # Create a max pooling layer
    pool = keras.layers.MaxPool2D(pool_size=(2*r+1, 2*r+1), strides=1, padding='same')
    
    # Calculate TP
    y_true_dilated = pool(y_true)
    tp = tf.reduce_sum(y_true_dilated * y_pred)
    
    # Calculate FP
    fp = tf.reduce_sum(y_pred * (1 - y_true_dilated))
    
    # Calculate FN
    y_pred_dilated = pool(y_pred)
    fn = tf.reduce_sum(y_true * (1 - y_pred_dilated))
    
    # Calculate Dice coefficient
    dice_coeff = (2 * tp + smooth) / (2 * tp + fp + fn + smooth)\
    
    return dice_coeff

def r_dice_loss(y_true, y_pred, r=2, smooth=1e-6):
    """
    Calculate Dice Loss with neighborhood consideration.
    
    Args:
    y_true: Ground truth binary mask
    y_pred: Predicted binary mask
    r: Radius of the neighborhood (kernel size will be 2r+1)
    smooth: Smoothing factor to avoid division by zero
    
    Returns:
    Dice Loss value
    """
    # Ensure inputs are float32
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    # Add batch and channel dimensions if not present
    if len(y_true.shape) == 2:
        y_true = y_true[tf.newaxis, ..., tf.newaxis]
    if len(y_pred.shape) == 2:
        y_pred = y_pred[tf.newaxis, ..., tf.newaxis]
    
    # Create a max pooling layer
    pool = keras.layers.MaxPool2D(pool_size=(2*r+1, 2*r+1), strides=1, padding='same')
    
    # Calculate TP
    y_true_dilated = pool(y_true)
    tp = tf.reduce_sum(y_true_dilated * y_pred)
    
    # Calculate FP
    fp = tf.reduce_sum(y_pred * (1 - y_true_dilated))
    
    # Calculate FN
    y_pred_dilated = pool(y_pred)
    fn = tf.reduce_sum(y_true * (1 - y_pred_dilated))
    
    # Calculate Dice coefficient
    dice_coeff = (2 * tp + smooth) / (2 * tp + fp + fn + smooth)
    
    # Calculate Dice Loss
    dice_loss = 1 - dice_coeff
    
    return dice_loss


## Focal Loss
def binary_focal_loss(gamma=2., alpha=0.25,epsilon=1e-7):
    def focal_loss(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        loss_pos = -alpha * y_true * tf.math.pow(1 - y_pred, gamma) * tf.math.log(y_pred)
        loss_neg = -(1 - alpha) * (1 - y_true) * tf.math.pow(y_pred, gamma) * tf.math.log(1 - y_pred)
        loss = tf.reduce_mean(loss_pos + loss_neg)
        return loss
    return focal_loss



# Define dice loss
def dice_loss(smooth=1e-6):
    def dice(y_true, y_pred):
        y_true_f = tf.reshape(y_true, [-1])
        y_pred_f = tf.reshape(y_pred, [-1])
        
        intersection = tf.reduce_sum(y_true_f * y_pred_f)
        dice_coeff = (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
        
        return 1 - dice_coeff
    return dice

# Combine binary focal loss and dice loss
def combined_loss(gamma=2., alpha=0.25, smooth=1e-6, weight_focal=0.5, weight_dice=0.5):
    def loss(y_true, y_pred):
        focal = binary_focal_loss(gamma, alpha)(y_true, y_pred)
        dice = dice_loss(smooth)(y_true, y_pred)
        return weight_focal * focal + weight_dice * dice
    return loss


# Combine binary focal loss and dice loss
def r_combined_loss(gamma=2., alpha=0.25, smooth=1e-6, weight_focal=0.5, weight_dice=0.5):
    def loss(y_true, y_pred):
        focal = binary_focal_loss(gamma, alpha)(y_true, y_pred)
        dice = r_dice_loss(y_true, y_pred)
        return weight_focal * focal + weight_dice * dice
    return loss