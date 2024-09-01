import keras
from keras import layers
from keras import ops

import os
import numpy as np
from glob import glob
import cv2
from scipy.io import loadmat
import matplotlib.pyplot as plt

import os
import sys
import random
from datetime import datetime

# For data preprocessing
import tensorflow as tf
from tensorflow import image as tf_image
from tensorflow import data as tf_data
from tensorflow import io as tf_io

from keras import backend as K

import segmentation_models as sm
from utils import *
from refined_metrcis import *
from ResUNet import ResUNet
from deeplabModelV3 import *


def run_resunet(DATA_DIR="/scratch/gza5dr/Canal_Datasets/NHDShape/procssed_data/",
               img_folder_name ='resunet_R110th0_it1output_nodialated_image_patches_512',
               mask_folder_name ='resunet_R110th0_it1output_nodialated_mask_patches_512',
               output_path = "/scratch/gza5dr/Current_Canal_Experiments/Proposed_Model_Pipeline/implementation/framework/final_framework/ResUNet/",
               from_scratch = True,
               epochs = 35,
               weight_prefix = "resunet",
               batch_size = 4,
               num_classes = 1,
               image_size = 512,
               learning_rate = 1e4,
               optimizer = 'Adam',
               pretrained_weights = '',
               loss = 'bce',
               model_type = ""
               ):
    ## Seeding 
    seed = 42
    random.seed = seed
    np.random.seed = seed
    tf.seed = seed

    IMAGE_SIZE = 512
    BATCH_SIZE = 4
    NUM_CLASSES = 1

    regions = ['r1','r2', 'r3']
    dh = DatasetHandler(DATA_DIR=DATA_DIR,BATCH_SIZE = batch_size,NUM_CLASSES = num_classes,IMAGE_SIZE=image_size,
                       img_folder_name=img_folder_name,mask_folder_name=mask_folder_name)
    # train_test_splits_1 = dh.prep_data_splits()
    # dh.save_json_1(regions=regions,fpath='/scratch/gza5dr/Canal_Datasets/NHDShape/procssed_data/5_data_json',n=3)
    # data = dh.load_json('/scratch/gza5dr/Canal_Datasets/NHDShape/procssed_data/5_data_json')
    # train_dataset,val_dataset,test_dataset=dh.prep_data(data,n_splits=2,d_set=0)
    
    test_dir = '/scratch/gza5dr/Canal_Datasets/NHDShape/procssed_data/'
    test_images = sorted(glob(os.path.join(test_dir, f"set1_images_512/*")))
    test_masks = sorted(glob(os.path.join(test_dir, f"set1_masks_512/*")))
    
    images = dh.train_images
    train_images = [filename for idx,filename in enumerate(dh.train_images) if any(train in filename for train in ['train'])]
    test_images = [filename for idx,filename in enumerate(test_images) if any(train in filename for train in ['test'])]
    train_masks = [filename for idx,filename in enumerate(dh.train_masks) if any(train in filename for train in ['train'])]
    test_masks = [filename for idx,filename in enumerate(test_masks) if any(train in filename for train in ['test'])]

    dp = DataProcessor(IMAGE_SIZE=dh.IMAGE_SIZE,BATCH_SIZE= dh.BATCH_SIZE,NUM_CLASSES=dh.NUM_CLASSES)
    train_dataset = dp.data_generator(train_images, train_masks,aug=True)
    test_dataset = dp.data_generator(test_images, test_masks,aug=False)

    smooth = 1.

    def dice_coef(y_true, y_pred):
        y_true_f = keras.layers.Flatten()(y_true)
        y_pred_f = keras.layers.Flatten()(y_pred)
        intersection = tf.reduce_sum(y_true_f * y_pred_f)
        return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

    bce = keras.losses.BinaryCrossentropy(from_logits=False)

    checkpoint = f'{output_path}models/'
    csvlogger = f'{output_path}logs/'
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        
    if not os.path.exists(checkpoint):
        os.makedirs(checkpoint)
    
    if not os.path.exists(csvlogger):
        os.makedirs(csvlogger)

    checkpoint_path = f"{checkpoint}/best_{weight_prefix}.weights.h5"
    ckpt = keras.callbacks.ModelCheckpoint(checkpoint_path, verbose=0, save_best_only=True, save_weights_only=True)
    csvlogger_path = f"{csvlogger}/{model_type.lower()}_log_{weight_prefix}"+'_'+datetime.now().strftime("%Y-%m-%d_%H:%M:%S")+".csv"
    csvlogger = keras.callbacks.CSVLogger(csvlogger_path, separator=',', append=True)
    # callbacks = [ckpt,csvlogger,keras.callbacks.ReduceLROnPlateau()]
    callbacks = [ckpt,csvlogger]
    
    
    if optimizer.lower() == 'adam':
        optimizer = keras.optimizers.Adam(learning_rate)
    if loss.lower() == 'bce':
        loss = bce
        
        
    if model_type.lower() == "resunet":
        model = ResUNet((image_size, image_size, 3))
    elif model_type.lower() == "deeplabv3+":
        resnet50, model = DeeplabV3Plus(image_size=image_size, num_classes=num_classes)
    elif model_type.lower() == "resnet":
        BACKBONE = 'resnet50'
        model = sm.Unet(BACKBONE, 
                input_shape=(image_size, image_size, 3),
                classes=num_classes, activation='sigmoid')
        
    if from_scratch == False:
        model.load_weights(pretrained_weights)
    
        
        
    model.compile(optimizer=optimizer, loss=loss, metrics=[f1_score, precision, recall,dice_coef,r_dice_coeff,r_precision,r_recall,r_f1_score])

    history = model.fit(train_dataset, validation_data=test_dataset, epochs=epochs, callbacks=callbacks, verbose = 2)
    model.save_weights(f"{checkpoint}/final_{weight_prefix}.weights.h5")

    return f"{checkpoint}/final_{weight_prefix}.weights.h5"