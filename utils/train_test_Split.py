import random
from itertools import combinations
import numpy as np
from sklearn.model_selection import KFold
from operator import itemgetter

def train_test_set(year_range, all_images,num_test_sets = 10,num_years_in_test = 3):
    # List of all years in the dataset
    all_years = [str(i) for i in range(year_range[0], year_range[1] + 1)]

    # Storing the train-test splits
    train_test_splits = []
    test_years_list = []
    random.seed(42)

    # Generate random test sets
    for _ in range(num_test_sets):
        test_years = random.sample(all_years, num_years_in_test)
        test_years_list.append(test_years)
        train_years = [year for year in all_years if year not in test_years]

        test_set = [filename for filename in all_images if any(test_year in filename for test_year in test_years)]
        test_idx = [idx for idx,filename in enumerate(all_images) if any(test_year in filename for test_year in test_years)]
        train_set = [filename for filename in all_images if any(train_year in filename for train_year in train_years)]
        train_idx = [idx for idx,filename in enumerate(all_images) if any(train_year in filename for train_year in train_years)]

        train_test_splits.append((train_idx, test_idx))
        
    return train_test_splits,test_years_list

    
def Kfold_splits(train_images,n_splits=5, shuffle=True, random_state=42):
    kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    kf.get_n_splits(train_images)
    
    train_indexes = []
    val_indexes = []
    for i, (train_index, val_index) in enumerate(kf.split(train_images)):
        train_indexes.append(train_index)
        val_indexes.append(val_index)
        
    return train_indexes, val_indexes


def image_mask_list(index,images,masks):
    # Extracting filenames
    image_names = list(itemgetter(*index)(images))
    mask_names = list(itemgetter(*index)(masks))

    return image_names, mask_names