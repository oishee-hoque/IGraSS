import tensorflow as tf
import keras

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


def data_prep(y_true, y_pred):
    y_pred = tf.cast(y_pred > 0.3, tf.float32)
    tp = true_positives(y_true, y_pred)
    fp = false_positives(y_true, y_pred)
    fn = false_negatives(y_true, y_pred)
    return tp, fp, fn

def r_precision(y_true, y_pred):
    tp, fp, fn = data_prep(y_true, y_pred)
    precision = tp / (tp + fp + 1e-5)
    return precision

def r_recall(y_true, y_pred):
    tp, fp, fn = data_prep(y_true, y_pred)
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

# Combine binary focal loss and dice loss
def r_combined_loss(gamma=2., alpha=0.25, smooth=1e-6, weight_focal=0.5, weight_dice=0.5):
    def loss(y_true, y_pred):
        focal = binary_focal_loss(gamma, alpha)(y_true, y_pred)
        dice = r_dice_loss(y_true, y_pred)
        return weight_focal * focal + weight_dice * dice
    return loss