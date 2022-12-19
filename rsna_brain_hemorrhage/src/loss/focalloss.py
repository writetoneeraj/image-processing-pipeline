import numpy as np
import tensorflow as tf

def multiclass_log_loss(y_true, y_pred, class_weights = None):
    eps = 1e-12
    y_pred = np.clip(y_pred, eps, 1-eps)
    single_class_log_loss = -np.mean(y_true * np.log(y_pred) + (1-y_true)*np.log(1-y_pred), axis=0)
    if class_weights == None:
        log_loss = np.mean(single_class_log_loss)
    else:
        log_loss = np.sum(class_weights * single_class_log_loss)

def focal_log_loss(class_weights = None, alpha = 0.5, gamma = 2):
    def multiclass_focal_log_loss(y_true, y_pred):
        print(f"y_true : {y_true}")
        print(f"y_pred : {y_pred}")
        eps = 1e-12
        pt = tf.where(tf.equal(y_true, 1), y_pred, 1-y_pred)
        alpha_t = tf.where(tf.equal(y_true, 1), alpha, 1-alpha)
        #print(pt)
        #print(alpha_t)
        pt = tf.clip_by_value(pt, eps, 1-eps)
        #print(alpha)
        focal_loss = -tf.reduce_mean(tf.multiply(tf.multiply(alpha_t,tf.pow(1-pt,gamma)),tf.math.log(pt)), axis=0)
        if class_weights is None:
            focal_loss = tf.reduce_mean(focal_loss, axis=0)
        else:
            focal_loss = tf.reduce_sum(tf.multiply(focal_loss, class_weights))
        print(focal_loss)
        return focal_loss
    return multiclass_focal_log_loss

def _get_raw_xentropies(y_true, y_pred):
    y_pred = tf.clip_by_value(y_pred, 1e-7, 1-1e-7)
    xentropies = y_true * tf.log(y_pred) + (1-y_true) * tf.log(1-y_pred)
    return -xentropies

# multilabel focal loss equals multilabel loss in case of alpha=0.5 and gamma=0 
def mutlilabel_focal_loss_inner(y_true, y_pred,class_weights=None, alpha=0.5, gamma=2):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    xentropies = _get_raw_xentropies(y_true, y_pred)

    # compute pred_t:
    y_t = tf.where(tf.equal(y_true,1), y_pred, 1.-y_pred)
    alpha_t = tf.where(tf.equal(y_true, 1), alpha * tf.ones_like(y_true), (1-alpha) * tf.ones_like(y_true))

    # compute focal loss contributions
    focal_loss_contributions =  tf.multiply(tf.multiply(tf.pow(1-y_t, gamma), xentropies), alpha_t) 

    # our focal loss contributions have shape (n_samples, s_classes), we need to reduce with mean over samples:
    focal_loss_per_class = tf.reduce_mean(focal_loss_contributions, axis=0)

    # compute the overall loss if class weights are None (equally weighted):
    if class_weights is None:
        focal_loss_result = tf.reduce_mean(focal_loss_per_class)
    else:
        # weight the single class losses and compute the overall loss
        weights = tf.constant(class_weights, dtype=tf.float32)
        focal_loss_result = tf.reduce_sum(tf.multiply(weights, focal_loss_per_class))
    with tf.Session() as sess:
        print(focal_loss_result.eval())

