import tensorflow as tf


def masked_softmax_cross_entropy(preds, labels, mask):
    """Softmax cross-entropy loss with masking."""

    # loss = -tf.reduce_mean(preds* tf.log(tf.clip_by_value(labels, 1e-10, 1.0)))
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=preds, labels=labels)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    loss *= mask
    return tf.reduce_mean(loss)


def masked_accuracy(preds, labels, mask):
    """Accuracy with masking."""
    correct_prediction = tf.equal(tf.argmax(preds, 1), tf.argmax(labels, 1))
    accuracy_all = tf.cast(correct_prediction, tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    accuracy_all *= mask
    return tf.reduce_mean(accuracy_all)


# 多标签数据集
def multilabel_masked_sigmoid_cross_entropy(preds, labels, mask):
    """Softmax cross-entropy loss with masking."""
    labels = tf.cast(labels, dtype=tf.float32)
    # reduce_sum/ reduce_mean
    loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=preds, labels=labels), axis=1)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    loss *= mask
    return tf.reduce_mean(loss)