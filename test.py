import tensorflow as tf
b = tf.constant([0,689, 1121, 104, 45, 29, 12, 729, 190, 564, 135, 151, 88, 124, 427, 649, 302, 876, 683, 508, 377, 102, 27, 83, 260, 140, 78, 85, 849, 61, 24, 30, 27, 78, 20, 16])
module_of_role = tf.expand_dims(tf.expand_dims(b,axis=0),axis=2)
print(module_of_role)
