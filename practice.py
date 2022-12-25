# create a float numpy ARRAY and convert it into a tensorflow tensor

# numpy array
import numpy as np
numpy_A = np.arange(15, 90, dtype=np.float128)
# print(numpy_A)

import tensorflow as tf
# pass in numpy directly
tensor_t = tf.constant(numpy_A)
print(tensor_t)
