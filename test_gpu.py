import tensorflow as tf

# Check for GPU availability
if tf.config.list_physical_devices('GPU'):
    print("GPU is available and detected.")
    # Run a simple TensorFlow operation on the GPU
    with tf.device('/GPU:0'):
        a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3])
        b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2])
        c = tf.matmul(a, b)
        print(c)
else:
    print("No GPU detected.")