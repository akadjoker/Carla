import tensorflow as tf
print("TensorFlow version:", tf.__version__)
print("GPU disponíveis:", tf.config.list_physical_devices('GPU'))
print("GPU em uso:", tf.test.is_gpu_available())  # Depreciado mas útil
