import tensorflow as tf

# Load the trained Keras CNN model
model = tf.keras.models.load_model('sign_word_cnn.h5')

# Create a TFLiteConverter
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Optional: Enable optimizations for quantization (smaller/faster model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Convert the model
tflite_model = converter.convert()

# Save the TFLite model
with open('sign_word_cnn.tflite', 'wb') as f:
    f.write(tflite_model)

print('TFLite model saved as sign_word_cnn.tflite') 