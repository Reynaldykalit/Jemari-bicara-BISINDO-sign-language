import tensorflow as tf
import os

model_path = r"D:\jemari bicara\assets\models\lstm_best_final.h5"
tflite_path = r"D:\jemari bicara\assets\models\classifier_final.tflite"

print("Loading model from:", model_path)
if not os.path.exists(model_path):
    print(f"Error: Model path does not exist: {model_path}")
    exit(1)

model = tf.keras.models.load_model(model_path)
print("Converting model to tflite...")

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
converter.experimental_new_converter = True

tflite_model = converter.convert()

print("Saving TFLite model to:", tflite_path)
with open(tflite_path, "wb") as f:
    f.write(tflite_model)
    
print("Successfully converted and saved TFLite model!")
