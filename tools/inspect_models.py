import tensorflow as tf

def inspect_model(path, name):
    print(f"\n=== Inspecting {name} ({path}) ===")
    try:
        interpreter = tf.lite.Interpreter(model_path=path)
        interpreter.allocate_tensors()
        
        inputs = interpreter.get_input_details()
        outputs = interpreter.get_output_details()
        
        print("Inputs:")
        for idx, i in enumerate(inputs):
            print(f"  [{idx}] Name: {i['name']}, Shape: {i['shape']}, Type: {i['dtype']}")
            
        print("Outputs:")
        for idx, o in enumerate(outputs):
            print(f"  [{idx}] Name: {o['name']}, Shape: {o['shape']}, Type: {o['dtype']}")
    except Exception as e:
        print(f"Error inspecting {name}: {e}")

inspect_model("assets/models/hand_detector_yolo.tflite", "YOLO Hand Detector")
inspect_model("assets/models/classifier_lstm.tflite", "LSTM Sign Classifier")
