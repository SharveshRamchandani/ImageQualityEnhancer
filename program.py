from flask import Flask, request, send_file
from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf
import numpy as np
from PIL import Image
import io

# Initialize the Flask app
app = Flask(_name_)

# Define and create the model
def create_model():
    input_layer = tf.keras.layers.Input(shape=(256, 256, 3))  # Specify input shape
    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(input_layer)
    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.UpSampling2D((2, 2))(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.Conv2D(3, (3, 3), activation='tanh', padding='same')(x)
    model = tf.keras.Model(inputs=input_layer, outputs=x)
    model.compile(optimizer='adam', loss='mse')
    return model

# Create and load the model
model = create_model()
try:
    model.load_weights('my_tensorflow_model.h5')  # Load your model weights
except FileNotFoundError:
    print("Model weights file not found. Please check the path.")

# Preprocess the image to match model input
def preprocess_image(image):
    image = image.resize((256, 256))  # Resize to the required input shape
    image = np.array(image) / 127.5 - 1  # Normalize to [-1, 1]
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Postprocess model output to convert back to image format
def postprocess_image(image):
    image = (image + 1) * 127.5  # Rescale to [0, 255]
    image = np.clip(image, 0, 255).astype(np.uint8)
    return Image.fromarray(image)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "No file part", 400

    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400

    # Read the uploaded image
    image = Image.open(file.stream).convert('RGB')

    # Preprocess image
    input_image = preprocess_image(image)

    # Get the super-resolved image from the model
    sr_image = model.predict(input_image)
    sr_image = np.squeeze(sr_image, axis=0)  # Remove batch dimension

    # Postprocess the image
    output_image = postprocess_image(sr_image)

    # Save the output to a byte stream
    img_io = io.BytesIO()
    output_image.save(img_io, 'JPEG', quality=95)
    img_io.seek(0)

    # Send the image back as response
    return send_file(img_io, mimetype='image/jpeg')

if _name_ == "_main_":
    app.run(host='0.0.0.0', port=5000, debug=True)