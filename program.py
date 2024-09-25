import json
from matplotlib import image
import numpy as np
from PIL import Image
from keras.models import Sequential
from keras.layers import Dense, Conv2D, UpSampling2D
from keras.preprocessing.image import load_img, img_to_array
import tensorflow as tf
import http.server
import socketserver
import os

def preprocess_image(image_path, target_size=(256, 256)):
    img = load_img(image_path, target_size=target_size)
    x = img_to_array(img)
    x /= 127.5
    x -= 1.
    return x

def postprocess_image(image):
    image = (image + 1) * 127.5
    image = np.clip(image, 0, 255).astype(np.uint8)
    return Image.fromarray(image)

# Load your model weights
model = Sequential()
model.add(Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(256, 256, 3)))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(3, (3, 3), activation='tanh', padding='same'))
model.compile(optimizer='adam', loss='mse')

# Load training data
train_images = []
train_labels = []

data_dir = 'files/'  # Replace with the path to your dataset directory
for filename in os.listdir(data_dir):
    if filename.endswith('.jpg'):
        # Load low-resolution image
        lr_image_path = os.path.join(data_dir, filename)
        # Preprocess images
        lr_image = preprocess_image(lr_image_path)

        # Load corresponding high-resolution image (adjust path as needed)
        hr_image_path = os.path.join(data_dir, filename.replace('lr', 'hr'))
        hr_image = preprocess_image(hr_image_path, target_size=(512, 512))

        # Append to training data
        train_images.append(lr_image)
        train_labels.append(hr_image)

# Train the model
model.fit(np.array(train_images), np.array(train_labels), epochs=100, batch_size=32)

# Save the trained model
model.save('my_model.h5')

def handle_image_upload(image_data):
    # Convert the image data to a PIL Image object
    image = Image.frombytes('RGB', (image_data['width'], image_data['height']), image_data['data'])

    # Preprocess the image
    preprocessed_image = preprocess_image(image)

    # Get the super-resolved image from the model
    sr_image = model.predict(np.array([preprocessed_image]))  # Add batch dimension
    sr_image = np.squeeze(sr_image, axis=0)  # Remove batch dimension

    # Postprocess the image
    output_image = postprocess_image(sr_image)

    # Convert the output image to a suitable format for sending (e.g., base64)
    output_image_data = {
        'width': output_image.width,
        'height': output_image.height,
        'data': output_image.tobytes()
    }

    return output_image_data

class MyHandler(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)

        # Parse the image data from the request body
        image_data = json.loads(post_data)

        # Process the image and get the super-resolved image data
        sr_image_data = handle_image_upload(image_data)

        # Send the super-resolved image data as a response
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(sr_image_data).encode('utf-8'))

with socketserver.TCPServer(('localhost', 5000), MyHandler) as httpd:
    print('Serving on port 5000...')
    httpd.serve_forever()
