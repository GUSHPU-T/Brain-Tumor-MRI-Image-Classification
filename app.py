#!/usr/bin/env python
# coding: utf-8

# In[5]:


from flask import Flask, render_template, request
import numpy as np
import cv2
from tensorflow.keras.models import model_from_json
import tensorflow as tf
from tensorflow import keras

app = Flask(__name__)

model_path = r"C:\Users\kushp\deployment_model\model.json"
weights_path = r"C:\Users\kushp\deployment_model\model_weights.h5"
with open(model_path, 'r') as json_file:
    loaded_model_json = json_file.read()
    loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights(weights_path)


# Define route for home page
@app.route('/')
def home():
    return render_template('index.html')

# Define function to handle image upload and prediction
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['image']
        img = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
        img = cv2.resize(img, (512, 512))  
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  
        img = np.expand_dims(img, axis=0)
        img = tf.keras.applications.mobilenet_v2.preprocess_input(img)

        # Make prediction
        pred = loaded_model.predict(img)
        pred_class = np.argmax(pred[0])

        # Define the classes
        classes = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']

        # Get the predicted class label
        predicted_label = classes[pred_class]

        return render_template('result.html', predicted_label=predicted_label)

if __name__ == '__main__':
    app.run(debug=True)
