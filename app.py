import os
import sys

# Flask
from flask import Flask, redirect, url_for, request, render_template, Response, jsonify, redirect
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
from flask_wtf import FlaskForm
from wtforms import StringField, IntegerField, FloatField, validators
from wtforms.validators import InputRequired

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras


from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Some utilites
import numpy as np
from util import base64_to_pil

from PIL import Image
import numpy as np

from skimage import transform


# Declare a flask app
app = Flask(__name__)

#class ObjectClassifier:
# You can use pretrained model from Keras
# Check https://keras.io/applications/
from tensorflow.keras.applications.resnet50 import ResNet50
model = ResNet50(weights='imagenet', include_top=False)

print('Model loaded. Check http://127.0.0.1:5000/')


# Model saved with Keras model.save()
MODEL_PATH = 'models/Cifar10.h5'

# Load your own trained model
model1 = load_model(MODEL_PATH)
print('Model loaded. Start serving...')


def model_predict(img, model1):
    
    np_image = np.array(img).astype('float32')/255
    np_image = transform.resize(np_image, (32, 32, 3))
    np_image = np.expand_dims(np_image, axis=0)

    return model1.predict_classes(np_image).item()
'''class DigitClassifier:
# You can use pretrained model from Keras
# Check https://keras.io/applications/
    from keras.applications.VGG16 import vgg16
    model1 = vgg16(weights='imagenet', include_top=False)
print('Model loaded. Check http://127.0.0.1:5000/')


# Model saved with Keras model.save()
MODEL_PATH = 'models/MNIST.h5'

# Load your own trained model
model1 = load_model(MODEL_PATH)
print('Model loaded. Start serving...')


def model_predict(img, model1):
    
    np_image = np.array(img).astype('float32')/255
    np_image = transform.resize(np_image, (28, 28, 1))
    np_image = np.expand_dims(np_image, axis=0)

    return model1.predict_classes(np_image).item()

'''
@app.route('/', methods=['GET'])
def index():
   # objectclassifier = ObjectClassifier()
    #digitclassifier = DigitClassifier()
    #    return render_template('showData.html', form = objectclassifier)

    #if digitclassifier.validate_on_submit():
     #  return render_template('showData.html', form1 = digitclassifier)   

    #return render_template('index.html',form1 = digitclassifier, form = objectclassifier)
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])

def predict():
    if request.method == 'POST':
        # Get the image from post request
        img = base64_to_pil(request.json)

        # Save the image to ./uploads
        img.save("./uploads/image.png")

        # Make prediction
        preds = model_predict(img, model1)
        
        switcher = {0:"Airplane",1:"auto-mobile",2:"Bird",3:"Cat",4:"Deer",5:"Dog",6:"Frog",7:"Horse",8:"Ship",9:"Truck"}
        
        result  = switcher.get(int(preds),"Nothing")
        # Serialize the result, you can add additional fields
        return jsonify(result=result)

    return None
    
@app.route('/uploader', methods = ['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        f.save(img)
if __name__ == '__main__':
    # app.run(port=5002, threaded=False)

    # Serve the app with gevent
    http_server = WSGIServer(('0.0.0.0', 5000), app)
    http_server.serve_forever()
