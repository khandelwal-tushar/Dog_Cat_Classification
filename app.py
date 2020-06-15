import base64 
import tensorflow as tf
GPU_OPTIONS = tf.compat.v1.GPUOptions(allow_growth=True)
CONFIG = tf.compat.v1.ConfigProto(gpu_options=GPU_OPTIONS)
sess = tf.compat.v1.Session(config = CONFIG)
import numpy as np
import io 
from PIL import Image
import keras
from keras import backend as K 
from keras.models import Sequential 
#from keras.models import load_model
from tensorflow.keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator 
from keras.preprocessing.image import img_to_array 
from flask import jsonify 
from flask import Flask, render_template, request, redirect, url_for
import cv2
app = Flask(__name__) 

def get_model():
    global model
    model = load_model("C:\\Users\\Asus\\dog_dep\\cats_dogs.h5") 
    print(" * Model loaded!")

def preprocess_image(image, target_size):

    #print(image)
    #image = image.resize(target_size)
    #print(image)
    #image = image.convert('1')
    image = img_to_array(image)
   # image = cv2.flip(image,-1)
    print(image)
    #print(image)
    #image = image / 255.
    #image = np.expand_dims(image, axis = 0)

    img = cv2.resize(image, (100,100))
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    scaled_img = gray_img / 255.
    final_img = scaled_img.reshape(-1,100,100,1)

    print(final_img)
    

    return final_img

print(" * Loading Keras model...") 
get_model()

@app.route("/")
def template_test():
    return render_template('index.html')

@app.route("/predict", methods=["POST"])
def predict():
    message = request.get_json(force=True)
    encoded = message['image'] 
    #rint(encoded)
    decoded = base64.b64decode(encoded) 
    image = Image.open(io.BytesIO(decoded)) 
    processed_image = preprocess_image(image, target_size=(100, 100))

    prediction = model.predict(processed_image)
    print(f"\n\n{prediction}\n\n")
    pred = prediction[0][0].round() 
    print(pred)

    #response = {
        #prediction: {

          #  'dog': pred,

    
       # } 
    #}
    return str(pred)

if __name__ == '__main__':
    app.run(debug=True)
