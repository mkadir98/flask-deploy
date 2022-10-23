from __future__ import division, print_function
# coding=utf-8

import os

import numpy as np

# Keras

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


# Flask utils

from flask import Flask, request, render_template

from werkzeug.utils import secure_filename
#from skimage import transform


app = Flask(__name__)


model = load_model("flask-heroku.h5",compile=False)

def model_predict(img_path, model):

  #  img = image_embedding_column.load_img(img_path, target_size=(500, 500))
   # img = np.array(img).astype('float32')/255
   # img = transform.resize(img, (500,500, 3))
   # img = np.expand_dims(img, axis=0)

    img=image.load_img(img_path,target_size=(200,200))
    img = image.img_to_array(img)
    img = np.expand_dims(img,axis=0)

    preds = model.predict(img)
    print("Tahmin Olasılıkları :",preds)
    print("-------------------------------")
    preds=np.argmax(preds[0])
    print("Tahmin :",preds)
    return preds

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route("/hakkında")
def hakkında():
    return "Bu web site Yaşar Niyazoğlu Tarafından Oluşturulmuşur"


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']

        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        preds = model_predict(file_path, model)
        os.remove(file_path)

        isim1 = 'Covid'
        isim2 = 'Normal'
        isim3 = 'Viral'
        if preds == 0:
            return isim1
        elif preds == 1:
            return isim2
        else :
            return isim3
    return None

if __name__ == '__main__':
    app.run()
    
