from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
import sys
from PIL import Image
import pytesseract
from pytesseract import Output
import argparse
import cv2
import pickle

import numpy as np
from sklearn import feature_extraction, model_selection, naive_bayes, metrics
from tensorflow import keras
import tensorflow as tf
from sklearn.feature_extraction.text import CountVectorizer
#from keras.preprocessing.image import load_img
#from keras.preprocessing.image import img_to_array
pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe' 

from img_model import load_model
import logging


app = Flask(__name__)
UPLOAD_FOLDER = './static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER 
app.config['MAX_CONTENT_LENGTH'] = 1 * 1024 * 1024

HEADERS = {'content-type': 'application/json'}
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])
MODEL_1 = 'models/final.pkl'
MODEL_2 = 'models/image_model_saved.h5'


MD5_dict = {}

# Image resizing utils
def resize_image_array(img, img_size_dims):
  img = cv2.resize(img, dsize=img_size_dims, 
                     interpolation=cv2.INTER_CUBIC)
  img = np.array(img, dtype=np.float32)
  return img

# load the text spam model
loaded_vectorizer = pickle.load(open('models/vectorizer.pickle', 'rb'))
text_model = pickle.load(open(MODEL_1, 'rb'))
app.logger.info(text_model)
image_model = load_model(MODEL_2)
app.logger.info(image_model)



def text_spam(img):
  def remove_non_ascii(text):
    return ''.join(i for i in text if ord(i)<128)

  def remove_newline_feed(text):
      return text.replace('\n','')

  def clearup_text(text):
      return remove_newline_feed(remove_non_ascii(text))


  text = pytesseract.image_to_string(img)
  text = clearup_text(text)
  iterables = [text]
  _score = text_model.predict(loaded_vectorizer.transform(iterables))
  probabilities = text_model.predict_proba(loaded_vectorizer.transform(iterables))[:, 1]
  # print(probabilities)
  return probabilities


@app.route("/baitaware/api/v1/liveness")
def liveness():
  return 'API live!'

@app.route("/baitaware/api/v1/model")
def model():
  return render_template("index.html")  

@app.route("/baitaware/api/v1/about")
def about():
  return render_template("about.html")

@app.route('/baitaware/api/v1/uploader', methods = ['GET', 'POST'])
def upload_file():
   if request.method == 'POST':
      f = request.files['file']

      def allowed_file(filename):
          return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

      # create a secure filename
      filename = secure_filename(f.filename)

      # save file to /static/uploads
      filepath = os.path.join(app.config['UPLOAD_FOLDER'],filename)
      f.save(filepath)

      # ofilename = os.path.join(app.config['UPLOAD_FOLDER'],"{}.png".format(os.getpid()))
      # cv2.imwrite(ofilename, gray)
      
      img = keras.preprocessing.image.load_img(filepath, target_size=(150, 150))
      app.logger.info(type(img))
      # convert to numpy array
      img_array =  np.array([keras.preprocessing.image.img_to_array(img)/255.])
      app.logger.info(img_array.dtype)
      app.logger.info(img_array.shape)

      # perform OCR on the processed image
      _score = text_spam(img)
      # print(_score)
      # image classification on processed image
      pred = image_model.predict(img_array)
      pred = pred[0][0]
      pred = float(((pred + pred + pred + _score)/4))
      
      result = 'Clickbait Detected' if pred > 0.5 else 'Legit'
      # load the example image and convert it to grayscale
      image = cv2.imread(filepath)
      if(img is not None):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # apply thresholding to preprocess the image
        gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

      # apply median blurring to remove any blurring
      img = cv2.medianBlur(gray, 3)

      

      out_img = image.copy()
      # _score = 0.67
      d = pytesseract.image_to_data(img, output_type=Output.DICT)
      n_boxes = len(d['level'])
      for i in range(n_boxes):
         (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
         cv2.rectangle(out_img, (x, y), (x + w, y + h), (0, 255, 0), 2)

      pfilename = os.path.join(app.config['UPLOAD_FOLDER'],"out_"+filename)
      cv2.imwrite(pfilename, out_img)

      # remove the processed image
      # os.remove(ofilename)

      return render_template("uploaded.html", fname=filename, fname2="out_" +filename, 
        result=result, score=str(round(pred*100,2)))

if __name__ == '__main__': 
   app.run(host="0.0.0.0", port=5000, debug=True)
