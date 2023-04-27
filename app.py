import flask
import io
import string
import time
import os
import numpy as np
from flask import Flask, jsonify, request, render_template
from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd
import dill
from os.path import join
from  function_preprocess import *
#model = pickle.load(open('models/best_model_passive_agressive_tfidf', 'rb'))
# création de l'objet API flask
app = Flask(__name__, template_folder='templates')

import pickle

def read_file(folder_path, file_name):
    with open(join(folder_path, file_name), "rb") as input:
        data= dill.load(input)
    return data
model = read_file("models/", "pipeline_tfidf_pac")


#chargement du modèle et du vectorizer
mlb = pd.read_pickle('models/multibinarizer')

#définition du schéma/point de terminaison (API)
@app.route('/', methods=['GET', 'POST'])
def home():
   request_type_str = request.method
   if request_type_str == 'GET' :
      return render_template('index.html')
   else:
      # récupération des infos envoyées par l'utilisateur
      message = request.form['message']
      title = request.form['title']
      question =  concatenation(message, title, 5)
      predictions = model.predict([question])
      tags = mlb.inverse_transform(predictions)
      tags = [' '.join(c for c in s if c not in string.punctuation) for s in tags]
      return render_template('index.html', tags_predits= tags)
   
if __name__ == "__main__":
    app.run(debug=True)