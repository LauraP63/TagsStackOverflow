import flask
import io
import string
import time
import os
import numpy as np
from flask import Flask, jsonify, request, render_template
import pickle
import bs4
from bs4 import BeautifulSoup as bs
import re # pour les  regex
import nltk
from nltk.tokenize import WhitespaceTokenizer
import pandas as pd

# création de l'objet API flask
app = Flask(__name__, template_folder='template')


#chargement du modèle 
model = pickle.load(open('models/model_passive_agressive_tfidf', 'rb'))
#définition de l'objet /class pour réaliser les reuêtes
# class Question(BaseModel):
#     title: str
#     message: str | None = None
#     tags: list
   
#définition du schéma/point de terminaison (API)
@app.route('/', methods=['GET', 'POST'])
def home():
   request_type_str = request.method
   if request_type_str == 'GET' :
      return render_template('index.html')
   else:
      message = request.form['message']
      title = request.form['title']
      question =  concatenation(message, title, 5)
      question = clean_data(question)
      question = preprocess_texte(question)
      question = pd.Series(question)
      model.predict(question)




if __name__ == "__main__":
    app.run(debug=True)

def remove_URL(text):
    """
        Supprime les URL présentes dans les questions
        param text : le texte à nettoyer
        return : le texte sans URL
    """

    return re.sub(r"https?://\S+|www\.\S+", "", text)
	
def remove_html(text):
    """
        Supprime le HTML présent dans les questions
        param text : le texte à nettoyer
        return : le texte sans HTML
    """

    text = text.replace('\n', '')
    html = re.compile(r'<.*?>')
    return re.sub(html, "", text)
	
def remove_non_ascii(text):
    """
        Supprime le caractères non ascii présent dans les questions
        param text : le texte à nettoyer
        return : le texte sans caractères non ascii
    """

    return re.sub(r'[^\x00-\x7f]',r'', text)


def remove_digit(text):
    """
        Supprime les nombres et chiffres présents dans les questions
        param text : le texte à nettoyer
        return : le texte sans nombres et chiffres
    """

    return re.sub(r'[0-9]', '', text)
	
def remove_punct(text):
    """
        Supprime la ponctuation
        param text : le texte à nettoyer
        return : le texte sans ponctuation
    """
    
    translating = str.maketrans(string.punctuation, ' '*len(string.punctuation)) 
    new_string = text.translate(translating)

    return new_string

def clean_tag(tag):
    """
      Supprime les balises présentes dans les tags
      param text : le tag à nettoyer
      return : le tag sans balises
    """
    tag =  tag.str.replace('>', ' ')
    tag =  tag.str.replace('<', ' ')
    return tag

def remove_code(text):
    """
        Supprime les parties de codes présente dans les messages
        param text : le texte à nettoyer
        return : le texte sans partie de code
    """
    return  re.sub('<code>.*?</code>', '', text, flags=re.DOTALL)

def extract_pre(text, ponderation=5):
    """
        Récupère l'attribut class de la balise pre
        et le concatène à la fin des questions
        param text : le texte où l'attibut et à extraire
        param ponderation  le nombre de fois où le contenu de la balise doit
        être concaténé
        return : la question avec la balise concaténée 
    """
    classes = []
    soup= bs(text, 'html.parser')
    pre =  soup.find_all(class_=True)
    for element  in pre:
      try:
        classes.extend(element["class"])
      except KeyError as e:
        continue # evite une erreur si la balise n'est pas trouvée dans une question

    list_classes =' '.join(classes)
    concatenate = text + " "  + list_classes
    for i in range(0, ponderation):
      concatenate += " " + list_classes

    return concatenate 

def clean_data(texte):
    """
        Nettoie le texte passé en paramètres : extraction de la balise pre, 
        suppression des parties de code, des url, du html, des carac non ascii,
        de la ponctuation et des chiffres. Passage en minuscules de tous les mots.
        param text : le texte à nettoyer
        return : le texte nettoyé (sous forme de string)
    """
   
    word_clean = extract_pre(texte)
    word_clean = remove_code(word_clean)
    word_clean = remove_URL(word_clean)
    word_clean = remove_html(word_clean)
    word_clean = remove_non_ascii(word_clean)
    word_clean = remove_punct(word_clean)
    word_clean = remove_digit(word_clean)
    word_clean = word_clean.lower()
 
    clean_texte = ' '.join(word_clean)
    return word_clean
	
def preprocess_texte(sentence):
    """
        Prépare le texte passé en paramètre pour les modèles de machine learning
        :  tokenization et suppression des stop words
        param text : le texte (nettoyé)
        return : une liste de tokens
    """
    wst = WhitespaceTokenizer()
    sentence_tokenized =  wst.tokenize(sentence)
    stop_words = nltk.corpus.stopwords.words('english')
    stop_words.extend(['lt', 'im', 'lang', 'prettyprint', 'would', 'dtype object', 'object dtype', 'make', 'seem', 'see','yet','quot', 'gt','code', 'strong', 'https', 'using', 'href', 'rel', 'noreferrer', 'error', 'use', 'want', 'file', 'way', 'em', 'stack', 'imgur', 'following', 'tried', 'one', 'trying', 'png', 'app', 'need', 'data', 'know', 'work', 'problem', 'will', 'example', 'run', 'image', 'function', 'src', 'project', 'new', 'now', 'something', 'set', 'find'])
    word_list = [word for word in sentence_tokenized if word not in stop_words]
    return word_list

def concatenation(message, title, ponderation):
  """
        Concatène 2 series pandas avec une pondération
        param pdSeries_original : la série à laquelle on va rajouter une concaténation
        param pdSeries_to_concatenate : la série à concaténer
        param ponderation : le nombre de fois où l'on rajoute la série à concaténer
        return : le résultat de la concaténation
   """
  for i in range(0, ponderation):
    message +=  " " + title
    
  return message